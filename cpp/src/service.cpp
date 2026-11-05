#include "faceid/service.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <spdlog/fmt/fmt.h>
#include <spdlog/spdlog.h>
#include <limits>
#include <stdexcept>
#include <thread>
#include <utility>

#include <filesystem>

namespace faceid {
namespace {

const int64_t kShutdownTimeoutMs = 30000;

std::string natsError(natsStatus status) {
  return natsStatus_GetText(status);
}

void checkNats(natsStatus status, const std::string& operation) {
  if (status != NATS_OK) {
    throw std::runtime_error(operation + ": " + natsError(status));
  }
}

// Summarise image metadata without the pixel/encoded byte payload.
std::string describeImage(const ImageRequest& image) {
  return fmt::format(
      "image={{{}x{} encoding={} preprocessed={} bytes={} keypoints={}}}",
      image.width, image.height,
      faceid::fbs::EnumNameImageEncoding(image.encoding), image.is_preprocessed,
      image.bytes.size(), image.keypoints.size());
}

std::string describeCandidates(const std::vector<Candidate>& candidates) {
  std::string result = "[";
  for (size_t index = 0; index < candidates.size(); ++index) {
    const Candidate& candidate = candidates[index];
    result += fmt::format("{}{{user_id={} name={} confidence={:.4f} "
                          "raw_distance={:.4f}}}",
                          index == 0 ? "" : ", ", candidate.user_id,
                          candidate.name, candidate.confidence,
                          candidate.raw_distance);
  }
  return result + "]";
}

int64_t currentTimeUnixMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

}  // namespace

FaceIdService::FaceIdService(const AppConfig& config, EmbeddingModel& model,
                             VectorStore& store)
    : config_(config), model_(model), store_(store), connection_(nullptr) {}

void FaceIdService::run(const volatile std::sig_atomic_t& stop_requested) {
  try {
    connect();
    subscribe(config_.nats.enrol_subject, Operation::Enrol);
    subscribe(config_.nats.query_subject, Operation::Query);
    subscribe(config_.nats.user_info_subject, Operation::UserInfo);
    subscribe(config_.nats.delete_subject, Operation::DeleteUser);
    subscribe(config_.nats.debug_subject, Operation::Debug);
    checkNats(natsConnection_FlushTimeout(connection_, 5000),
              "Flush NATS subscriptions");
    spdlog::info("Face identification service is listening on NATS");
    while (stop_requested == 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    spdlog::info("Stop requested; shutting down (press Ctrl+C again to force)");
  } catch (...) {
    shutdown();
    throw;
  }
  shutdown();
}

void FaceIdService::connect() {
  natsOptions* options = nullptr;
  checkNats(natsOptions_Create(&options), "Create NATS options");
  try {
    checkNats(natsOptions_SetURL(options, config_.nats.url.c_str()),
              "Set NATS URL");
    checkNats(natsOptions_SetName(options, "imx-faceid"), "Set NATS client name");
    if (!config_.nats.credentials_file.empty()) {
      checkNats(natsOptions_SetUserCredentialsFromFiles(
                    options, config_.nats.credentials_file.c_str(), nullptr),
                "Set NATS credentials file");
    } else if (!config_.nats.token.empty()) {
      checkNats(natsOptions_SetToken(options, config_.nats.token.c_str()),
                "Set NATS token");
    } else if (!config_.nats.username.empty()) {
      checkNats(natsOptions_SetUserInfo(options, config_.nats.username.c_str(),
                                        config_.nats.password.c_str()),
                "Set NATS username and password");
    }
    const natsStatus status = natsConnection_Connect(&connection_, options);
    natsOptions_Destroy(options);
    options = nullptr;
    checkNats(status, "Connect to NATS at " + config_.nats.url);
  } catch (...) {
    if (options != nullptr) {
      natsOptions_Destroy(options);
    }
    throw;
  }
}

void FaceIdService::subscribe(const std::string& subject,
                              Operation operation) {
  std::unique_ptr<SubscriptionBinding> binding(new SubscriptionBinding());
  binding->service = this;
  binding->operation = operation;
  binding->subscription = nullptr;
  const natsStatus status = natsConnection_Subscribe(
      &binding->subscription, connection_, subject.c_str(), &FaceIdService::onMessage,
      binding.get());
  checkNats(status, "Subscribe to NATS subject " + subject);
  subscriptions_.push_back(std::move(binding));
}

void FaceIdService::onMessage(natsConnection*, natsSubscription*, natsMsg* message,
                              void* closure) {
  std::unique_ptr<natsMsg, void (*)(natsMsg*)> owned_message(message,
                                                            &natsMsg_Destroy);
  if (message == nullptr || closure == nullptr) {
    return;
  }
  SubscriptionBinding* binding = static_cast<SubscriptionBinding*>(closure);
  binding->service->dispatch(binding->operation, message);
}

void FaceIdService::dispatch(Operation operation, natsMsg* message) {
  const char* subject = natsMsg_GetSubject(message);
  spdlog::debug("NATS request subject={}", subject == nullptr ? "" : subject);
  spdlog::trace("NATS rx subject={} payload_bytes={}",
                subject == nullptr ? "" : subject,
                natsMsg_GetDataLength(message));
  switch (operation) {
    case Operation::Enrol:
      handleEnrol(message);
      return;
    case Operation::Query:
      handleQuery(message);
      return;
    case Operation::UserInfo:
      handleUserInfo(message);
      return;
    case Operation::DeleteUser:
      handleDeleteUser(message);
      return;
    case Operation::Debug:
      handleDebug(message);
      return;
  }
}

void FaceIdService::handleEnrol(natsMsg* message) {
  std::string user_id;
  std::string reply_to;
  try {
    const std::vector<uint8_t> data = messageData(message);
    const EnrolRequest request = parseEnrolRequest(data.data(), data.size());
    user_id = request.user_id;
    reply_to = request.reply_to;
    spdlog::trace(
        "NATS rx Enrol user_id={} name={} reply_to={} time_unix_ms={} "
        "custom_bytes={} {}",
        request.user_id, request.name, request.reply_to, request.time_unix_ms,
        request.custom.size(), describeImage(request.image));
    std::vector<float> embedding;
    InferenceTiming timing;
    {
      std::lock_guard<std::mutex> lock(inference_mutex_);
      embedding = model_.embed(request.image, &timing);
    }
    spdlog::info(
        "Inference enrol user_id={} runtime={} preprocess_ms={:.2f} "
        "inference_ms={:.2f}",
        user_id, model_.runtimeName(), timing.preprocess_ms,
        timing.inference_ms);
    const int64_t time_unix_ms =
        request.time_unix_ms == 0 ? currentTimeUnixMs() : request.time_unix_ms;
    const std::shared_ptr<std::mutex> user_lock = userMutex(user_id);
    {
      std::lock_guard<std::mutex> lock(*user_lock);
      store_.enroll(user_id, request.name, request.custom, embedding,
                    request.image.bytes, time_unix_ms);
    }
    spdlog::trace("NATS tx EnrolResult reply_to={} user_id={} success=true",
                  reply_to, user_id);
    publish(reply_to, makeEnrolResult(user_id, true));
  } catch (const std::exception& error) {
    logFailure("Enrol request", user_id, error);
    spdlog::trace(
        "NATS tx EnrolResult reply_to={} user_id={} success=false reason={}",
        reply_to, user_id, error.what());
    publish(reply_to, makeEnrolResult(user_id, false, error.what()));
  }
}

void FaceIdService::handleQuery(natsMsg* message) {
  std::string track_id;
  std::string reply_to;
  try {
    const std::vector<uint8_t> data = messageData(message);
    const QueryRequest request = parseQueryRequest(data.data(), data.size());
    track_id = request.track_id;
    reply_to = request.reply_to;
    spdlog::trace(
        "NATS rx IdRequest track_id={} reply_to={} min_confidence={} "
        "max_results={} pts={} time_unix_ms={} custom_bytes={} {}",
        request.track_id, request.reply_to, request.min_confidence,
        request.max_results, request.pts, request.time_unix_ms,
        request.custom.size(), describeImage(request.image));
    if (request.max_results == 0) {
      throw std::invalid_argument("max_results must be at least one");
    }
    if (!std::isfinite(request.min_confidence) ||
      request.min_confidence < 0.0f || request.min_confidence > 1.0f) {
      throw std::invalid_argument("min_confidence must be in the range 0..1");
    }
    std::vector<float> embedding;
    InferenceTiming timing;
    {
      std::lock_guard<std::mutex> lock(inference_mutex_);
      embedding = model_.embed(request.image, &timing);
    }
    spdlog::info(
        "Inference query track_id={} runtime={} preprocess_ms={:.2f} "
        "inference_ms={:.2f}",
        track_id, model_.runtimeName(), timing.preprocess_ms,
        timing.inference_ms);
    const std::vector<Candidate> candidates =
        store_.query(embedding, request.max_results, request.min_confidence);
    if (spdlog::should_log(spdlog::level::trace)) {
      spdlog::trace(
          "NATS tx IdResult reply_to={} track_id={} success=true "
          "candidates={} {}",
          reply_to, track_id, candidates.size(), describeCandidates(candidates));
    }
    publish(reply_to, makeQueryResult(track_id, candidates, true));
  } catch (const std::exception& error) {
    logFailure("Identity query", track_id, error);
    spdlog::trace(
        "NATS tx IdResult reply_to={} track_id={} success=false reason={}",
        reply_to, track_id, error.what());
    publish(reply_to, makeQueryResult(track_id, std::vector<Candidate>(), false,
                                      error.what()));
  }
}

void FaceIdService::handleUserInfo(natsMsg* message) {
  std::string user_id;
  std::string reply_to;
  try {
    const std::vector<uint8_t> data = messageData(message);
    const UserInfoRequest request = parseUserInfoRequest(data.data(), data.size());
    user_id = request.user_id;
    reply_to = request.reply_to;
    spdlog::trace("NATS rx UserInfo user_id={} reply_to={} time_unix_ms={}",
                  request.user_id, request.reply_to, request.time_unix_ms);
    const std::vector<EnrolmentRecord> records = store_.userInfo(user_id);
    if (spdlog::should_log(spdlog::level::trace)) {
      std::string timestamps;
      for (const EnrolmentRecord& record : records) {
        timestamps += (timestamps.empty() ? "" : ",") +
                      std::to_string(record.time_unix_ms);
      }
      spdlog::trace(
          "NATS tx UserInfoResponse reply_to={} user_id={} success=true "
          "records={} embedding_dim={} time_unix_ms=[{}]",
          reply_to, user_id, records.size(),
          records.empty() ? 0 : records.front().embedding.size(), timestamps);
    }
    publish(reply_to, makeUserInfoResponse(user_id, records, true));
  } catch (const std::exception& error) {
    logFailure("User info request", user_id, error);
    spdlog::trace(
        "NATS tx UserInfoResponse reply_to={} user_id={} success=false "
        "reason={}",
        reply_to, user_id, error.what());
    publish(reply_to, makeUserInfoResponse(
                          user_id, std::vector<EnrolmentRecord>(), false,
                          error.what()));
  }
}

void FaceIdService::handleDeleteUser(natsMsg* message) {
  std::string user_id;
  std::string reply_to;
  try {
    const std::vector<uint8_t> data = messageData(message);
    const DeleteRequest request = parseDeleteRequest(data.data(), data.size());
    user_id = request.user_id;
    reply_to = request.reply_to;
    spdlog::trace("NATS rx DeleteUser user_id={} reply_to={} time_unix_ms={}",
                  request.user_id, request.reply_to, request.time_unix_ms);
    const std::shared_ptr<std::mutex> user_lock = userMutex(user_id);
    {
      std::lock_guard<std::mutex> lock(*user_lock);
      store_.deleteUser(user_id);
    }
    spdlog::trace(
        "NATS tx DeleteUserResponse reply_to={} user_id={} success=true",
        reply_to, user_id);
    publish(reply_to, makeDeleteResponse(user_id, true));
  } catch (const std::exception& error) {
    logFailure("User deletion", user_id, error);
    spdlog::trace(
        "NATS tx DeleteUserResponse reply_to={} user_id={} success=false "
        "reason={}",
        reply_to, user_id, error.what());
    publish(reply_to, makeDeleteResponse(user_id, false, error.what()));
  }
}

void FaceIdService::handleDebug(natsMsg* message) {
  std::string reply_to;
  try {
    const std::vector<uint8_t> data = messageData(message);
    const DebugRequest request = parseDebugRequest(data.data(), data.size());
    reply_to = request.reply_to;
    spdlog::trace("NATS rx DebugInfo reply_to={} time_unix_ms={}",
                  request.reply_to, request.time_unix_ms);
    const std::pair<std::string, std::string> database = store_.debugInfo();
    const std::string model_name =
        std::filesystem::path(config_.model.path).filename().string();
    spdlog::trace(
        "NATS tx DebugInfoResponse reply_to={} success=true database={} "
        "database_version={} inference_backend={} model={} embedding_dim={} "
        "nats_url={}",
        reply_to, database.first, database.second, model_.runtimeName(),
        model_name, model_.embeddingDimension(), config_.nats.url);
    publish(reply_to,
            makeDebugResponse(database.first, database.second,
                              model_.runtimeName(), model_name,
                              static_cast<uint32_t>(model_.embeddingDimension()),
                              config_.nats.url, true));
  } catch (const std::exception& error) {
    logFailure("Debug info request", std::string(), error);
    const std::string model_name =
        std::filesystem::path(config_.model.path).filename().string();
    spdlog::trace(
        "NATS tx DebugInfoResponse reply_to={} success=false reason={}",
        reply_to, error.what());
    publish(reply_to,
            makeDebugResponse("qdrant", std::string(), model_.runtimeName(),
                              model_name,
                              static_cast<uint32_t>(model_.embeddingDimension()),
                              config_.nats.url, false, error.what()));
  }
}

void FaceIdService::publish(const std::string& reply_to,
                            const std::vector<uint8_t>& payload) {
  if (reply_to.empty()) {
    spdlog::trace("NATS tx skipped: request has no reply_to");
    return;
  }
  if (payload.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
    spdlog::error("NATS response is too large for libnats");
    return;
  }
  spdlog::debug("NATS response subject={} bytes={}", reply_to, payload.size());
  const natsStatus status = natsConnection_Publish(
      connection_, reply_to.c_str(), payload.data(),
      static_cast<int>(payload.size()));
  if (status != NATS_OK) {
    spdlog::error("Publish response to {} failed: {}", reply_to, natsError(status));
  }
}

void FaceIdService::logFailure(const std::string& operation,
                               const std::string& identity,
                               const std::exception& error) const {
  if (!identity.empty()) {
    spdlog::error("{} failed for {}: {}", operation, identity, error.what());
  } else {
    spdlog::error("{} failed: {}", operation, error.what());
  }
}

std::shared_ptr<std::mutex> FaceIdService::userMutex(
    const std::string& user_id) {
  std::lock_guard<std::mutex> lock(user_mutex_map_mutex_);
  std::shared_ptr<std::mutex>& user_lock = user_mutexes_[user_id];
  if (!user_lock) {
    user_lock = std::make_shared<std::mutex>();
  }
  return user_lock;
}

std::vector<uint8_t> FaceIdService::messageData(const natsMsg* message) const {
  const char* data = natsMsg_GetData(message);
  const int length = natsMsg_GetDataLength(message);
  if (length <= 0 || data == nullptr) {
    return std::vector<uint8_t>();
  }
  const uint8_t* begin = reinterpret_cast<const uint8_t*>(data);
  return std::vector<uint8_t>(begin, begin + length);
}

void FaceIdService::shutdown() {
  // Draining or flushing a disconnected connection only waits for timeouts.
  const bool connected =
      connection_ != nullptr &&
      natsConnection_Status(connection_) == NATS_CONN_STATUS_CONNECTED;
  if (connection_ != nullptr && !connected) {
    spdlog::warn("NATS is not connected; skipping drain and flush");
  }
  if (connected) {
    for (const std::unique_ptr<SubscriptionBinding>& binding : subscriptions_) {
      if (binding->subscription != nullptr) {
        const natsStatus status = natsSubscription_DrainTimeout(
            binding->subscription, kShutdownTimeoutMs);
        if (status != NATS_OK) {
          spdlog::error("Drain NATS subscription failed: {}", natsError(status));
        }
      }
    }
  }
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(kShutdownTimeoutMs);
  for (const std::unique_ptr<SubscriptionBinding>& binding : subscriptions_) {
    if (binding->subscription != nullptr) {
      if (connected) {
        const int64_t remaining_ms = std::max<int64_t>(
            1, std::chrono::duration_cast<std::chrono::milliseconds>(
                   deadline - std::chrono::steady_clock::now())
                   .count());
        const natsStatus status = natsSubscription_WaitForDrainCompletion(
            binding->subscription, remaining_ms);
        if (status != NATS_OK) {
          spdlog::error("Wait for NATS subscription drain failed: {}",
                        natsError(status));
        }
      }
      natsSubscription_Destroy(binding->subscription);
      binding->subscription = nullptr;
    }
  }
  if (connection_ != nullptr) {
    if (natsConnection_Status(connection_) == NATS_CONN_STATUS_CONNECTED) {
      const natsStatus flush_status =
          natsConnection_FlushTimeout(connection_, 5000);
      if (flush_status != NATS_OK) {
        spdlog::error("Flush NATS responses failed: {}", natsError(flush_status));
      }
    }
    natsConnection_Close(connection_);
    natsConnection_Destroy(connection_);
    connection_ = nullptr;
  }
  subscriptions_.clear();
}

}  // namespace faceid