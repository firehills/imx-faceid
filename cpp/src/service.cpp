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

/** Convert a libnats status into its human-readable diagnostic. */
std::string natsError(natsStatus status) {
  return natsStatus_GetText(status);
}

/**
 * Turn a failed libnats operation into an exception with operation context.
 * @param status Result returned by libnats.
 * @param operation Description of the connection/subscription operation.
 */
void checkNats(natsStatus status, const std::string& operation) {
  if (status != NATS_OK) {
    throw std::runtime_error(operation + ": " + natsError(status));
  }
}

/**
 * Format request image metadata while deliberately excluding image contents.
 * @param image Parsed image dimensions, encoding, flags, bytes, and landmarks.
 * @return Compact trace-level description safe from dumping pixel data.
 */
std::string describeImage(const ImageRequest& image) {
  return fmt::format(
      "image={{{}x{} encoding={} preprocessed={} bytes={} keypoints={}}}",
      image.width, image.height,
      faceid::fbs::EnumNameImageEncoding(image.encoding), image.is_preprocessed,
      image.bytes.size(), image.keypoints.size());
}

/**
 * Format query matches with identity and score fields, but no embedding vectors.
 * @param candidates Results returned by the vector store.
 * @return Human-readable candidate summaries for trace diagnostics.
 */
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

/** Return the current wall-clock time in milliseconds since the Unix epoch. */
int64_t currentTimeUnixMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

}  // namespace

/**
 * Bind validated settings and service dependencies without starting I/O.
 * @param config Application and NATS configuration; must outlive the service.
 * @param model Initialized inference model; shared by message handlers.
 * @param store Connected Qdrant-backed vector store.
 */
FaceIdService::FaceIdService(const AppConfig& config, EmbeddingModel& model,
                             VectorStore& store)
    : config_(config), model_(model), store_(store), connection_(nullptr) {}

/**
 * Connect, register request handlers, then block until a signal requests stop.
 *
 * Incoming NATS callbacks route through the subscriptions created here. On
 * normal stop or startup/registration failure, the same shutdown path drains
 * subscriptions when possible and releases this service's connection.
 * @param stop_requested Signal-safe flag set by the process signal handler.
 */
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

/**
 * Create an authenticated libnats connection from the validated NATS settings.
 *
 * Credentials-file, token, or username/password authentication is selected
 * exclusively according to the parsed config. The temporary options object is
 * destroyed on both successful connection and failure.
 */
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

/**
 * Bind one configured subject to a service operation callback.
 * @param subject NATS subject to subscribe to.
 * @param operation Request handler selected for messages on that subject.
 */
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

/**
 * Own an asynchronous NATS message and dispatch it using subscription metadata.
 * @param message Delivered request; ownership is released after callback returns.
 * @param closure SubscriptionBinding installed by subscribe().
 */
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

/**
 * Log and route a received message to the operation-specific request handler.
 * @param operation Handler selected when its subject was subscribed.
 * @param message NATS envelope payload to decode and process.
 */
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

/**
 * Decode an enrol request, generate its embedding, update Qdrant, and reply.
 *
 * The request identity lock serializes updates to the same user, while the
 * inference lock protects the backend. Any standard exception is logged and
 * converted into a failure result when a reply subject was supplied.
 * @param message Incoming enrolment envelope.
 */
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

/**
 * Decode an identity query, infer an embedding, search Qdrant, and reply.
 * @param message Incoming query envelope with track ID, image, and thresholds.
 * Invalid thresholds and downstream failures become failure responses.
 */
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

/**
 * Fetch a user's raw enrolment metadata and vectors, then serialize a reply.
 * @param message Incoming user-info envelope.
 * Embedding contents remain in the protocol response and are never logged.
 */
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

/**
 * Delete all stored enrolments for one user while holding that user's lock.
 * @param message Incoming delete-user envelope.
 * Store errors are converted to a failure response when reply_to is present.
 */
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

/**
 * Collect Qdrant/runtime/model diagnostics and return them to the caller.
 * @param message Incoming debug-info envelope.
 * On failure, return the available runtime details with the error reason.
 */
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

/**
 * Publish a serialized response to its requested NATS subject.
 * @param reply_to Response subject from the request; empty means no reply.
 * @param payload Complete serialized FACE response envelope.
 * Oversized responses are rejected before conversion to libnats' int length.
 */
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

/**
 * Report a failed request operation without throwing from its callback.
 * @param operation Human-readable operation name.
 * @param identity Optional user or track identifier associated with the failure.
 * @param error Exception caught by the request handler.
 */
void FaceIdService::logFailure(const std::string& operation,
                               const std::string& identity,
                               const std::exception& error) const {
  if (!identity.empty()) {
    spdlog::error("{} failed for {}: {}", operation, identity, error.what());
  } else {
    spdlog::error("{} failed: {}", operation, error.what());
  }
}

/**
 * Get or create a mutex dedicated to one user's Qdrant mutations.
 * @param user_id User identifier used as the lock-map key.
 * @return Shared lock object retained in the service's per-user map.
 */
std::shared_ptr<std::mutex> FaceIdService::userMutex(
    const std::string& user_id) {
  std::lock_guard<std::mutex> lock(user_mutex_map_mutex_);
  std::shared_ptr<std::mutex>& user_lock = user_mutexes_[user_id];
  if (!user_lock) {
    user_lock = std::make_shared<std::mutex>();
  }
  return user_lock;
}

/**
 * Copy the opaque payload bytes from a libnats message into owned storage.
 * @param message NATS message whose data remains valid during this call.
 * @return Payload bytes, or an empty vector for absent/empty data.
 */
std::vector<uint8_t> FaceIdService::messageData(const natsMsg* message) const {
  const char* data = natsMsg_GetData(message);
  const int length = natsMsg_GetDataLength(message);
  if (length <= 0 || data == nullptr) {
    return std::vector<uint8_t>();
  }
  const uint8_t* begin = reinterpret_cast<const uint8_t*>(data);
  return std::vector<uint8_t>(begin, begin + length);
}

/**
 * Drain and destroy this service's NATS subscriptions and connection.
 *
 * Connected subscriptions share a single bounded drain deadline. If the
 * connection has dropped, drain and flush calls are skipped to avoid waiting
 * through their timeouts; the local client connection is still closed.
 */
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