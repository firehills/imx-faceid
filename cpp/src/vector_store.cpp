#include "faceid/vector_store.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <grpcpp/grpcpp.h>

#include "collections_service.grpc.pb.h"
#include "points_service.grpc.pb.h"
#include "qdrant_health.grpc.pb.h"

namespace faceid {
namespace {

const size_t kScrollPageSize = 256;
const char kBase64Alphabet[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

std::string makeUuid() {
  static thread_local std::mt19937_64 generator([]() {
    std::random_device random;
    std::seed_seq seed{random(), random(), random(), random(), random(), random()};
    return std::mt19937_64(seed);
  }());

  std::array<uint8_t, 16> bytes;
  const uint64_t high = generator();
  const uint64_t low = generator();
  for (size_t index = 0; index < 8; ++index) {
    bytes[index] = static_cast<uint8_t>(high >> ((7 - index) * 8));
    bytes[index + 8] = static_cast<uint8_t>(low >> ((7 - index) * 8));
  }
  bytes[6] = static_cast<uint8_t>((bytes[6] & 0x0f) | 0x40);
  bytes[8] = static_cast<uint8_t>((bytes[8] & 0x3f) | 0x80);

  std::ostringstream result;
  result << std::hex << std::setfill('0');
  for (size_t index = 0; index < bytes.size(); ++index) {
    if (index == 4 || index == 6 || index == 8 || index == 10) {
      result << '-';
    }
    result << std::setw(2) << static_cast<unsigned int>(bytes[index]);
  }
  return result.str();
}

std::string encodeBase64(const std::vector<uint8_t>& bytes) {
  std::string encoded;
  encoded.reserve(((bytes.size() + 2) / 3) * 4);
  for (size_t index = 0; index < bytes.size(); index += 3) {
    const uint32_t first = bytes[index];
    const uint32_t second = index + 1 < bytes.size() ? bytes[index + 1] : 0;
    const uint32_t third = index + 2 < bytes.size() ? bytes[index + 2] : 0;
    const uint32_t value = (first << 16) | (second << 8) | third;
    encoded.push_back(kBase64Alphabet[(value >> 18) & 0x3f]);
    encoded.push_back(kBase64Alphabet[(value >> 12) & 0x3f]);
    encoded.push_back(index + 1 < bytes.size()
                          ? kBase64Alphabet[(value >> 6) & 0x3f]
                          : '=');
    encoded.push_back(index + 2 < bytes.size() ? kBase64Alphabet[value & 0x3f]
                                               : '=');
  }
  return encoded;
}

void setStringPayload(qdrant::PointStruct* point, const std::string& key,
                      const std::string& value) {
  (*point->mutable_payload())[key].set_string_value(value);
}

void setIntegerPayload(qdrant::PointStruct* point, const std::string& key,
                       int64_t value) {
  (*point->mutable_payload())[key].set_integer_value(value);
}

qdrant::Filter userFilter(const std::string& user_id) {
  qdrant::Filter filter;
  qdrant::FieldCondition* field =
      filter.add_must()->mutable_field();
  field->set_key("user_id");
  field->mutable_match()->set_keyword(user_id);
  return filter;
}

std::string stringPayload(const google::protobuf::Map<std::string, qdrant::Value>&
                              payload,
                          const std::string& key, bool required) {
  const auto found = payload.find(key);
  if (found == payload.end() ||
      found->second.kind_case() != qdrant::Value::kStringValue) {
    if (required) {
      throw std::runtime_error("Qdrant point is missing string payload '" + key +
                               "'");
    }
    return std::string();
  }
  return found->second.string_value();
}

int64_t integerPayload(const google::protobuf::Map<std::string, qdrant::Value>&
                           payload,
                       const std::string& key) {
  const auto found = payload.find(key);
  if (found == payload.end()) {
    return 0;
  }
  if (found->second.kind_case() == qdrant::Value::kIntegerValue) {
    return found->second.integer_value();
  }
  if (found->second.kind_case() == qdrant::Value::kDoubleValue) {
    return static_cast<int64_t>(found->second.double_value());
  }
  return 0;
}

std::runtime_error rpcError(const std::string& operation,
                            const grpc::Status& status) {
  std::ostringstream message;
  message << "Qdrant " << operation << " failed (code " << status.error_code()
          << "): " << status.error_message();
  return std::runtime_error(message.str());
}

}  // namespace

class VectorStore::Impl {
 public:
  Impl(const QdrantConfig& config, size_t embedding_dim)
      : config_(config), embedding_dim_(embedding_dim) {
    const std::string target = config.host + ":" + std::to_string(config.grpc_port);
    std::shared_ptr<grpc::ChannelCredentials> credentials =
        config.https ? grpc::SslCredentials(grpc::SslCredentialsOptions())
                     : grpc::InsecureChannelCredentials();
    channel_ = grpc::CreateChannel(target, credentials);
    collections_ = qdrant::Collections::NewStub(channel_);
    points_ = qdrant::Points::NewStub(channel_);
    health_ = qdrant::Qdrant::NewStub(channel_);
    ensureCollection(config_.raw_collection);
    ensureCollection(config_.normalized_collection);
  }

  void enroll(const std::string& user_id, const std::string& name,
              const std::string& custom, const std::vector<float>& embedding,
              const std::vector<uint8_t>& image_bytes, int64_t time_unix_ms) {
    validateEmbedding(embedding, "Enrolment");

    qdrant::UpsertPoints raw_request;
    raw_request.set_collection_name(config_.raw_collection);
    raw_request.set_wait(true);
    qdrant::PointStruct* raw_point = raw_request.add_points();
    raw_point->mutable_id()->set_uuid(makeUuid());
    setStringPayload(raw_point, "user_id", user_id);
    setStringPayload(raw_point, "name", name);
    setStringPayload(raw_point, "custom", custom);
    setIntegerPayload(raw_point, "time_unix_ms", time_unix_ms);
    setStringPayload(raw_point, "image", encodeBase64(image_bytes));
    setVector(raw_point, embedding);
    upsert(raw_request, "raw enrolment");

    std::vector<qdrant::RetrievedPoint> raw_points = scrollUser(user_id);
    if (raw_points.empty()) {
      throw std::runtime_error("Qdrant returned no enrolments for user '" +
                               user_id + "'");
    }

    bool update_names = false;
    std::vector<std::vector<float> > vectors;
    vectors.reserve(raw_points.size());
    for (const qdrant::RetrievedPoint& point : raw_points) {
      if (stringPayload(point.payload(), "name", false) != name) {
        update_names = true;
      }
      vectors.push_back(pointVector(point));
    }
    if (update_names) {
      setNameForPoints(raw_points, name);
    }

    std::vector<float> mean(embedding_dim_, 0.0f);
    for (const std::vector<float>& vector : vectors) {
      for (size_t index = 0; index < embedding_dim_; ++index) {
        mean[index] += vector[index] / static_cast<float>(vectors.size());
      }
    }
    float squared_norm = 0.0f;
    for (float value : mean) {
      squared_norm += value * value;
    }
    const float norm = std::sqrt(squared_norm);
    if (!std::isfinite(norm) || norm == 0.0f) {
      throw std::runtime_error(
          "The normalized mean of this user's enrolments is zero");
    }
    for (float& value : mean) {
      value /= norm;
    }

    deleteByUser(config_.normalized_collection, user_id);
    qdrant::UpsertPoints normalized_request;
    normalized_request.set_collection_name(config_.normalized_collection);
    normalized_request.set_wait(true);
    qdrant::PointStruct* normalized_point = normalized_request.add_points();
    normalized_point->mutable_id()->set_uuid(makeUuid());
    setStringPayload(normalized_point, "user_id", user_id);
    setStringPayload(normalized_point, "name", name);
    setIntegerPayload(normalized_point, "num_enrolments",
                      static_cast<int64_t>(vectors.size()));
    setIntegerPayload(normalized_point, "time_unix_ms", time_unix_ms);
    setVector(normalized_point, mean);
    upsert(normalized_request, "normalized enrolment");
  }

  std::vector<Candidate> query(const std::vector<float>& embedding,
                               uint32_t max_results,
                               float min_confidence) {
    if (max_results == 0) {
      return std::vector<Candidate>();
    }
    validateEmbedding(embedding, "Query");

    qdrant::QueryPoints request;
    request.set_collection_name(config_.normalized_collection);
    request.set_limit(max_results);
    request.mutable_with_payload()->set_enable(true);
    qdrant::DenseVector* dense = request.mutable_query()
                                    ->mutable_nearest()
                                    ->mutable_dense();
    for (float value : embedding) {
      dense->add_data(value);
    }

    qdrant::QueryResponse response;
    grpc::ClientContext context;
    addAuth(context);
    const grpc::Status status = points_->Query(&context, request, &response);
    if (!status.ok()) {
      throw rpcError("query", status);
    }

    std::vector<Candidate> matches;
    matches.reserve(static_cast<size_t>(response.result_size()));
    for (const qdrant::ScoredPoint& point : response.result()) {
      if (point.payload().empty()) {
        throw std::runtime_error("Normalized enrolment is missing its payload");
      }
      const std::pair<float, float> score =
          scoreToDistanceConfidence(point.score(), config_.metric);
      if (score.second < min_confidence) {
        continue;
      }
      Candidate candidate;
      candidate.user_id = stringPayload(point.payload(), "user_id", true);
      candidate.name = stringPayload(point.payload(), "name", false);
      candidate.raw_distance = score.first;
      candidate.confidence = score.second;
      matches.push_back(candidate);
    }
    return matches;
  }

  std::vector<EnrolmentRecord> userInfo(const std::string& user_id) {
    const std::vector<qdrant::RetrievedPoint> points = scrollUser(user_id);
    std::vector<EnrolmentRecord> records;
    records.reserve(points.size());
    for (const qdrant::RetrievedPoint& point : points) {
      EnrolmentRecord record;
      record.embedding = pointVector(point);
      record.time_unix_ms = integerPayload(point.payload(), "time_unix_ms");
      records.push_back(record);
    }
    return records;
  }

  void deleteUser(const std::string& user_id) {
    deleteByUser(config_.raw_collection, user_id);
    deleteByUser(config_.normalized_collection, user_id);
  }

  std::pair<std::string, std::string> debugInfo() {
    qdrant::HealthCheckRequest request;
    qdrant::HealthCheckReply response;
    grpc::ClientContext context;
    addAuth(context);
    const grpc::Status status = health_->HealthCheck(&context, request, &response);
    if (!status.ok()) {
      throw rpcError("health check", status);
    }
    return std::make_pair("qdrant", response.version());
  }

 private:
  void addAuth(grpc::ClientContext& context) const {
    if (!config_.api_key.empty()) {
      context.AddMetadata("api-key", config_.api_key);
    }
  }

  void validateEmbedding(const std::vector<float>& embedding,
                         const std::string& operation) const {
    if (embedding.size() != embedding_dim_) {
      std::ostringstream message;
      message << operation << " embedding size " << embedding.size()
              << " does not match collection size " << embedding_dim_;
      throw std::invalid_argument(message.str());
    }
  }

  qdrant::Distance distance() const {
    if (config_.metric == "cosine") {
      return qdrant::Cosine;
    }
    if (config_.metric == "euclidean") {
      return qdrant::Euclid;
    }
    throw std::invalid_argument("Unsupported distance metric: " +
                                config_.metric);
  }

  void ensureCollection(const std::string& name) {
    qdrant::CollectionExistsRequest exists_request;
    exists_request.set_collection_name(name);
    qdrant::CollectionExistsResponse exists_response;
    grpc::ClientContext exists_context;
    addAuth(exists_context);
    grpc::Status status = collections_->CollectionExists(
        &exists_context, exists_request, &exists_response);
    if (!status.ok()) {
      throw rpcError("collection existence check for '" + name + "'", status);
    }
    if (!exists_response.result().exists()) {
      qdrant::CreateCollection create_request;
      create_request.set_collection_name(name);
      create_request.mutable_vectors_config()->mutable_params()->set_size(
          embedding_dim_);
      create_request.mutable_vectors_config()
          ->mutable_params()
          ->set_distance(distance());
      qdrant::CollectionOperationResponse create_response;
      grpc::ClientContext create_context;
      addAuth(create_context);
      status = collections_->Create(&create_context, create_request,
                                    &create_response);
      if (!status.ok()) {
        throw rpcError("create collection '" + name + "'", status);
      }
      return;
    }

    qdrant::GetCollectionInfoRequest get_request;
    get_request.set_collection_name(name);
    qdrant::GetCollectionInfoResponse get_response;
    grpc::ClientContext get_context;
    addAuth(get_context);
    status = collections_->Get(&get_context, get_request, &get_response);
    if (!status.ok()) {
      throw rpcError("get collection '" + name + "'", status);
    }
    const qdrant::CollectionInfo& info = get_response.result();
    if (!info.config().params().has_vectors_config()) {
      throw std::runtime_error("Collection '" + name +
                               "' has no dense vector configuration");
    }
    const qdrant::VectorsConfig& vectors =
        info.config().params().vectors_config();
    if (vectors.config_case() == qdrant::VectorsConfig::kParamsMap) {
      throw std::runtime_error("Collection '" + name +
                               "' uses named vectors; this service requires a "
                               "single vector");
    }
    if (vectors.config_case() != qdrant::VectorsConfig::kParams) {
      throw std::runtime_error("Collection '" + name +
                               "' has an unsupported vector configuration");
    }
    const qdrant::VectorParams& params = vectors.params();
    if (params.size() != embedding_dim_) {
      std::ostringstream message;
      message << "Collection '" << name << "' has vector size " << params.size()
              << "; the model returns " << embedding_dim_;
      throw std::runtime_error(message.str());
    }
    if (params.distance() != distance()) {
      throw std::runtime_error("Collection '" + name + "' uses " +
                               qdrant::Distance_Name(params.distance()) +
                               "; configured metric is " + config_.metric);
    }
  }

  void setVector(qdrant::PointStruct* point,
                 const std::vector<float>& values) const {
    qdrant::DenseVector* dense =
        point->mutable_vectors()->mutable_vector()->mutable_dense();
    for (float value : values) {
      dense->add_data(value);
    }
  }

  void upsert(const qdrant::UpsertPoints& request,
              const std::string& operation) {
    qdrant::PointsOperationResponse response;
    grpc::ClientContext context;
    addAuth(context);
    const grpc::Status status = points_->Upsert(&context, request, &response);
    if (!status.ok()) {
      throw rpcError(operation, status);
    }
  }

  std::vector<qdrant::RetrievedPoint> scrollUser(const std::string& user_id) {
    std::vector<qdrant::RetrievedPoint> points;
    qdrant::PointId offset;
    bool has_offset = false;
    while (true) {
      qdrant::ScrollPoints request;
      request.set_collection_name(config_.raw_collection);
      request.mutable_filter()->CopyFrom(userFilter(user_id));
      request.set_limit(static_cast<uint32_t>(kScrollPageSize));
      request.mutable_with_payload()->set_enable(true);
      request.mutable_with_vectors()->set_enable(true);
      if (has_offset) {
        request.mutable_offset()->CopyFrom(offset);
      }
      qdrant::ScrollResponse response;
      grpc::ClientContext context;
      addAuth(context);
      const grpc::Status status = points_->Scroll(&context, request, &response);
      if (!status.ok()) {
        throw rpcError("scroll raw enrolments", status);
      }
      points.insert(points.end(), response.result().begin(),
                    response.result().end());
      if (!response.has_next_page_offset()) {
        return points;
      }
      offset.CopyFrom(response.next_page_offset());
      has_offset = true;
    }
  }

  std::vector<float> pointVector(const qdrant::RetrievedPoint& point) const {
    if (!point.has_vectors() ||
        point.vectors().vectors_options_case() != qdrant::VectorsOutput::kVector ||
        point.vectors().vector().vector_case() != qdrant::VectorOutput::kDense) {
      throw std::runtime_error(
          "Raw enrolment point does not contain a single dense vector");
    }
    const google::protobuf::RepeatedField<float>& data =
        point.vectors().vector().dense().data();
    if (static_cast<size_t>(data.size()) != embedding_dim_) {
      throw std::runtime_error("Stored enrolment has an unexpected vector dimension");
    }
    return std::vector<float>(data.begin(), data.end());
  }

  void setNameForPoints(const std::vector<qdrant::RetrievedPoint>& points,
                        const std::string& name) {
    qdrant::SetPayloadPoints request;
    request.set_collection_name(config_.raw_collection);
    request.set_wait(true);
    (*request.mutable_payload())["name"].set_string_value(name);
    qdrant::Filter* selector =
      request.mutable_points_selector()->mutable_filter();
    selector->CopyFrom(
      userFilter(stringPayload(points.front().payload(), "user_id", true)));
    qdrant::PointsOperationResponse response;
    grpc::ClientContext context;
    addAuth(context);
    const grpc::Status status = points_->SetPayload(&context, request, &response);
    if (!status.ok()) {
      throw rpcError("update enrolment names", status);
    }
  }

  void deleteByUser(const std::string& collection, const std::string& user_id) {
    qdrant::DeletePoints request;
    request.set_collection_name(collection);
    request.set_wait(true);
    request.mutable_points()->mutable_filter()->CopyFrom(userFilter(user_id));
    qdrant::PointsOperationResponse response;
    grpc::ClientContext context;
    addAuth(context);
    const grpc::Status status = points_->Delete(&context, request, &response);
    if (!status.ok()) {
      throw rpcError("delete user from collection '" + collection + "'", status);
    }
  }

  QdrantConfig config_;
  size_t embedding_dim_;
  std::shared_ptr<grpc::Channel> channel_;
  std::unique_ptr<qdrant::Collections::Stub> collections_;
  std::unique_ptr<qdrant::Points::Stub> points_;
  std::unique_ptr<qdrant::Qdrant::Stub> health_;
};

VectorStore::VectorStore(const QdrantConfig& config, size_t embedding_dim)
    : impl_(new Impl(config, embedding_dim)) {}

VectorStore::~VectorStore() = default;

void VectorStore::enroll(const std::string& user_id, const std::string& name,
                         const std::string& custom,
                         const std::vector<float>& embedding,
                         const std::vector<uint8_t>& image_bytes,
                         int64_t time_unix_ms) {
  impl_->enroll(user_id, name, custom, embedding, image_bytes, time_unix_ms);
}

std::vector<Candidate> VectorStore::query(const std::vector<float>& embedding,
                                          uint32_t max_results,
                                          float min_confidence) {
  return impl_->query(embedding, max_results, min_confidence);
}

std::vector<EnrolmentRecord> VectorStore::userInfo(const std::string& user_id) {
  return impl_->userInfo(user_id);
}

void VectorStore::deleteUser(const std::string& user_id) {
  impl_->deleteUser(user_id);
}

std::pair<std::string, std::string> VectorStore::debugInfo() {
  return impl_->debugInfo();
}

}  // namespace faceid