#include "faceid/flatbuffer_codec.h"

#include <flatbuffers/flatbuffers.h>

#include <chrono>
#include <sstream>

namespace faceid {
namespace {

/** Return the current wall-clock time in milliseconds since the Unix epoch. */
int64_t currentTimeUnixMs() {
  const std::chrono::system_clock::duration elapsed =
      std::chrono::system_clock::now().time_since_epoch();
  return std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
}

/**
 * Verify an incoming FACE FlatBuffer envelope and check its union variant.
 * @param data Raw NATS payload bytes.
 * @param size Number of bytes available at data.
 * @param expected Body variant required by the calling request parser.
 * @return Verified envelope borrowing the supplied buffer's storage.
 */
const faceid::fbs::Envelope* parseEnvelope(const uint8_t* data, size_t size,
                                           faceid::fbs::Body expected) {
  if (data == nullptr || size == 0) {
    throw FlatBufferError("Message is empty");
  }
  flatbuffers::Verifier verifier(data, size);
  if (!faceid::fbs::VerifyEnvelopeBuffer(verifier)) {
    throw FlatBufferError("Message is not a valid FACE FlatBuffer");
  }
  const faceid::fbs::Envelope* envelope = faceid::fbs::GetEnvelope(data);
  if (envelope->body_type() != expected) {
    std::ostringstream message;
    message << "Expected body type " << static_cast<int>(expected)
            << ", received " << static_cast<int>(envelope->body_type());
    throw FlatBufferError(message.str());
  }
  return envelope;
}

/**
 * Copy a required FlatBuffers string into standard storage.
 * @param value String field from the verified envelope.
 * @param field_name Field label for missing/empty input errors.
 * @return Owned string content.
 */
std::string requiredString(const flatbuffers::String* value,
                           const char* field_name) {
  if (value == nullptr || value->size() == 0) {
    throw FlatBufferError(std::string("Required field '") + field_name +
                          "' must not be empty");
  }
  return value->str();
}

/**
 * Copy an optional FlatBuffers string, mapping a missing field to empty text.
 * @param value Optional string field from the verified envelope.
 * @return Owned string content or an empty string.
 */
std::string optionalString(const flatbuffers::String* value) {
  return value == nullptr ? std::string() : value->str();
}

/**
 * Copy image bytes and preprocessing metadata out of a FlatBuffer request.
 * @param image Required pixel or encoded-image byte vector.
 * @param width Source image width carried by the request.
 * @param height Source image height carried by the request.
 * @param is_preprocessed Whether upstream already performed spatial processing.
 * @param encoding Declared pixel layout or compressed-image format.
 * @param keypoints Optional ordered facial landmarks.
 * @return An owned ImageRequest consumed by image preparation and inference.
 */
ImageRequest readImage(const flatbuffers::Vector<uint8_t>* image,
                       uint32_t width, uint32_t height, bool is_preprocessed,
                       faceid::fbs::ImageEncoding encoding,
                       const flatbuffers::Vector<const faceid::fbs::Keypoint*>*
                           keypoints) {
  if (image == nullptr) {
    throw FlatBufferError("Required image vector is missing");
  }

  ImageRequest result;
  result.bytes.assign(image->begin(), image->end());
  result.width = width;
  result.height = height;
  result.is_preprocessed = is_preprocessed;
  result.encoding = encoding;
  if (keypoints != nullptr) {
    result.keypoints.reserve(keypoints->size());
    for (flatbuffers::uoffset_t index = 0; index < keypoints->size(); ++index) {
      const faceid::fbs::Keypoint* point = keypoints->Get(index);
      result.keypoints.push_back({{point->x(), point->y()}});
    }
  }
  return result;
}

/**
 * Finish a typed response body as a complete FACE envelope.
 * @param builder Builder containing the response body and its strings/vectors.
 * @param body_type Union tag identifying the response schema.
 * @param body Offset to the body table.
 * @return Complete serialized buffer, owned by the returned byte vector.
 */
std::vector<uint8_t> finishEnvelope(flatbuffers::FlatBufferBuilder* builder,
                                   faceid::fbs::Body body_type,
                                   flatbuffers::Offset<void> body) {
  const flatbuffers::Offset<faceid::fbs::Envelope> envelope =
      faceid::fbs::CreateEnvelope(*builder, body_type, body);
  faceid::fbs::FinishEnvelopeBuffer(*builder, envelope);
  return std::vector<uint8_t>(builder->GetBufferPointer(),
                              builder->GetBufferPointer() + builder->GetSize());
}

}  // namespace

/** Construct the codec exception used for malformed or mismatched envelopes. */
FlatBufferError::FlatBufferError(const std::string& message)
    : std::runtime_error(message) {}

/**
 * Decode an enrolment envelope into the service's owned request structure.
 * @param data Serialized envelope bytes received from NATS.
 * @param size Number of bytes in data.
 * @return User, image, reply, timestamp, and custom metadata for enrolment.
 */
EnrolRequest parseEnrolRequest(const uint8_t* data, size_t size) {
  const faceid::fbs::EnrolMsg* body =
      parseEnvelope(data, size, faceid::fbs::Body::EnrolMsg)
          ->body_as_EnrolMsg();
  if (body == nullptr) {
    throw FlatBufferError("Envelope does not contain an enrolment body");
  }

  EnrolRequest request;
  request.user_id = requiredString(body->user_id(), "user_id");
  request.name = requiredString(body->name(), "name");
  request.image = readImage(body->image(), body->width(), body->height(),
                            body->is_preprocessed(), body->encoding(),
                            body->keypoints());
  request.reply_to = optionalString(body->reply_to());
  request.time_unix_ms = body->time_unix_ms();
  request.custom = optionalString(body->custom());
  return request;
}

/**
 * Decode an identity-query envelope into validated service request fields.
 * @param data Serialized envelope bytes received from NATS.
 * @param size Number of bytes in data.
 * @return Query identifiers, image data, matching thresholds, and metadata.
 */
QueryRequest parseQueryRequest(const uint8_t* data, size_t size) {
  const faceid::fbs::IdRequestMsg* body =
      parseEnvelope(data, size, faceid::fbs::Body::IdRequestMsg)
          ->body_as_IdRequestMsg();
  if (body == nullptr) {
    throw FlatBufferError("Envelope does not contain an identity request body");
  }

  QueryRequest request;
  request.track_id = requiredString(body->track_id(), "track_id");
  request.image = readImage(body->image(), body->width(), body->height(),
                            body->is_preprocessed(), body->encoding(),
                            body->keypoints());
  request.reply_to = requiredString(body->reply_to(), "reply_to");
  request.min_confidence = body->min_confidence();
  request.max_results = body->max_results();
  request.pts = body->pts();
  request.time_unix_ms = body->time_unix_ms();
  request.custom = optionalString(body->custom());
  return request;
}

/**
 * Decode a user-info request after verifying its envelope type and user ID.
 * @param data Serialized envelope bytes received from NATS.
 * @param size Number of bytes in data.
 * @return Requested user, optional reply subject, and request timestamp.
 */
UserInfoRequest parseUserInfoRequest(const uint8_t* data, size_t size) {
  const faceid::fbs::UserInfoMsg* body =
      parseEnvelope(data, size, faceid::fbs::Body::UserInfoMsg)
          ->body_as_UserInfoMsg();
  if (body == nullptr) {
    throw FlatBufferError("Envelope does not contain a user-info body");
  }

  UserInfoRequest request;
  request.user_id = requiredString(body->user_id(), "user_id");
  request.reply_to = optionalString(body->reply_to());
  request.time_unix_ms = body->time_unix_ms();
  return request;
}

/**
 * Decode a delete-user request from a verified FACE envelope.
 * @param data Serialized envelope bytes received from NATS.
 * @param size Number of bytes in data.
 * @return User ID, optional reply subject, and request timestamp.
 */
DeleteRequest parseDeleteRequest(const uint8_t* data, size_t size) {
  const faceid::fbs::DeleteUserMsg* body =
      parseEnvelope(data, size, faceid::fbs::Body::DeleteUserMsg)
          ->body_as_DeleteUserMsg();
  if (body == nullptr) {
    throw FlatBufferError("Envelope does not contain a delete-user body");
  }

  DeleteRequest request;
  request.user_id = requiredString(body->user_id(), "user_id");
  request.reply_to = optionalString(body->reply_to());
  request.time_unix_ms = body->time_unix_ms();
  return request;
}

/**
 * Decode a service-diagnostics request from a verified FACE envelope.
 * @param data Serialized envelope bytes received from NATS.
 * @param size Number of bytes in data.
 * @return Optional reply subject and request timestamp.
 */
DebugRequest parseDebugRequest(const uint8_t* data, size_t size) {
  const faceid::fbs::DebugInfoMsg* body =
      parseEnvelope(data, size, faceid::fbs::Body::DebugInfoMsg)
          ->body_as_DebugInfoMsg();
  if (body == nullptr) {
    throw FlatBufferError("Envelope does not contain a debug-info body");
  }

  DebugRequest request;
  request.reply_to = optionalString(body->reply_to());
  request.time_unix_ms = body->time_unix_ms();
  return request;
}

/**
 * Serialize the outcome of an enrolment operation.
 * @param user_id User identifier echoed to the caller.
 * @param success Whether the operation completed.
 * @param reason Empty on success, otherwise the failure detail.
 * @return Serialized EnrolResult envelope with a server timestamp.
 */
std::vector<uint8_t> makeEnrolResult(const std::string& user_id, bool success,
                                     const std::string& reason) {
  flatbuffers::FlatBufferBuilder builder;
  const flatbuffers::Offset<faceid::fbs::EnrolResultMsg> body =
      faceid::fbs::CreateEnrolResultMsgDirect(
          builder, user_id.c_str(), success, reason.c_str(), currentTimeUnixMs());
  return finishEnvelope(&builder, faceid::fbs::Body::EnrolResultMsg,
                        body.Union());
}

/**
 * Serialize identity matches and their distance/confidence values.
 * @param track_id Request identifier echoed to the caller.
 * @param candidates Matching users; each vector is represented by metadata only.
 * @param success Whether the query completed.
 * @param reason Empty on success, otherwise the failure detail.
 * @return Serialized IdResult envelope with a server timestamp.
 */
std::vector<uint8_t> makeQueryResult(const std::string& track_id,
                                     const std::vector<Candidate>& candidates,
                                     bool success, const std::string& reason) {
  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<faceid::fbs::IdCandidate> > candidate_offsets;
  candidate_offsets.reserve(candidates.size());
  for (size_t index = 0; index < candidates.size(); ++index) {
    const Candidate& candidate = candidates[index];
    candidate_offsets.push_back(faceid::fbs::CreateIdCandidateDirect(
        builder, candidate.user_id.c_str(), candidate.name.c_str(),
        candidate.confidence, candidate.raw_distance));
  }
  const flatbuffers::Offset<flatbuffers::Vector<
      flatbuffers::Offset<faceid::fbs::IdCandidate> > > candidate_vector =
      builder.CreateVector(candidate_offsets);
    const flatbuffers::Offset<flatbuffers::String> track_id_offset =
      builder.CreateString(track_id);
    const flatbuffers::Offset<flatbuffers::String> reason_offset =
      builder.CreateString(reason);
  const flatbuffers::Offset<faceid::fbs::IdResultMsg> body =
      faceid::fbs::CreateIdResultMsg(builder, track_id_offset, candidate_vector,
                      currentTimeUnixMs(), success,
                      reason_offset);
  return finishEnvelope(&builder, faceid::fbs::Body::IdResultMsg,
                        body.Union());
}

/**
 * Serialize enrolment timestamps and vectors for a user-info response.
 * @param user_id User identifier echoed to the caller.
 * @param records Raw enrolment records returned from the vector store.
 * @param success Whether the lookup completed.
 * @param reason Empty on success, otherwise the failure detail.
 * @return Serialized UserInfoResponse envelope with record count and timestamp.
 */
std::vector<uint8_t> makeUserInfoResponse(
    const std::string& user_id, const std::vector<EnrolmentRecord>& records,
    bool success, const std::string& reason) {
  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<faceid::fbs::EnrolmentRecordMsg> > offsets;
  offsets.reserve(records.size());
  for (size_t index = 0; index < records.size(); ++index) {
    const EnrolmentRecord& record = records[index];
    const flatbuffers::Offset<flatbuffers::Vector<float> > embedding =
        builder.CreateVector(record.embedding);
    offsets.push_back(faceid::fbs::CreateEnrolmentRecordMsg(
        builder, embedding, record.time_unix_ms));
  }
  const flatbuffers::Offset<faceid::fbs::UserInfoResponseMsg> body =
      faceid::fbs::CreateUserInfoResponseMsgDirect(
          builder, user_id.c_str(), static_cast<uint32_t>(records.size()),
          &offsets, success, reason.c_str(), currentTimeUnixMs());
  return finishEnvelope(&builder, faceid::fbs::Body::UserInfoResponseMsg,
                        body.Union());
}

/**
 * Serialize a user-deletion outcome.
 * @param user_id User identifier echoed to the caller.
 * @param success Whether deletion completed.
 * @param reason Empty on success, otherwise the failure detail.
 * @return Serialized DeleteUserResponse envelope with a server timestamp.
 */
std::vector<uint8_t> makeDeleteResponse(const std::string& user_id,
                                       bool success,
                                       const std::string& reason) {
  flatbuffers::FlatBufferBuilder builder;
  const flatbuffers::Offset<faceid::fbs::DeleteUserResponseMsg> body =
      faceid::fbs::CreateDeleteUserResponseMsgDirect(
          builder, user_id.c_str(), success, reason.c_str(), currentTimeUnixMs());
  return finishEnvelope(&builder, faceid::fbs::Body::DeleteUserResponseMsg,
                        body.Union());
}

/**
 * Serialize database, runtime, model, and NATS diagnostic information.
 * @param database_type Backing-store identity.
 * @param database_version Backing-store server version.
 * @param inference_backend Selected inference runtime.
 * @param model_name Configured model basename.
 * @param embedding_dim Number of floats produced by the model.
 * @param nats_url Configured service connection endpoint.
 * @param success Whether diagnostics were collected successfully.
 * @param reason Empty on success, otherwise the failure detail.
 * @return Serialized DebugInfoResponse envelope with a server timestamp.
 */
std::vector<uint8_t> makeDebugResponse(
    const std::string& database_type, const std::string& database_version,
    const std::string& inference_backend, const std::string& model_name,
    uint32_t embedding_dim, const std::string& nats_url, bool success,
    const std::string& reason) {
  flatbuffers::FlatBufferBuilder builder;
  const flatbuffers::Offset<faceid::fbs::DebugInfoResponseMsg> body =
      faceid::fbs::CreateDebugInfoResponseMsgDirect(
          builder, database_type.c_str(), database_version.c_str(),
          inference_backend.c_str(), model_name.c_str(), embedding_dim,
          nats_url.c_str(), success, reason.c_str(), currentTimeUnixMs());
  return finishEnvelope(&builder, faceid::fbs::Body::DebugInfoResponseMsg,
                        body.Union());
}

}  // namespace faceid