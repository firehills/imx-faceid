#ifndef FACEID_FLATBUFFER_CODEC_H
#define FACEID_FLATBUFFER_CODEC_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "faceid_generated.h"

namespace faceid {

/** Raised when a request does not match the FACE FlatBuffers contract. */
class FlatBufferError : public std::runtime_error {
 public:
  /** Create an error with a message suitable for an operation failure reply. */
  explicit FlatBufferError(const std::string& message);
};

/**
 * Owned image payload and preprocessing metadata shared by enrolment and query.
 * Parsed byte/keypoint data is copied out of FlatBuffers so it can safely
 * outlive the incoming NATS message buffer.
 */
struct ImageRequest {
  std::vector<uint8_t> bytes;  ///< Raw RGB bytes or compressed image bytes.
  uint32_t width;              ///< Declared source width; zero if unspecified.
  uint32_t height;             ///< Declared source height; zero if unspecified.
  bool is_preprocessed;        ///< Upstream spatial preprocessing indicator.
  faceid::fbs::ImageEncoding encoding;  ///< Pixel layout or encoded image type.
  std::vector<std::array<float, 2> > keypoints;  ///< Ordered facial landmarks.
};

/** Decoded fields used to add one image embedding to a user's enrolments. */
struct EnrolRequest {
  std::string user_id;  ///< Required owner identity.
  std::string name;     ///< Required display name stored with enrolments.
  ImageRequest image;   ///< Pixels and preprocessing metadata for inference.
  std::string reply_to; ///< Optional NATS response subject.
  int64_t time_unix_ms; ///< Client timestamp; zero requests server time.
  std::string custom;   ///< Opaque application metadata retained by the store.
};

/** Decoded image query and matching constraints for identity search. */
struct QueryRequest {
  std::string track_id;   ///< Required caller-provided request identifier.
  ImageRequest image;     ///< Pixels and preprocessing metadata for inference.
  std::string reply_to;   ///< Required NATS response subject.
  float min_confidence;   ///< Minimum confidence retained in the response.
  uint32_t max_results;   ///< Maximum nearest Qdrant candidates to request.
  int64_t pts;            ///< Caller presentation timestamp.
  int64_t time_unix_ms;   ///< Caller wall-clock timestamp.
  std::string custom;     ///< Opaque request metadata, not used for matching.
};

/** User identity and reply metadata for a raw-enrolment lookup. */
struct UserInfoRequest {
  std::string user_id;   ///< Required owner identity to look up.
  std::string reply_to;  ///< Optional NATS response subject.
  int64_t time_unix_ms;  ///< Caller request timestamp.
};

/** User identity and reply metadata for deleting all enrolments. */
struct DeleteRequest {
  std::string user_id;   ///< Required owner identity to delete.
  std::string reply_to;  ///< Optional NATS response subject.
  int64_t time_unix_ms;  ///< Caller request timestamp.
};

/** Reply metadata for a service/database diagnostic request. */
struct DebugRequest {
  std::string reply_to;  ///< Optional NATS response subject.
  int64_t time_unix_ms;  ///< Caller request timestamp.
};

/** Identity and converted score fields for one Qdrant query match. */
struct Candidate {
  std::string user_id;  ///< Matched owner identity.
  std::string name;     ///< Stored display name, possibly empty.
  float confidence;     ///< Metric-normalized confidence in [0, 1].
  float raw_distance;   ///< Distance converted from Qdrant's raw score.
};

/** Vector and timestamp copied from one raw Qdrant enrolment point. */
struct EnrolmentRecord {
  std::vector<float> embedding;  ///< Stored raw embedding; serialized in response.
  int64_t time_unix_ms;          ///< Timestamp stored with this enrolment.
};

/**
 * Verify an incoming envelope and copy its EnrolMsg fields.
 * @param data Serialized FACE envelope bytes.
 * @param size Number of bytes available in data.
 * @return Owned enrolment request; malformed or wrong-type envelopes throw.
 */
EnrolRequest parseEnrolRequest(const uint8_t* data, size_t size);

/**
 * Verify an incoming envelope and copy its IdRequestMsg fields.
 * @param data Serialized FACE envelope bytes.
 * @param size Number of bytes available in data.
 * @return Owned query request; malformed or wrong-type envelopes throw.
 */
QueryRequest parseQueryRequest(const uint8_t* data, size_t size);

/**
 * Decode the identity and reply metadata of a UserInfoMsg envelope.
 * @param data Serialized FACE envelope bytes.
 * @param size Number of bytes available in data.
 * @return User-info request fields; malformed or wrong-type envelopes throw.
 */
UserInfoRequest parseUserInfoRequest(const uint8_t* data, size_t size);

/**
 * Decode the identity and reply metadata of a DeleteUserMsg envelope.
 * @param data Serialized FACE envelope bytes.
 * @param size Number of bytes available in data.
 * @return Delete request fields; malformed or wrong-type envelopes throw.
 */
DeleteRequest parseDeleteRequest(const uint8_t* data, size_t size);

/**
 * Decode reply metadata from a DebugInfoMsg envelope.
 * @param data Serialized FACE envelope bytes.
 * @param size Number of bytes available in data.
 * @return Diagnostic request fields; malformed or wrong-type envelopes throw.
 */
DebugRequest parseDebugRequest(const uint8_t* data, size_t size);

/**
 * Build a timestamped EnrolResult envelope.
 * @param user_id User identity echoed to the caller.
 * @param success Operation outcome.
 * @param reason Failure explanation; ignored by callers on success.
 * @return Serialized FACE envelope bytes ready to publish on NATS.
 */
std::vector<uint8_t> makeEnrolResult(const std::string& user_id, bool success,
                                     const std::string& reason = "");

/**
 * Build a timestamped IdResult envelope with candidate identity and scores.
 * @param track_id Request identifier echoed to the caller.
 * @param candidates Matching users and their converted score metadata.
 * @param success Operation outcome.
 * @param reason Failure explanation; ignored by callers on success.
 * @return Serialized FACE envelope bytes ready to publish on NATS.
 */
std::vector<uint8_t> makeQueryResult(const std::string& track_id,
                                     const std::vector<Candidate>& candidates,
                                     bool success,
                                     const std::string& reason = "");

/**
 * Build a timestamped response containing raw enrolment vectors and times.
 * @param user_id User identity echoed to the caller.
 * @param records Raw vectors and timestamps retrieved from Qdrant.
 * @param success Operation outcome.
 * @param reason Failure explanation; ignored by callers on success.
 * @return Serialized FACE envelope bytes ready to publish on NATS.
 */
std::vector<uint8_t> makeUserInfoResponse(
    const std::string& user_id, const std::vector<EnrolmentRecord>& records,
    bool success, const std::string& reason = "");

/**
 * Build a timestamped DeleteUserResponse envelope.
 * @param user_id User identity echoed to the caller.
 * @param success Operation outcome.
 * @param reason Failure explanation; ignored by callers on success.
 * @return Serialized FACE envelope bytes ready to publish on NATS.
 */
std::vector<uint8_t> makeDeleteResponse(const std::string& user_id,
                                       bool success,
                                       const std::string& reason = "");

/**
 * Build a timestamped response containing dependency and model diagnostics.
 * @param database_type Backing-store identity.
 * @param database_version Backing-store server version.
 * @param inference_backend Selected inference runtime.
 * @param model_name Configured model basename.
 * @param embedding_dim Number of floats produced by the model.
 * @param nats_url Configured NATS endpoint.
 * @param success Whether diagnostics were collected successfully.
 * @param reason Failure explanation; ignored by callers on success.
 * @return Serialized FACE envelope bytes ready to publish on NATS.
 */
std::vector<uint8_t> makeDebugResponse(
    const std::string& database_type, const std::string& database_version,
    const std::string& inference_backend, const std::string& model_name,
    uint32_t embedding_dim, const std::string& nats_url, bool success,
    const std::string& reason = "");

}  // namespace faceid

#endif  // FACEID_FLATBUFFER_CODEC_H