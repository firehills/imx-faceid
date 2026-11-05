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

/** Image bytes and preprocessing metadata shared by enrolment and query. */
struct ImageRequest {
  std::vector<uint8_t> bytes;
  uint32_t width;
  uint32_t height;
  bool is_preprocessed;
  faceid::fbs::ImageEncoding encoding;
  std::vector<std::array<float, 2> > keypoints;
};

/** A request to add an image embedding to a user's enrolments. */
struct EnrolRequest {
  std::string user_id;
  std::string name;
  ImageRequest image;
  std::string reply_to;
  int64_t time_unix_ms;
  std::string custom;
};

/** A request to identify the person represented by an image. */
struct QueryRequest {
  std::string track_id;
  ImageRequest image;
  std::string reply_to;
  float min_confidence;
  uint32_t max_results;
  int64_t pts;
  int64_t time_unix_ms;
  std::string custom;
};

/** A request to return the raw enrolments associated with a user. */
struct UserInfoRequest {
  std::string user_id;
  std::string reply_to;
  int64_t time_unix_ms;
};

/** A request to remove all enrolments associated with a user. */
struct DeleteRequest {
  std::string user_id;
  std::string reply_to;
  int64_t time_unix_ms;
};

/** A request for service and dependency diagnostic information. */
struct DebugRequest {
  std::string reply_to;
  int64_t time_unix_ms;
};

/** One Qdrant match returned in an identity query response. */
struct Candidate {
  std::string user_id;
  std::string name;
  float confidence;
  float raw_distance;
};

/** One raw enrolment returned by a user-info request. */
struct EnrolmentRecord {
  std::vector<float> embedding;
  int64_t time_unix_ms;
};

/** Decode and validate an envelope containing an EnrolMsg. */
EnrolRequest parseEnrolRequest(const uint8_t* data, size_t size);

/** Decode and validate an envelope containing an IdRequestMsg. */
QueryRequest parseQueryRequest(const uint8_t* data, size_t size);

/** Decode and validate an envelope containing a UserInfoMsg. */
UserInfoRequest parseUserInfoRequest(const uint8_t* data, size_t size);

/** Decode and validate an envelope containing a DeleteUserMsg. */
DeleteRequest parseDeleteRequest(const uint8_t* data, size_t size);

/** Decode and validate an envelope containing a DebugInfoMsg. */
DebugRequest parseDebugRequest(const uint8_t* data, size_t size);

/** Encode an enrolment success or failure response. */
std::vector<uint8_t> makeEnrolResult(const std::string& user_id, bool success,
                                     const std::string& reason = "");

/** Encode an identity query response, including candidate confidence and distance. */
std::vector<uint8_t> makeQueryResult(const std::string& track_id,
                                     const std::vector<Candidate>& candidates,
                                     bool success,
                                     const std::string& reason = "");

/** Encode a user-info response containing each raw embedding and timestamp. */
std::vector<uint8_t> makeUserInfoResponse(
    const std::string& user_id, const std::vector<EnrolmentRecord>& records,
    bool success, const std::string& reason = "");

/** Encode a delete-user success or failure response. */
std::vector<uint8_t> makeDeleteResponse(const std::string& user_id,
                                       bool success,
                                       const std::string& reason = "");

/** Encode database, inference, and NATS diagnostic information. */
std::vector<uint8_t> makeDebugResponse(
    const std::string& database_type, const std::string& database_version,
    const std::string& inference_backend, const std::string& model_name,
    uint32_t embedding_dim, const std::string& nats_url, bool success,
    const std::string& reason = "");

}  // namespace faceid

#endif  // FACEID_FLATBUFFER_CODEC_H