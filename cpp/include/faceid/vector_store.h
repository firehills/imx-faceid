#ifndef FACEID_VECTOR_STORE_H
#define FACEID_VECTOR_STORE_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "faceid/config.h"
#include "faceid/flatbuffer_codec.h"
#include "faceid/vector_store_score.h"

namespace faceid {

/** Store raw enrolments and normalized per-user vectors in Qdrant. */
class VectorStore {
 public:
  /** Connect to Qdrant and create or validate both configured collections. */
  VectorStore(const QdrantConfig& config, size_t embedding_dim);

  /** Release the gRPC stubs and channel. */
  ~VectorStore();

  VectorStore(const VectorStore&) = delete;
  VectorStore& operator=(const VectorStore&) = delete;

  /** Add a raw enrolment and replace the user's normalized mean vector. */
  void enroll(const std::string& user_id, const std::string& name,
              const std::string& custom, const std::vector<float>& embedding,
              const std::vector<uint8_t>& image_bytes, int64_t time_unix_ms);

  /** Return nearest normalized-user matches meeting the confidence threshold. */
  std::vector<Candidate> query(const std::vector<float>& embedding,
                               uint32_t max_results,
                               float min_confidence);

  /** Return each raw enrolment vector and its timestamp for a user. */
  std::vector<EnrolmentRecord> userInfo(const std::string& user_id);

  /** Delete all raw and normalized points belonging to a user. */
  void deleteUser(const std::string& user_id);

  /** Return the Qdrant product name and server version for diagnostics. */
  std::pair<std::string, std::string> debugInfo();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace faceid

#endif  // FACEID_VECTOR_STORE_H