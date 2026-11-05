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

/**
 * Persist raw enrolments and one normalized mean vector per user in Qdrant.
 * This public wrapper owns its implementation and gRPC resources through PImpl.
 */
class VectorStore {
 public:
  /**
   * Connect to Qdrant and create or validate both configured collections.
   * @param config Endpoint, credentials, collection names, and metric.
   * @param embedding_dim Required dimension for raw and normalized vectors.
   * @throws std::runtime_error or std::invalid_argument on RPC/schema failure.
   */
  VectorStore(const QdrantConfig& config, size_t embedding_dim);

  /** Release implementation, Qdrant stubs, and gRPC channel. */
  ~VectorStore();

  VectorStore(const VectorStore&) = delete;
  VectorStore& operator=(const VectorStore&) = delete;

  /**
   * Store a raw enrolment and recompute that user's normalized mean vector.
   * @param user_id Stable owner key.
   * @param name Display name shared by the user's enrolments.
   * @param custom Opaque application metadata retained with the raw point.
   * @param embedding Unit-length model vector of the configured dimension.
   * @param image_bytes Original request image data retained with the raw point.
   * @param time_unix_ms Request timestamp, or server-generated time.
   */
  void enroll(const std::string& user_id, const std::string& name,
              const std::string& custom, const std::vector<float>& embedding,
              const std::vector<uint8_t>& image_bytes, int64_t time_unix_ms);

  /**
   * Search normalized user vectors and keep matches above the confidence floor.
   * @param embedding Unit-length query vector.
   * @param max_results Upper bound on Qdrant's nearest points to inspect.
   * @param min_confidence Minimum converted confidence to return.
   * @return Candidate identity and score fields, without stored vectors.
   */
  std::vector<Candidate> query(const std::vector<float>& embedding,
                               uint32_t max_results,
                               float min_confidence);

  /**
   * Return raw embeddings and their stored timestamps for one user.
   * @param user_id Owner key used by the Qdrant payload filter.
   * @return One record per raw enrolment; empty when none exist.
   */
  std::vector<EnrolmentRecord> userInfo(const std::string& user_id);

  /**
   * Delete the user's points from both raw and normalized collections.
   * @param user_id Owner key used by the Qdrant payload filters.
   */
  void deleteUser(const std::string& user_id);

  /**
   * Contact Qdrant health service for database diagnostics.
   * @return Pair of database product name and server version.
   */
  std::pair<std::string, std::string> debugInfo();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace faceid

#endif  // FACEID_VECTOR_STORE_H