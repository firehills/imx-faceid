#ifndef FACEID_VECTOR_STORE_SCORE_H
#define FACEID_VECTOR_STORE_SCORE_H

#include <string>
#include <utility>

namespace faceid {

/** Convert a Qdrant score into the service's raw-distance and confidence pair. */
std::pair<float, float> scoreToDistanceConfidence(float score,
                                                  const std::string& metric);

}  // namespace faceid

#endif  // FACEID_VECTOR_STORE_SCORE_H