#ifndef FACEID_VECTOR_STORE_SCORE_H
#define FACEID_VECTOR_STORE_SCORE_H

#include <string>
#include <utility>

namespace faceid {

/**
 * Convert a Qdrant score into the service's raw-distance/confidence pair.
 * Cosine uses distance 1-score and confidence (1+score)/2; Euclidean uses
 * distance=score and confidence=1-score^2/4, clamped to [0,1].
 * @param score Raw similarity or distance returned by Qdrant.
 * @param metric Validated metric name: "cosine" or "euclidean".
 * @return Pair whose first item is distance and second item is bounded confidence.
 * @throws std::invalid_argument when metric is unsupported.
 */
std::pair<float, float> scoreToDistanceConfidence(float score,
                                                  const std::string& metric);

}  // namespace faceid

#endif  // FACEID_VECTOR_STORE_SCORE_H