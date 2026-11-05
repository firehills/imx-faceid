#include "faceid/vector_store_score.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace faceid {

std::pair<float, float> scoreToDistanceConfidence(
    float score, const std::string& metric) {
  float raw_distance;
  float confidence;
  if (metric == "cosine") {
    raw_distance = 1.0f - score;
    confidence = (1.0f + score) / 2.0f;
  } else if (metric == "euclidean") {
    raw_distance = score;
    confidence = 1.0f - (score * score) / 4.0f;
  } else {
    throw std::invalid_argument("Unsupported distance metric: " + metric);
  }
  return std::make_pair(raw_distance,
                        std::max(0.0f, std::min(1.0f, confidence)));
}

}  // namespace faceid