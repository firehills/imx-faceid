#include "faceid/vector_store_score.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

void expectNear(float actual, float expected, const std::string& name) {
  if (std::fabs(actual - expected) > 1e-5f) {
    throw std::runtime_error(name + " did not match the expected value");
  }
}

}  // namespace

int main() {
  const std::pair<float, float> cosine =
      faceid::scoreToDistanceConfidence(0.8f, "cosine");
  expectNear(cosine.first, 0.2f, "cosine raw distance");
  expectNear(cosine.second, 0.9f, "cosine confidence");

  const std::pair<float, float> euclidean =
      faceid::scoreToDistanceConfidence(1.0f, "euclidean");
  expectNear(euclidean.first, 1.0f, "Euclidean raw distance");
  expectNear(euclidean.second, 0.75f, "Euclidean confidence");

  const std::pair<float, float> high_cosine =
      faceid::scoreToDistanceConfidence(3.0f, "cosine");
  expectNear(high_cosine.second, 1.0f, "upper confidence bound");

  const std::pair<float, float> high_distance =
      faceid::scoreToDistanceConfidence(3.0f, "euclidean");
  expectNear(high_distance.second, 0.0f, "lower confidence bound");

  bool rejected_metric = false;
  try {
    faceid::scoreToDistanceConfidence(0.0f, "dot");
  } catch (const std::invalid_argument&) {
    rejected_metric = true;
  }
  if (!rejected_metric) {
    throw std::runtime_error("Unsupported metrics must be rejected");
  }
  return 0;
}