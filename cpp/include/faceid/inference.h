#ifndef FACEID_INFERENCE_H
#define FACEID_INFERENCE_H

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "faceid/config.h"
#include "faceid/flatbuffer_codec.h"

namespace faceid {

/** Raised when image decoding or model execution cannot produce an embedding. */
class InferenceError : public std::runtime_error {
 public:
  /** Create an inference error with an explanation safe to return to a client. */
  explicit InferenceError(const std::string& message);
};

/** Return a finite, non-zero vector scaled to unit Euclidean length. */
std::vector<float> normalizeEmbedding(const std::vector<float>& values);

/** Decode and normalize an image into the configured model tensor layout. */
std::vector<float> prepareImage(const ImageRequest& image,
                                const ModelConfig& config);

/** Own an ONNX Runtime or TensorFlow Lite model and generate normalized vectors. */
class EmbeddingModel {
 public:
  /** Load the configured model and discover its embedding dimension. */
  explicit EmbeddingModel(const ModelConfig& config);

  /** Release model sessions, interpreters, and any configured delegate. */
  ~EmbeddingModel();

  EmbeddingModel(const EmbeddingModel&) = delete;
  EmbeddingModel& operator=(const EmbeddingModel&) = delete;

  /** Run preprocessing and inference, returning a unit-length float vector. */
  std::vector<float> embed(const ImageRequest& image);

  /** Return the embedding size discovered by a startup inference. */
  size_t embeddingDimension() const;

  /** Return the selected runtime name: "onnx" or "litert". */
  const std::string& runtimeName() const;

  /** Return the configured model path for service diagnostics. */
  const std::string& modelPath() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace faceid

#endif  // FACEID_INFERENCE_H