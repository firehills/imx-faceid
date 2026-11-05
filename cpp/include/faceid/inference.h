#ifndef FACEID_INFERENCE_H
#define FACEID_INFERENCE_H

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "faceid/config.h"
#include "faceid/flatbuffer_codec.h"

namespace faceid {

/**
 * Raised when request image conversion or backend execution cannot produce an
 * embedding accepted by the service.
 */
class InferenceError : public std::runtime_error {
 public:
  /** Construct an inference failure with a caller-readable diagnostic. */
  explicit InferenceError(const std::string& message);
};

/**
 * Validate and normalize a raw model output to unit Euclidean length.
 * @param values Raw model output components.
 * @return Owned finite, non-zero vector with unit norm.
 * @throws InferenceError if values are empty, non-finite, or zero-length.
 */
std::vector<float> normalizeEmbedding(const std::vector<float>& values);

/**
 * Decode and preprocess request pixels into the model's normalized tensor.
 * @param image Raw RGB or supported compressed image and request metadata.
 * @param config Model input size, layout, color order, alignment, and scaling.
 * @return Contiguous float tensor in the configured NCHW or NHWC order.
 * @throws InferenceError for invalid image data or incompatible metadata.
 */
std::vector<float> prepareImage(const ImageRequest& image,
                                const ModelConfig& config);

/**
 * Per-call wall-clock stage durations, excluding wait for the inference mutex.
 * Values are populated on successful embed() calls only.
 */
struct InferenceTiming {
  /** Image decoding, alignment, resizing, and tensor conversion. */
  double preprocess_ms = 0.0;
  /** ONNX Runtime or LiteRT execution plus embedding normalization. */
  double inference_ms = 0.0;
};

/**
 * Own one ONNX Runtime or LiteRT model and transform request images to vectors.
 * Construction loads the model and performs a warm-up inference to determine
 * output dimension; embed() preprocesses each request and serializes backend use.
 */
class EmbeddingModel {
 public:
  /**
   * Load the configured runtime/model and discover output dimension by warm-up.
   * @param config Validated model path, runtime, and tensor/preprocessing data.
   * @throws InferenceError if model loading or warm-up fails.
   */
  explicit EmbeddingModel(const ModelConfig& config);

  /** Release runtime sessions, interpreter tensors, and any external delegate. */
  ~EmbeddingModel();

  EmbeddingModel(const EmbeddingModel&) = delete;
  EmbeddingModel& operator=(const EmbeddingModel&) = delete;

  /**
   * Decode and preprocess one image, execute the model, and normalize its output.
   * @param image Request payload and image preprocessing metadata.
   * @param timing Optional output receiving preprocessing and model durations;
   *               left unchanged if the operation throws.
   * @return Unit-length embedding with embeddingDimension() components.
   * @throws InferenceError for invalid images, incompatible tensors, or runtime
   *         failures.
   */
  std::vector<float> embed(const ImageRequest& image,
                           InferenceTiming* timing = nullptr);

  /** Return the output vector length measured during constructor warm-up. */
  size_t embeddingDimension() const;

  /** Return the selected runtime identifier: "onnx" or "litert". */
  const std::string& runtimeName() const;

  /** Return the resolved model file path retained from ModelConfig. */
  const std::string& modelPath() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace faceid

#endif  // FACEID_INFERENCE_H