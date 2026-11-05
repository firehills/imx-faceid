#ifndef FACEID_CONFIG_H
#define FACEID_CONFIG_H

#include <array>
#include <stdexcept>
#include <string>
#include <vector>

namespace faceid {

/** Raised when a YAML configuration is missing or contains invalid values. */
class ConfigurationError : public std::runtime_error {
 public:
  /** Create a configuration error with a user-facing explanation. */
  explicit ConfigurationError(const std::string& message);
};

/**
 * NATS endpoint, authentication credentials, and request subject routing.
 * The loader enforces exactly one authentication mode and unique non-empty
 * subjects; empty credential fields indicate that mode is not selected.
 */
struct NatsConfig {
  std::string url;                 ///< NATS server URL.
  std::string enrol_subject;       ///< Subject receiving enrolment envelopes.
  std::string query_subject;       ///< Subject receiving identity-query envelopes.
  std::string user_info_subject;   ///< Subject receiving user-info envelopes.
  std::string delete_subject;      ///< Subject receiving user-deletion envelopes.
  std::string debug_subject;       ///< Subject receiving diagnostic envelopes.
  std::string token;               ///< Optional token authentication value.
  std::string username;            ///< Username; paired with password.
  std::string password;            ///< Password; paired with username.
  std::string credentials_file;    ///< Optional NATS credentials-file path.
};

/**
 * Qdrant endpoint and collection schema used by the gRPC vector store.
 * The HTTP port is retained for configuration compatibility; the C++ client
 * uses grpc_port and validates collection vector size and metric at startup.
 */
struct QdrantConfig {
  std::string host;                 ///< Qdrant hostname or IP address.
  int port;                         ///< Configured HTTP API port.
  int grpc_port;                    ///< Port used by the C++ gRPC client.
  std::string api_key;              ///< Optional gRPC API-key metadata value.
  bool https;                       ///< Use TLS for the gRPC channel.
  std::string metric;               ///< Validated "cosine" or "euclidean" metric.
  std::string raw_collection;       ///< Collection retaining each enrolment.
  std::string normalized_collection;///< Collection retaining one vector per user.
};

/**
 * Inference backend selection and the preprocessing contract for its input.
 * Values are validated by loadConfig before model initialization; mean and
 * stddev are channel-ordered RGB/BGR normalization constants.
 */
struct ModelConfig {
  std::string path;     ///< Resolved model file path.
  std::string runtime;  ///< "onnx" or "litert", selected explicitly or by suffix.
  std::string input_layout;  ///< Tensor memory order: "NCHW" or "NHWC".
  int input_size;            ///< Square model input width and height in pixels.
  std::string input_name;    ///< Optional explicit input tensor name.
  std::string output_name;   ///< Optional explicit output tensor name.
  std::string color_order;   ///< Model channel order: "RGB" or "BGR".
  float pixel_scale;         ///< Multiplier applied to each source pixel.
  std::array<float, 3> mean; ///< Per-channel value subtracted after scaling.
  std::array<float, 3> stddev;  ///< Per-channel nonzero normalization divisor.
  std::string alignment;        ///< "none" or "five_point".
  std::vector<std::array<float, 2> > reference_keypoints;  ///< Five target [x,y] points.
  bool honor_preprocessed_flag;  ///< Skip spatial preprocessing when request says so.
  std::string delegate_library;  ///< Optional external LiteRT delegate library.
  std::string delegate_options;  ///< Delegate options serialized as YAML text.
};

/** Fully parsed settings passed from startup into service and dependency setup. */
struct AppConfig {
  NatsConfig nats;       ///< Message transport configuration.
  QdrantConfig qdrant;   ///< Persistent vector-store configuration.
  ModelConfig model;     ///< Inference and image preprocessing configuration.
};

/**
 * Load application YAML into typed NATS, Qdrant, and model settings.
 * Relative model paths are resolved from the YAML file's directory.
 * @param path Application YAML file to read.
 * @return Complete validated configuration.
 * @throws ConfigurationError for missing files, invalid values, or
 *         incompatible setting combinations.
 */
AppConfig loadConfig(const std::string& path);

}  // namespace faceid

#endif  // FACEID_CONFIG_H