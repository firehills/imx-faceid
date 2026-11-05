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

/** NATS connection, authentication, and subject settings. */
struct NatsConfig {
  std::string url;
  std::string enrol_subject;
  std::string query_subject;
  std::string user_info_subject;
  std::string delete_subject;
  std::string debug_subject;
  std::string token;
  std::string username;
  std::string password;
  std::string credentials_file;
};

/** Qdrant endpoint, authentication, collection, and distance settings. */
struct QdrantConfig {
  std::string host;
  int port;
  int grpc_port;
  std::string api_key;
  bool https;
  std::string metric;
  std::string raw_collection;
  std::string normalized_collection;
};

/** Model loading and image preprocessing settings. */
struct ModelConfig {
  std::string path;
  std::string runtime;
  std::string input_layout;
  int input_size;
  std::string input_name;
  std::string output_name;
  std::string color_order;
  float pixel_scale;
  std::array<float, 3> mean;
  std::array<float, 3> stddev;
  std::string alignment;
  std::vector<std::array<float, 2> > reference_keypoints;
  bool honor_preprocessed_flag;
  std::string delegate_library;
  std::string delegate_options;
};

/** Complete validated configuration for the face identification service. */
struct AppConfig {
  NatsConfig nats;
  QdrantConfig qdrant;
  ModelConfig model;
};

/** Load and validate YAML settings, resolving relative model paths from the file. */
AppConfig loadConfig(const std::string& path);

}  // namespace faceid

#endif  // FACEID_CONFIG_H