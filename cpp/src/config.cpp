#include "faceid/config.h"

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>

namespace faceid {
namespace {

/**
 * Fetch a required, non-null child from a YAML mapping.
 * @param node Mapping that owns the setting.
 * @param key Setting name within the mapping.
 * @param section Configuration section used to identify the setting in errors.
 * @return The YAML node to be converted by the caller.
 */
YAML::Node required(const YAML::Node& node, const std::string& key,
                    const std::string& section) {
  const YAML::Node value = node[key];
  if (!value || value.IsNull()) {
    throw ConfigurationError("Missing required configuration: " + section +
                             "." + key);
  }
  return value;
}

/**
 * Read a named configuration section and ensure it is a YAML mapping.
 * @param root Top-level configuration mapping.
 * @param name Required section name.
 * @return The section node; missing or non-mapping sections are rejected.
 */
YAML::Node section(const YAML::Node& root, const std::string& name) {
  const YAML::Node value = root[name];
  if (!value || !value.IsMap()) {
    throw ConfigurationError("Configuration section '" + name +
                             "' must be a mapping");
  }
  return value;
}

/**
 * Convert a YAML scalar into a string and normalize conversion failures.
 * @param node Scalar value supplied by YAML.
 * @param field Full setting name, used to identify invalid input.
 * @return The converted string.
 */
std::string readString(const YAML::Node& node, const std::string& field) {
  try {
    return node.as<std::string>();
  } catch (const YAML::Exception&) {
    throw ConfigurationError("'" + field + "' must be a string");
  }
}

/**
 * Convert a YAML scalar into a finite float.
 * @param node Scalar value supplied by YAML.
 * @param field Setting name used in conversion and range errors.
 * @return A finite floating-point value.
 */
float readFiniteFloat(const YAML::Node& node, const std::string& field) {
  float value = 0.0f;
  try {
    value = node.as<float>();
  } catch (const YAML::Exception&) {
    throw ConfigurationError("'" + field + "' must be a number");
  }
  if (!std::isfinite(value)) {
    throw ConfigurationError("'" + field + "' must be finite");
  }
  return value;
}

/**
 * Convert a YAML scalar into a strictly positive integer.
 * @param node Scalar value supplied by YAML.
 * @param field Setting name used in conversion and range errors.
 * @return A positive integer.
 */
int readPositiveInt(const YAML::Node& node, const std::string& field) {
  int value = 0;
  try {
    value = node.as<int>();
  } catch (const YAML::Exception&) {
    throw ConfigurationError("'" + field + "' must be an integer");
  }
  if (value < 1) {
    throw ConfigurationError("'" + field + "' must be positive");
  }
  return value;
}

/**
 * Read a three-element sequence of finite floats, used for RGB normalization.
 * @param node YAML sequence containing exactly three scalar values.
 * @param field Setting name used to report malformed values.
 * @return The values in their original channel order.
 */
std::array<float, 3> readTriple(const YAML::Node& node,
                                const std::string& field) {
  if (!node.IsSequence() || node.size() != 3) {
    throw ConfigurationError("'" + field + "' must contain exactly three numbers");
  }
  std::array<float, 3> result;
  for (size_t index = 0; index < result.size(); ++index) {
    result[index] = readFiniteFloat(node[index], field);
  }
  return result;
}

/**
 * Convert an optional YAML scalar to a string, using empty text when absent.
 * @param node Optional scalar node.
 * @param field Setting name used if conversion fails.
 * @return The configured string, or an empty string for absent/null input.
 */
std::string optionalString(const YAML::Node& node, const char* field) {
  if (!node || node.IsNull()) {
    return std::string();
  }
  return readString(node, field);
}

/**
 * Read an optional boolean while preserving the caller's default.
 * @param node Optional scalar node.
 * @param field Setting name used if conversion fails.
 * @param fallback Value used for absent or null input.
 * @return The configured boolean or fallback.
 */
bool optionalBool(const YAML::Node& node, const char* field, bool fallback) {
  if (!node || node.IsNull()) {
    return fallback;
  }
  try {
    return node.as<bool>();
  } catch (const YAML::Exception&) {
    throw ConfigurationError(std::string("'") + field +
                             "' must be true or false");
  }
}

/**
 * Read five finite [x, y] reference points for face alignment.
 * @param node Optional YAML sequence; absent input means no reference points.
 * @return Ordered landmark coordinates, or an empty vector when not configured.
 */
std::vector<std::array<float, 2> > readReferencePoints(
    const YAML::Node& node) {
  std::vector<std::array<float, 2> > result;
  if (!node || node.IsNull()) {
    return result;
  }
  if (!node.IsSequence() || node.size() != 5) {
    throw ConfigurationError(
        "'model.reference_keypoints' must be five [x, y] pairs");
  }
  result.reserve(5);
  for (size_t index = 0; index < 5; ++index) {
    const YAML::Node point = node[index];
    if (!point.IsSequence() || point.size() != 2) {
      throw ConfigurationError(
          "'model.reference_keypoints' must be five [x, y] pairs");
    }
    result.push_back({{readFiniteFloat(point[0], "model.reference_keypoints"),
                       readFiniteFloat(point[1], "model.reference_keypoints")}});
  }
  return result;
}

/** Normalize ASCII uppercase letters for case-insensitive enum settings. */
std::string lowercase(std::string value) {
  for (size_t index = 0; index < value.size(); ++index) {
    if (value[index] >= 'A' && value[index] <= 'Z') {
      value[index] = static_cast<char>(value[index] - 'A' + 'a');
    }
  }
  return value;
}

/** Normalize ASCII lowercase letters for case-insensitive layout settings. */
std::string uppercase(std::string value) {
  for (size_t index = 0; index < value.size(); ++index) {
    if (value[index] >= 'a' && value[index] <= 'z') {
      value[index] = static_cast<char>(value[index] - 'a' + 'A');
    }
  }
  return value;
}

}  // namespace

/** Preserve a validated configuration diagnostic as a standard exception. */
ConfigurationError::ConfigurationError(const std::string& message)
    : std::runtime_error(message) {}

/**
 * Load and validate all application settings from one YAML document.
 *
 * The loader converts NATS, Qdrant, and model sections into typed structures,
 * fills documented defaults, normalizes enum-like values, resolves model paths
 * relative to the YAML file, and rejects invalid combinations before startup.
 * @param path Filesystem path of the application YAML document.
 * @return Complete settings safe to pass to the service, inference, and store.
 */
AppConfig loadConfig(const std::string& path) {
  YAML::Node root;
  try {
    root = YAML::LoadFile(path);
  } catch (const YAML::Exception& error) {
    throw ConfigurationError("Cannot load configuration '" + path +
                             "': " + error.what());
  }
  if (!root.IsMap()) {
    throw ConfigurationError("Configuration root must be a mapping");
  }

  AppConfig config;
  // Read NATS connection settings and default subject names, then ensure the
  // subjects are distinct and only one authentication method is configured.
  const YAML::Node nats = section(root, "nats");
  const YAML::Node subjects = nats["subjects"];
  if (subjects && !subjects.IsMap()) {
    throw ConfigurationError("'nats.subjects' must be a mapping");
  }
  config.nats.url = readString(required(nats, "url", "nats"), "nats.url");
  config.nats.enrol_subject =
      subjects["enrol"] ? readString(subjects["enrol"], "nats.subjects.enrol")
                        : "faceid.enrol";
  config.nats.query_subject =
      subjects["query"] ? readString(subjects["query"], "nats.subjects.query")
                        : "faceid.query";
  config.nats.user_info_subject =
      subjects["user_info"]
          ? readString(subjects["user_info"], "nats.subjects.user_info")
          : "faceid.user_info";
  config.nats.delete_subject =
      subjects["delete"]
          ? readString(subjects["delete"], "nats.subjects.delete")
          : "faceid.delete";
  config.nats.debug_subject =
      subjects["debug"] ? readString(subjects["debug"], "nats.subjects.debug")
                        : "faceid.debug";
  config.nats.token = optionalString(nats["token"], "nats.token");
  config.nats.username = optionalString(nats["username"], "nats.username");
  config.nats.password = optionalString(nats["password"], "nats.password");
  config.nats.credentials_file =
      optionalString(nats["credentials_file"], "nats.credentials_file");

  const std::string subjects_to_check[] = {
      config.nats.enrol_subject, config.nats.query_subject,
      config.nats.user_info_subject, config.nats.delete_subject,
      config.nats.debug_subject};
  std::set<std::string> unique_subjects;
  for (size_t index = 0; index < 5; ++index) {
    if (subjects_to_check[index].empty() ||
        !unique_subjects.insert(subjects_to_check[index]).second) {
      throw ConfigurationError("NATS subjects must be non-empty and unique");
    }
  }
  const int auth_modes = (!config.nats.token.empty()) +
                         (!config.nats.username.empty() ||
                          !config.nats.password.empty()) +
                         (!config.nats.credentials_file.empty());
  if (auth_modes > 1) {
    throw ConfigurationError(
        "Configure only one NATS authentication method: token, "
        "username/password, or credentials_file");
  }
  if (config.nats.username.empty() != config.nats.password.empty()) {
    throw ConfigurationError(
        "NATS username and password must be configured together");
  }

  // Validate Qdrant endpoint, collections, and the score metric used by queries.
  const YAML::Node qdrant = section(root, "qdrant");
  config.qdrant.host = readString(required(qdrant, "host", "qdrant"),
                                  "qdrant.host");
  config.qdrant.port = readPositiveInt(required(qdrant, "port", "qdrant"),
                                       "qdrant.port");
  config.qdrant.grpc_port = readPositiveInt(
      required(qdrant, "grpc_port", "qdrant"), "qdrant.grpc_port");
  config.qdrant.api_key = optionalString(qdrant["api_key"], "qdrant.api_key");
  config.qdrant.https = optionalBool(qdrant["https"], "qdrant.https", false);
  config.qdrant.metric =
      lowercase(readString(required(qdrant, "metric", "qdrant"),
                           "qdrant.metric"));
  if (config.qdrant.metric != "cosine" && config.qdrant.metric != "euclidean") {
    throw ConfigurationError("'qdrant.metric' must be 'cosine' or 'euclidean'");
  }
  config.qdrant.raw_collection =
      qdrant["raw_collection"]
          ? readString(qdrant["raw_collection"], "qdrant.raw_collection")
          : "raw_enrol";
  config.qdrant.normalized_collection =
      qdrant["normalized_collection"]
          ? readString(qdrant["normalized_collection"],
                       "qdrant.normalized_collection")
          : "normalized_enrol";
  if (config.qdrant.raw_collection.empty() ||
      config.qdrant.normalized_collection.empty() ||
      config.qdrant.raw_collection == config.qdrant.normalized_collection) {
    throw ConfigurationError("Qdrant collection names must be non-empty and different");
  }

  // Resolve the model location and validate tensor layout, preprocessing,
  // runtime selection, and the compatibility of optional LiteRT settings.
  const YAML::Node model = section(root, "model");
  config.model.path = readString(required(model, "path", "model"), "model.path");
  std::filesystem::path model_path(config.model.path);
  if (model_path.is_relative()) {
    model_path = std::filesystem::path(path).parent_path() / model_path;
  }
  config.model.path = model_path.lexically_normal().string();
  config.model.runtime = model["runtime"]
                             ? lowercase(readString(model["runtime"], "model.runtime"))
                             : "auto";
  if (config.model.runtime == "auto") {
    const size_t dot = config.model.path.find_last_of('.');
    const std::string extension =
        dot == std::string::npos ? std::string() : lowercase(config.model.path.substr(dot));
    config.model.runtime = extension == ".onnx" ? "onnx" :
                           extension == ".tflite" ? "litert" : "";
  }
  if (config.model.runtime != "onnx" && config.model.runtime != "litert") {
    throw ConfigurationError(
        "Model runtime must be onnx or litert, or inferable from its file extension");
  }
  config.model.input_layout =
      uppercase(readString(required(model, "input_layout", "model"),
                           "model.input_layout"));
  if (config.model.input_layout != "NCHW" &&
      config.model.input_layout != "NHWC") {
    throw ConfigurationError("'model.input_layout' must be NCHW or NHWC");
  }
  config.model.input_size =
      readPositiveInt(required(model, "input_size", "model"), "model.input_size");
  config.model.input_name = optionalString(model["input_name"], "model.input_name");
  config.model.output_name =
      optionalString(model["output_name"], "model.output_name");
  config.model.color_order =
      uppercase(readString(required(model, "color_order", "model"),
                           "model.color_order"));
  if (config.model.color_order != "RGB" && config.model.color_order != "BGR") {
    throw ConfigurationError("'model.color_order' must be RGB or BGR");
  }
  config.model.pixel_scale =
      readFiniteFloat(required(model, "pixel_scale", "model"),
                      "model.pixel_scale");
  config.model.mean = readTriple(required(model, "mean", "model"), "model.mean");
  config.model.stddev = readTriple(required(model, "std", "model"), "model.std");
  for (size_t index = 0; index < config.model.stddev.size(); ++index) {
    if (config.model.stddev[index] == 0.0f) {
      throw ConfigurationError("Values in 'model.std' must be non-zero");
    }
  }
  config.model.alignment =
      model["alignment"]
          ? lowercase(readString(model["alignment"], "model.alignment"))
          : "none";
  if (config.model.alignment != "none" &&
      config.model.alignment != "five_point") {
    throw ConfigurationError("'model.alignment' must be none or five_point");
  }
  config.model.reference_keypoints =
      readReferencePoints(model["reference_keypoints"]);
  if (config.model.alignment == "five_point" &&
      config.model.reference_keypoints.empty()) {
    throw ConfigurationError(
        "'model.reference_keypoints' is required when alignment is five_point");
  }
  config.model.honor_preprocessed_flag = optionalBool(
      model["honor_preprocessed_flag"], "model.honor_preprocessed_flag", true);
  config.model.delegate_library =
      optionalString(model["delegate_library"], "model.delegate_library");
  if (model["delegate_options"] && !model["delegate_options"].IsMap()) {
    throw ConfigurationError("'model.delegate_options' must be a mapping");
  }
  config.model.delegate_options = model["delegate_options"]
                                      ? YAML::Dump(model["delegate_options"])
                                      : "{}";
  if (config.model.runtime == "onnx" &&
      !config.model.delegate_library.empty()) {
    throw ConfigurationError(
        "LiteRT delegate settings cannot be used with an ONNX model");
  }

  // Fail during configuration loading instead of deferring a missing-file
  // error until the model runtime is initialized.
  std::ifstream model_file(config.model.path.c_str(), std::ios::binary);
  if (!model_file.good()) {
    throw ConfigurationError("Model file does not exist: " + config.model.path);
  }
  return config;
}

}  // namespace faceid