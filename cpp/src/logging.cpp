#include "faceid/logging.h"

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <spdlog/version.h>
#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

static_assert(SPDLOG_VER_MAJOR == 1 && SPDLOG_VER_MINOR == 17 &&
              SPDLOG_VER_PATCH == 0, "spdlog must be version 1.17.0");

namespace faceid {
namespace {

spdlog::level::level_enum readLevel(const std::string& value) {
  const std::pair<const char*, spdlog::level::level_enum> levels[] = {
      {"trace", spdlog::level::trace}, {"debug", spdlog::level::debug},
      {"info", spdlog::level::info}, {"warn", spdlog::level::warn},
      {"error", spdlog::level::err}, {"critical", spdlog::level::critical},
      {"off", spdlog::level::off}};
  for (const auto& level : levels) {
    if (value == level.first) {
      return level.second;
    }
  }
  throw std::invalid_argument(
      "level must be trace, debug, info, warn, error, critical, or off");
}

}  // namespace

void initializeLogging(const std::string& path) {
  try {
    const YAML::Node root = YAML::LoadFile(path);
    if (!root.IsMap()) {
      throw std::invalid_argument("root must be a mapping");
    }
    for (const auto& entry : root) {
      const std::string key = entry.first.as<std::string>();
      if (key != "level" && key != "pattern" && key != "console" &&
          key != "file") {
        throw std::invalid_argument("unknown setting: " + key);
      }
    }
    const auto level = readLevel(root["level"].as<std::string>());
    const std::string pattern = root["pattern"].as<std::string>();
    const bool console = root["console"].as<bool>();
    if (pattern.empty()) {
      throw std::invalid_argument("pattern must not be empty");
    }
    std::filesystem::path file;
    if (root["file"] && !root["file"].IsNull()) {
      file = root["file"].as<std::string>();
    }
    if (!console && file.empty()) {
      throw std::invalid_argument("enable console or specify a file");
    }

    std::vector<spdlog::sink_ptr> sinks;
    if (console) {
      sinks.push_back(std::make_shared<spdlog::sinks::stderr_color_sink_mt>());
    }
    if (!file.empty()) {
      if (file.is_relative()) {
        file = std::filesystem::path(path).parent_path() / file;
      }
      sinks.push_back(
          std::make_shared<spdlog::sinks::basic_file_sink_mt>(file.string(), false));
    }
    auto logger = std::make_shared<spdlog::logger>(
        "imx-faceid-cpp", sinks.begin(), sinks.end());
    logger->set_level(level);
    logger->set_pattern(pattern);
    logger->flush_on(spdlog::level::err);
    spdlog::set_default_logger(std::move(logger));
  } catch (const YAML::Exception& error) {
    throw std::runtime_error("Invalid logging configuration '" + path +
                             "': " + error.what());
  } catch (const std::invalid_argument& error) {
    throw std::runtime_error("Invalid logging configuration '" + path +
                             "': " + error.what());
  } catch (const spdlog::spdlog_ex& error) {
    throw std::runtime_error("Cannot initialize logging from '" + path +
                             "': " + error.what());
  }
}

}  // namespace faceid
