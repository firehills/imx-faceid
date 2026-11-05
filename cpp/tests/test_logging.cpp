#include "faceid/logging.h"

#include <spdlog/spdlog.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

void expect(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

class TemporaryDirectory {
 public:
  TemporaryDirectory()
      : path(std::filesystem::temp_directory_path() /
             ("faceid-logging-" + std::to_string(
                 std::chrono::steady_clock::now().time_since_epoch().count()))) {
    expect(std::filesystem::create_directory(path), "Create test directory");
  }
  ~TemporaryDirectory() {
    spdlog::shutdown();
    std::filesystem::remove_all(path);
  }
  std::filesystem::path path;
};

void writeFile(const std::filesystem::path& path, const std::string& content) {
  std::ofstream output(path);
  output << content;
  expect(output.good(), "Write test configuration");
}

std::string readFile(const std::filesystem::path& path) {
  std::ifstream input(path);
  expect(input.good(), "Open test log");
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

void expectRejected(const std::filesystem::path& path) {
  const auto logger = spdlog::default_logger();
  try {
    faceid::initializeLogging(path.string());
  } catch (const std::runtime_error& error) {
    expect(std::string(error.what()).find(path.string()) != std::string::npos,
           "Configuration diagnostic must include its path");
    expect(spdlog::default_logger() == logger,
           "Invalid configuration must preserve the active logger");
    return;
  }
  throw std::runtime_error("Invalid logging configuration was accepted");
}

}  // namespace

int main(int argc, char** argv) {
  TemporaryDirectory directory;
  expect(argc == 2, "Example configuration argument is required");
  faceid::initializeLogging(argv[1]);
  expect(spdlog::get_level() == spdlog::level::info, "Example level is info");
  expect(spdlog::default_logger()->sinks().size() == 1,
         "Example must have stderr output only");

  const auto config = directory.path / "spdlog.yaml";
  const std::string levels[] = {
      "trace", "debug", "info", "warn", "error", "critical", "off"};
  const spdlog::level::level_enum severities[] = {
      spdlog::level::trace, spdlog::level::debug, spdlog::level::info,
      spdlog::level::warn, spdlog::level::err, spdlog::level::critical};
  for (size_t threshold = 0; threshold < 7; ++threshold) {
    const std::string filename = levels[threshold] + ".log";
    writeFile(config, "level: " + levels[threshold] +
                          "\npattern: '%n|%l|%v'\nconsole: false\nfile: " +
                          filename + "\n");
    faceid::initializeLogging(config.string());
    for (size_t index = 0; index < 6; ++index) {
      spdlog::log(severities[index], "message-{}", index);
    }
    spdlog::default_logger()->flush();
    const std::string output = readFile(directory.path / filename);
    for (size_t index = 0; index < 6; ++index) {
      expect((output.find("message-" + std::to_string(index)) !=
              std::string::npos) == (index >= threshold),
             "Incorrect filtering for " + levels[threshold]);
    }
    if (threshold < 6) {
      expect(output.find("imx-faceid-cpp|") != std::string::npos,
             "Configured pattern was not applied");
    }
  }

  writeFile(config,
            "level: info\npattern: '%v'\nconsole: true\nfile: concurrent.log\n");
  faceid::initializeLogging(config.string());
  expect(spdlog::default_logger()->sinks().size() == 2, "Both sinks enabled");
  writeFile(config,
            "level: info\npattern: '%v'\nconsole: false\nfile: concurrent.log\n");
  faceid::initializeLogging(config.string());
  spdlog::info("append-marker");
  faceid::initializeLogging(config.string());
  std::vector<std::thread> threads;
  for (int thread = 0; thread < 4; ++thread) {
    threads.emplace_back([thread]() {
      for (int message = 0; message < 25; ++message) {
        spdlog::info("thread-{}-message-{}", thread, message);
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  spdlog::error("flush-marker");
  const std::string output = readFile(directory.path / "concurrent.log");
  expect(output.find("append-marker\n") == 0, "File output must append");
  expect(output.find("flush-marker\n") != std::string::npos,
         "Errors must immediately flush the log");
  size_t lines = 0;
  for (char character : output) {
    lines += character == '\n';
  }
  expect(lines == 102, "Concurrent logging must preserve complete lines");

  expectRejected(directory.path / "missing.yaml");
  const std::string invalid[] = {
      "[]", "level: [", "{}",
      "level: typo\npattern: '%v'\nconsole: true",
      "level: info\npattern: ''\nconsole: true",
      "level: info\npattern: '%v'\nconsole: invalid",
      "level: info\npattern: '%v'\nconsole: false\nfile: null",
      "level: info\npattern: '%v'\nconsole: true\nunknown: true",
      "level: info\npattern: '%v'\nconsole: true\nfile: []",
      "level: info\npattern: '%v'\nconsole: false\nfile: '.'"};
  for (const auto& content : invalid) {
    writeFile(config, content);
    expectRejected(config);
  }
  spdlog::info("shutdown-marker");
  spdlog::shutdown();
  expect(readFile(directory.path / "concurrent.log").find("shutdown-marker\n") !=
             std::string::npos,
         "Shutdown must flush pending logs");
  return 0;
}
