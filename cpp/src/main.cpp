#include <csignal>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include "faceid/config.h"
#include "faceid/inference.h"
#include "faceid/logging.h"
#include "faceid/service.h"
#include "faceid/vector_store.h"

namespace {

volatile std::sig_atomic_t g_stop_requested = 0;

void requestStop(int) {
  if (g_stop_requested != 0) {
    std::_Exit(130);
  }
  g_stop_requested = 1;
}

struct Options {
  std::string config_path = "config.yaml";
  std::string logging_path = "spdlog.yaml";
};

Options parseOptions(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument(argv[index]);
    if (argument == "--config" && index + 1 < argc) {
      options.config_path = argv[++index];
    } else if (argument == "--logging-config" && index + 1 < argc) {
      options.logging_path = argv[++index];
    } else if (argument == "--help" || argument == "-h") {
      std::cout << "Usage: imx_faceid_cpp [--config PATH] "
                   "[--logging-config PATH]" << std::endl;
      std::exit(0);
    } else {
      throw std::invalid_argument("Unknown or incomplete argument: " + argument);
    }
  }
  return options;
}

}  // namespace

int main(int argc, char** argv) {
  const auto startup_logger = spdlog::stderr_color_mt("startup");
  spdlog::set_default_logger(startup_logger);
  try {
    const Options options = parseOptions(argc, argv);
    faceid::initializeLogging(options.logging_path);
    const faceid::AppConfig config = faceid::loadConfig(options.config_path);
    faceid::EmbeddingModel model(config.model);
    faceid::VectorStore store(config.qdrant, model.embeddingDimension());
    faceid::FaceIdService service(config, model, store);
    std::signal(SIGINT, requestStop);
    std::signal(SIGTERM, requestStop);
    service.run(g_stop_requested);
    spdlog::info("imx-faceid-cpp stopped");
    spdlog::shutdown();
    return 0;
  } catch (const std::exception& error) {
    startup_logger->critical("imx-faceid-cpp: {}", error.what());
    spdlog::shutdown();
    return 2;
  }
}