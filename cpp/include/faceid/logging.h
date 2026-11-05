#ifndef FACEID_LOGGING_H
#define FACEID_LOGGING_H

#include <string>

namespace faceid {

/**
 * Read the C++ spdlog YAML and install its configured default logger.
 * @param path Logging configuration file with level, pattern, console, and file.
 * Relative file sink paths resolve from the configuration file's directory.
 * @throws std::runtime_error if YAML, settings, or sink initialization fails.
 */
void initializeLogging(const std::string& path);

}  // namespace faceid

#endif  // FACEID_LOGGING_H
