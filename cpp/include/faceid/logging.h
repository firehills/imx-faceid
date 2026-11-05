#ifndef FACEID_LOGGING_H
#define FACEID_LOGGING_H

#include <string>

namespace faceid {

/** Configure the default spdlog logger from YAML; invalid settings throw. */
void initializeLogging(const std::string& path);

}  // namespace faceid

#endif  // FACEID_LOGGING_H
