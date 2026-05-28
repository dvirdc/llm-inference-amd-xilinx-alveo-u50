// logging.hpp -- tiny printf-style logger used everywhere. Three levels.
//
// Why a header-only mini-logger instead of spdlog / glog / boost::log:
//   * Zero deps -- this project is meant to be cloneable + buildable with
//     just g++ and XRT. Pulling in a log lib would mean an extra package
//     dep on the U50 build machine.
//   * The volume of log output is small (per-token messages, build info)
//     so the overhead of stdio is fine.
//
// Use:
//   LOG_INFO("loaded checkpoint %s (vocab=%d)", path.c_str(), v);
//   LOG_WARN("checkpoint has %zu extra bytes after wcls", extra);
//   LOG_ERR ("failed to read header from %s", path.c_str());

#pragma once

#include <cstdio>
#include <cstdarg>

namespace m2 {

inline void log_at(const char* level, const char* fmt, va_list ap) {
    std::fprintf(stderr, "[%s] ", level);
    std::vfprintf(stderr, fmt, ap);
    std::fprintf(stderr, "\n");
}

inline void log_info(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt); log_at("INFO", fmt, ap); va_end(ap);
}
inline void log_warn(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt); log_at("WARN", fmt, ap); va_end(ap);
}
inline void log_err(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt); log_at("ERROR", fmt, ap); va_end(ap);
}

} // namespace m2

#define LOG_INFO(...) ::m2::log_info(__VA_ARGS__)
#define LOG_WARN(...) ::m2::log_warn(__VA_ARGS__)
#define LOG_ERR(...)  ::m2::log_err(__VA_ARGS__)
