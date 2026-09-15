#pragma once

// Diagnostic trace log.
//
// PIXERA's own log messages (HostContext::log_*) are the primary channel, but
// they are not always easy to retrieve, and they are lost entirely if the host
// crashes. This writes an unbuffered line-per-step trace to a file, so the
// last line written is the last step that completed before a crash.
//
// Lines are timestamped to match PIXERA's own log format, so the two can be
// read side by side. The timestamps also make a stall self-evident: a gap
// between consecutive heartbeats is the duration of the stall, which is
// otherwise only inferable.
//
// Defaults to ExtOnnxVision-log.txt in the user's Documents folder. Override
// the location by setting the EXTONNX_TRACE environment variable to a path.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <mutex>
#include <string>

namespace rxext::onnxvision {

inline std::string trace_path() {
  if (const auto* explicit_path = std::getenv("EXTONNX_TRACE"))
    return explicit_path;
  if (const auto* profile = std::getenv("USERPROFILE"))
    return std::string(profile) + R"(\Documents\ExtOnnxVision-log.txt)";
  return "ExtOnnxVision-log.txt";
}

inline void trace(const std::string& message) noexcept try {
  static const std::string path = trace_path();
  static std::mutex mutex;
  const auto lock = std::lock_guard(mutex);

  const auto now = std::chrono::system_clock::now();
  const auto time = std::chrono::system_clock::to_time_t(now);
  const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(
    now.time_since_epoch()).count() % 1000;

  auto parts = std::tm{ };
  char stamp[32] = "??:??:??";
  if (localtime_s(&parts, &time) == 0)
    std::strftime(stamp, sizeof(stamp), "%H:%M:%S", &parts);

  if (auto* f = std::fopen(path.c_str(), "a")) {
    std::fprintf(f, "%s.%03d  %s\n", stamp, static_cast<int>(millis),
      message.c_str());
    std::fflush(f);
    std::fclose(f);
  }
}
catch (...) {
}

} // namespace
