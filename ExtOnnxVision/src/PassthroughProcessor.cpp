
#include "PassthroughProcessor.h"
#include <algorithm>
#include <cstring>

namespace rxext::onnxvision {

bool PassthroughProcessor::configure(const FrameView&) noexcept {
  // Format agnostic: it copies bytes without interpreting them.
  return true;
}

bool PassthroughProcessor::process(const FrameView& input,
    FrameBuffer& output) noexcept {
  if (!input.data || !output.data)
    return false;
  if (input.bytes_per_pixel != output.bytes_per_pixel)
    return false;

  const auto rows = std::min(input.height, output.height);
  const auto row_bytes = std::min(input.width, output.width) * input.bytes_per_pixel;

  for (auto y = size_t{ }; y < rows; ++y)
    std::memcpy(output.data + y * output.pitch,
                input.data + y * input.pitch,
                row_bytes);
  return true;
}

} // namespace
