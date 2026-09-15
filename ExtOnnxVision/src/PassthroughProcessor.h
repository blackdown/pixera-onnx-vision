#pragma once

#include "FrameProcessor.h"

namespace rxext::onnxvision {

// Identity processor: copies the frame through unchanged.
//
// This exists to prove the full pixel path — PIXERA texture, download to
// system memory, process, upload, back to PIXERA — independently of any
// model. It is also the fallback whenever a real processor fails to load,
// so a broken model degrades to visible passthrough rather than a black
// layer.
class PassthroughProcessor final : public FrameProcessor {
public:
  const char* name() const noexcept override { return "Passthrough"; }
  bool configure(const FrameView& view) noexcept override;
  bool process(const FrameView& input, FrameBuffer& output) noexcept override;
};

} // namespace
