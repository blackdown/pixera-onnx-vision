#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace rxext::onnxvision {

// Description of a CPU-side image buffer handed to a FrameProcessor.
struct FrameView {
  const uint8_t* data;
  size_t width;
  size_t height;
  size_t pitch;             // bytes per row, may exceed width * bytes_per_pixel
  size_t bytes_per_pixel;
  std::string format_name;  // e.g. "R16G16B16A16_SFLOAT"

  // True when channel 0 is blue and channel 2 is red, which is how PIXERA
  // writes into the extension's texture. A model expecting RGB must read the
  // channels in reverse; see README "Colour and orientation".
  bool bgr_order;
};

// Mutable destination buffer a FrameProcessor writes its result into.
struct FrameBuffer {
  uint8_t* data;
  size_t width;
  size_t height;
  size_t pitch;
  size_t bytes_per_pixel;
};

// What a model found, reduced to numbers rather than pixels.
//
// A segmentation or matting model already knows where the subject is and how
// much of the frame it fills; that is worth handing back to the timeline, not
// just painting into alpha. These are published as output parameters so a
// PIXERA layer can read them.
struct FrameMeasurements {
  bool valid{ };
  double coverage{ };   // 0..1, fraction of the frame the subject occupies
  double centre_x{ };   // 0..1 across the frame, 0.5 when nothing is found
  double centre_y{ };
  double mean{ };       // mean map value, e.g. average depth
};

// The seam between the streaming plumbing and the vision model.
//
// Everything above this interface deals with PIXERA, textures and frame
// timing; everything below it deals with pixels and inference. Swapping the
// model means providing a different implementation of this interface and
// nothing else.
class FrameProcessor {
public:
  virtual ~FrameProcessor() = default;

  // Human readable name, used in log messages.
  virtual const char* name() const noexcept = 0;

  // Called when the frame geometry changes, before any process() call with
  // those dimensions. Returns false if this processor cannot handle the
  // format, in which case frames are passed through untouched.
  virtual bool configure(const FrameView& view) noexcept = 0;

  // Transform one frame. Called on a worker thread, never on the render
  // thread. Must not throw. Returns false if the frame could not be
  // processed, in which case the input is passed through untouched.
  virtual bool process(const FrameView& input, FrameBuffer& output) noexcept = 0;

  // Milliseconds spent in the last inference call, for reporting to the host.
  // Zero for processors that do no inference. Virtual rather than a
  // dynamic_cast because the SDK builds with RTTI disabled (/GR-).
  virtual double last_inference_ms() const noexcept { return 0.0; }

  // One-line breakdown of where time went, for the host log.
  virtual std::string timing_summary() const { return { }; }

  // Human-readable state, surfaced in PIXERA so an operator can see what the
  // extension is doing. Model loading can take minutes, and silence during
  // that is indistinguishable from a fault.
  virtual std::string status() const { return "Ready"; }

  // True while a model is loading and frames are passing through untouched.
  virtual bool is_loading() const noexcept { return false; }

  // What the last processed frame measured. Invalid for processors that
  // produce no map, and before the first frame.
  virtual FrameMeasurements measurements() const noexcept { return { }; }
};

} // namespace
