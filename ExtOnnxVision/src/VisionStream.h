#pragma once

#include "rxext_client.h"
#include "FrameProcessor.h"
#include "TextureReader.h"
#include "TensorRTProcessor.h"
#include "d3d11/d3d11.h"
#include <atomic>
#include <chrono>
#include <future>
#include <utility>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace rxext::onnxvision {

// The processor that runs models. Kept as an alias so the stream does not
// name TensorRT directly; anything implementing FrameProcessor with the same
// Config would drop in here.
using Processor = TensorRTProcessor;

// One PIXERA layer's worth of vision processing.
//
// Exposes two texture parameters:
//   "source"  (input)  - drag a PIXERA Resource onto this slot; the camera's
//                        Live Input goes here. This is how content reaches
//                        the extension.
//   "sampler" (output) - the result, usable in PIXERA like any other source.
//
// The Mode parameter selects how much of the pipeline runs. Process is the
// default and the only mode that produces correct output; the other two are
// diagnostics.
class VisionStream final : public rxext::InputStream {
public:
  VisionStream(std::shared_ptr<d3d11::Device> d3d_device, ValueSet settings);
  ~VisionStream() override;

  bool initialize() noexcept override;
  bool update_settings(ValueSet settings) noexcept override;
  ValueSet get_state() noexcept override;
  bool update() noexcept override;

private:
  // Values of the "mode" parameter, in the order they appear in PIXERA.
  enum Mode : int {
    ModeIdle = 0,        // publish the placeholder colour only
    ModePassthrough = 1, // republish the source texture directly, GPU side
    ModeReadback = 2,    // read to system memory, process, publish back
  };

  // Where a frame is in the download -> process -> upload cycle. Only one
  // frame is in flight at a time; if the processor is slower than the frame
  // rate, frames are skipped rather than queued.
  enum class Stage {
    Idle,
    Downloading,
    Processing,
    ReadyToUpload,
    Uploading,
  };

  void publish_placeholder() noexcept;
  void ensure_output_texture(const TextureDesc& desc) noexcept;
  void begin_download() noexcept;
  void capture_input(const BufferDesc& buffer) noexcept;
  void run_processor() noexcept;
  void begin_upload() noexcept;
  void log_geometry(const TextureDesc& desc) noexcept;
  void ensure_processor() noexcept;
  void publish_status() noexcept;
  void publish_measurements() noexcept;
  void drain_log() noexcept;
  void queue_log(bool is_error, const std::string& text) noexcept;
  Mode mode() const noexcept;

  ValueSet m_settings;
  std::shared_ptr<d3d11::Device> m_d3d_device;
  std::unique_ptr<TextureReader> m_reader;

  ParameterTexture* m_source{ };   // PIXERA content in
  ParameterTexture* m_sampler{ };  // result out
  ParameterInt* m_mode{ };
  ParameterBool* m_swap_rb{ };
  ParameterString* m_status{ };

  // Flipping this reloads the model. See ensure_processor().
  ParameterBool* m_reload{ };
  bool m_reload_seen{ };


  // What the model found, as numbers rather than pixels. Output parameters,
  // so PIXERA reads them off the layer.
  ParameterValue* m_out_coverage{ };
  ParameterValue* m_out_centre_x{ };
  ParameterValue* m_out_centre_y{ };
  ParameterValue* m_out_mean{ };
  FrameMeasurements m_published_measurements{ };
  ParameterString* m_model_path{ };
  ParameterInt* m_normalisation{ };
  ParameterInt* m_activation{ };
  ParameterInt* m_target{ };
  ParameterInt* m_size{ };
  ParameterInt* m_trt_effort{ };

  TextureRef m_placeholder;
  TextureRef m_output_texture;

  std::unique_ptr<FrameProcessor> m_processor;
  // Superseded processors still compiling, held until they finish so that
  // destroying one never blocks the render thread.
  std::vector<std::unique_ptr<FrameProcessor>> m_retired;
  // Built on a worker thread and adopted when ready, so neither construction
  // nor loading ever runs on the host's render thread.
  std::future<std::unique_ptr<Processor>> m_pending;

  std::atomic<Stage> m_stage{ Stage::Idle };

  std::mutex m_mutex;
  std::vector<uint8_t> m_input_bytes;
  std::vector<uint8_t> m_output_bytes;
  size_t m_width{ };
  size_t m_height{ };
  size_t m_pitch{ };
  size_t m_bytes_per_pixel{ };
  Format m_format{ Format::None };

  // Logged geometry, so the log records changes rather than every frame.
  size_t m_logged_width{ };
  size_t m_logged_height{ };
  Format m_logged_format{ Format::None };

  Mode m_logged_mode{ ModeIdle };
  bool m_processor_configured{ };
  bool m_reported_first_frame{ };
  bool m_warned_read_failed{ };
  bool m_reported_read{ };
  bool m_reported_upload{ };
  std::string m_loaded_model;
  std::string m_published_status;
  std::chrono::steady_clock::time_point m_status_published_at;
  // Host logging is only safe from the thread the host calls us on, so worker
  // threads queue here and update() drains.
  std::mutex m_log_mutex;
  std::vector<std::pair<bool, std::string>> m_log_queue;
  int m_loaded_options{ -1 };
  uint64_t m_frames{ };
  uint64_t m_ticks{ };
  double m_inference_ms{ };
  double m_readback_ms{ };
};

} // namespace
