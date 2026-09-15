#pragma once

#include "FrameProcessor.h"
#include "ModelOptions.h"
#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace rxext::onnxvision {

// Runs an ONNX vision model through TensorRT.
//
// The model is parsed with nvonnxparser and compiled with nvinfer into an
// engine optimised for the GPU in the machine, which is cached so the cost is
// paid once per model per GPU. TensorRT depends on nothing but Windows and
// the NVIDIA driver, so the extension needs no CUDA toolkit, cuDNN or other
// runtime beside it.
//
// TensorRT is not linked. Its entry points are resolved at run time and handed
// the version the loaded library reports for itself, rather than one fixed at
// compile time, so one build works against whichever TensorRT 10 release the
// machine has.
//
// TensorRT has no per-operator fallback: a model using an operator it cannot
// build does not load at all, and the log names the operator.
class TensorRTProcessor final : public FrameProcessor {
public:
  using Logger = std::function<void(const std::string&)>;

  struct Config {
    std::string model_path;
    Normalisation normalisation{ Normalisation::ImageNet };
    Activation activation{ Activation::Auto };
    OutputTarget target{ OutputTarget::Alpha };
    int size_override{ };
    int trt_build_effort{ 3 };
  };

  TensorRTProcessor(Config config, Logger log_info, Logger log_error);
  ~TensorRTProcessor() override;

  const char* name() const noexcept override { return "TensorRT"; }
  bool configure(const FrameView& view) noexcept override;
  bool process(const FrameView& input, FrameBuffer& output) noexcept override;
  double last_inference_ms() const noexcept override { return m_last_inference_ms; }
  std::string timing_summary() const override;
  std::string status() const override;
  bool is_loading() const noexcept override;
  FrameMeasurements measurements() const noexcept override;

  void begin_loading() noexcept { start_loading(); }

  // The TensorRT this actually bound to, for the log. Empty until loaded.
  std::string library_version() const;

private:
  void start_loading() noexcept;
  bool build_engine() noexcept;
  bool load_or_compile(const std::string& cache_file) noexcept;
  void preprocess(const FrameView& input) noexcept;
  void decode_map(const float* data, size_t count) noexcept;
  void measure_map() noexcept;
  void write_map(const FrameView& input, FrameBuffer& output) noexcept;
  void write_colour(const float* data, const FrameView& input,
    FrameBuffer& output) noexcept;
  void resolve_colour_scale(const float* data, size_t count) noexcept;

  const Config m_config;
  Logger m_log_info;
  Logger m_log_error;

  struct Impl;
  std::unique_ptr<Impl> m_impl;

  std::thread m_loader;
  std::atomic<bool> m_session_ready{ };
  std::atomic<bool> m_session_failed{ };
  std::atomic<bool> m_loading_started{ };
  std::atomic<bool> m_engine_was_cached{ };
  std::chrono::steady_clock::time_point m_load_started_at;
  std::string m_failure;
  mutable std::mutex m_status_mutex;

  enum class OutputKind { Unsupported, SingleChannelMap, ColourImage };

  size_t m_in_width{ }, m_in_height{ };
  bool m_input_is_nhwc{ };
  size_t m_out_width{ }, m_out_height{ }, m_out_channels{ };
  bool m_output_is_nhwc{ };
  OutputKind m_output_kind{ OutputKind::Unsupported };
  Activation m_resolved_activation{ Activation::None };
  float m_colour_scale{ 1.0f };
  float m_colour_bias{ 0.0f };
  bool m_colour_scale_resolved{ };

  std::vector<float> m_input_tensor;
  std::vector<float> m_output_tensor;
  std::vector<float> m_map;

  mutable std::mutex m_measure_mutex;
  FrameMeasurements m_measurements;

  double m_last_inference_ms{ };
  std::atomic<double> m_reported_ms{ };
  double m_last_preprocess_ms{ };
  double m_last_postprocess_ms{ };
  bool m_logged_output_kind{ };
};

} // namespace
