#include "TensorRTProcessor.h"
#include "Trace.h"

#include <mutex>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <DirectXPackedVector.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <thread>
#include <windows.h>

namespace rxext::onnxvision {
namespace {

constexpr float kImageNetMean[3] = { 0.485f, 0.456f, 0.406f };
constexpr float kImageNetStd[3]  = { 0.229f, 0.224f, 0.225f };

float half_to_float(uint16_t h) {
  return DirectX::PackedVector::XMConvertHalfToFloat(h);
}

uint16_t float_to_half(float f) {
  return DirectX::PackedVector::XMConvertFloatToHalf(f);
}

float sigmoid(float x) {
  if (x >= 0.0f)
    return 1.0f / (1.0f + std::exp(-x));
  const auto e = std::exp(x);
  return e / (1.0f + e);
}

struct PixelReader {
  const uint8_t* base;
  size_t pitch;
  size_t bytes_per_pixel;
  bool bgr;

  void read(size_t x, size_t y, float rgb[3]) const {
    const auto* p = base + y * pitch + x * bytes_per_pixel;
    float c[3];
    if (bytes_per_pixel == 8) {
      const auto* h = reinterpret_cast<const uint16_t*>(p);
      c[0] = half_to_float(h[0]); c[1] = half_to_float(h[1]); c[2] = half_to_float(h[2]);
    }
    else if (bytes_per_pixel == 16) {
      const auto* f = reinterpret_cast<const float*>(p);
      c[0] = f[0]; c[1] = f[1]; c[2] = f[2];
    }
    else {
      c[0] = p[0] / 255.0f; c[1] = p[1] / 255.0f; c[2] = p[2] / 255.0f;
    }
    if (bgr) std::swap(c[0], c[2]);
    rgb[0] = c[0]; rgb[1] = c[1]; rgb[2] = c[2];
  }
};

void write_channel(uint8_t* pixel, size_t bytes_per_pixel, int index, float value) {
  if (bytes_per_pixel == 8)
    reinterpret_cast<uint16_t*>(pixel)[index] = float_to_half(value);
  else if (bytes_per_pixel == 16)
    reinterpret_cast<float*>(pixel)[index] = value;
  else
    pixel[index] = static_cast<uint8_t>(std::clamp(value, 0.0f, 1.0f) * 255.0f + 0.5f);
}

float sample_bilinear(const std::vector<float>& map, size_t w, size_t h,
    float fx, float fy) {
  if (map.empty() || w == 0 || h == 0)
    return 0.0f;
  fx = std::clamp(fx, 0.0f, static_cast<float>(w) - 1.0f);
  fy = std::clamp(fy, 0.0f, static_cast<float>(h) - 1.0f);
  const auto x0 = static_cast<size_t>(fx), y0 = static_cast<size_t>(fy);
  const auto x1 = std::min(x0 + 1, w - 1), y1 = std::min(y0 + 1, h - 1);
  const auto tx = fx - static_cast<float>(x0), ty = fy - static_cast<float>(y0);
  const auto a = map[y0 * w + x0] * (1 - tx) + map[y0 * w + x1] * tx;
  const auto b = map[y1 * w + x0] * (1 - tx) + map[y1 * w + x1] * tx;
  return a * (1 - ty) + b * ty;
}

template<typename Fn>
void parallel_rows(size_t rows, Fn&& fn) noexcept try {
  const auto cores = static_cast<size_t>(std::thread::hardware_concurrency());
  const auto workers = std::min<size_t>(cores ? cores : 1, 8);
  if (workers <= 1 || rows < 64) {
    fn(size_t{ }, rows);
    return;
  }
  const auto chunk = (rows + workers - 1) / workers;
  auto pending = std::vector<std::future<void>>();
  pending.reserve(workers - 1);
  for (auto i = size_t{ 1 }; i < workers; ++i) {
    const auto begin = i * chunk;
    if (begin >= rows)
      break;
    pending.push_back(std::async(std::launch::async,
      [&fn, begin, end = std::min(rows, begin + chunk)]() { fn(begin, end); }));
  }
  fn(size_t{ }, std::min(rows, chunk));
  for (auto& f : pending)
    f.get();
}
catch (...) {
  fn(size_t{ }, rows);
}

std::string engine_cache_directory() {
  auto base = std::string();
  if (const auto* local = std::getenv("LOCALAPPDATA"))
    base = local;
  else if (const auto* profile = std::getenv("USERPROFILE"))
    base = std::string(profile) + R"(\AppData\Local)";
  else
    return ".";
  const auto dir = base + R"(\ExtOnnxVision\trt-engines)";
  auto error = std::error_code{ };
  std::filesystem::create_directories(dir, error);
  return (error ? std::string(".") : dir);
}

//-------------------------------------------------------------------------
// CUDA, through the driver API.
//
// The obvious choice is the CUDA runtime (cudaMalloc, cudaMemcpyAsync), but
// that means shipping cudart64_12.dll and building against the CUDA toolkit.
// The driver API lives in nvcuda.dll, which is installed with the NVIDIA
// display driver and is therefore always present on a machine that can run
// any of this. TensorRT uses it too.
//
// So the declarations below are written out rather than included from cuda.h:
// it keeps the package to a single file and removes the toolkit from the
// build requirements. A CUstream and a cudaStream_t are the same pointer, so
// a stream made here is what enqueueV3 expects.
using CUdevice = int;
using CUcontext = struct CUctx_st*;
using CUstream = struct CUstream_st*;
using CUdeviceptr = unsigned long long;

// From cuda.h; the two this needs to name the engine cache by architecture.
constexpr int kComputeCapabilityMajor = 75;
constexpr int kComputeCapabilityMinor = 76;

struct CudaDriver {
  using InitFn          = int(__stdcall*)(unsigned int);
  using DeviceGetFn     = int(__stdcall*)(CUdevice*, int);
  using PrimaryCtxFn    = int(__stdcall*)(CUcontext*, CUdevice);
  using CtxSetFn        = int(__stdcall*)(CUcontext);
  using MemAllocFn      = int(__stdcall*)(CUdeviceptr*, size_t);
  using MemFreeFn       = int(__stdcall*)(CUdeviceptr);
  using MemcpyHtoDFn    = int(__stdcall*)(CUdeviceptr, const void*, size_t, CUstream);
  using MemcpyDtoHFn    = int(__stdcall*)(void*, CUdeviceptr, size_t, CUstream);
  using StreamCreateFn  = int(__stdcall*)(CUstream*, unsigned int);
  using StreamSyncFn    = int(__stdcall*)(CUstream);
  using StreamDestroyFn = int(__stdcall*)(CUstream);
  using DeviceAttribFn  = int(__stdcall*)(int*, int, CUdevice);

  InitFn          init{ };
  DeviceGetFn     device_get{ };
  PrimaryCtxFn    primary_context{ };
  CtxSetFn        set_context{ };
  MemAllocFn      mem_alloc{ };
  MemFreeFn       mem_free{ };
  MemcpyHtoDFn    memcpy_to_device{ };
  MemcpyDtoHFn    memcpy_to_host{ };
  StreamCreateFn  stream_create{ };
  StreamSyncFn    stream_sync{ };
  StreamDestroyFn stream_destroy{ };
  DeviceAttribFn  device_attribute{ };
  bool ready{ };

  bool ok() const { return ready; }
};

const CudaDriver& cuda() {
  static const auto driver = [] {
    auto d = CudaDriver{ };
    auto* lib = LoadLibraryW(L"nvcuda.dll");
    if (!lib)
      return d;

    const auto bind = [lib](auto& fn, const char* name) {
      fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(
        reinterpret_cast<void*>(GetProcAddress(lib, name)));
      return fn != nullptr;
    };

    // The _v2 names are the current ABI for these; the unsuffixed exports are
    // the original CUDA 1.x signatures and take 32-bit sizes.
    d.ready =
      bind(d.init, "cuInit") &&
      bind(d.device_get, "cuDeviceGet") &&
      bind(d.primary_context, "cuDevicePrimaryCtxRetain") &&
      bind(d.set_context, "cuCtxSetCurrent") &&
      bind(d.mem_alloc, "cuMemAlloc_v2") &&
      bind(d.mem_free, "cuMemFree_v2") &&
      bind(d.memcpy_to_device, "cuMemcpyHtoDAsync_v2") &&
      bind(d.memcpy_to_host, "cuMemcpyDtoHAsync_v2") &&
      bind(d.stream_create, "cuStreamCreate") &&
      bind(d.stream_sync, "cuStreamSynchronize") &&
      bind(d.stream_destroy, "cuStreamDestroy_v2") &&
      bind(d.device_attribute, "cuDeviceGetAttribute");
    return d;
  }();
  return driver;
}

//-------------------------------------------------------------------------
// Binding to whatever TensorRT the machine has.

// TensorRT's public headers wrap three exported C functions, and each takes
// the version the caller was compiled against. The library rejects a version
// it does not recognise, which is how a build pinned to one TensorRT refuses
// to run against another.
//
// These headers are 10.13, the newest NVIDIA has published, while the
// libraries this has run against include 10.15.1 and 10.16.1. Passing the
// compile-time constant would refuse both. Passing the version the loaded library reports
// for itself asks it to accept its own API, which is what lets one build work
// against whichever TensorRT is present.
struct TensorRTLibrary {
  using CreateBuilderFn = void* (*)(void*, int32_t);
  using CreateRuntimeFn = void* (*)(void*, int32_t);
  using CreateParserFn  = void* (*)(void*, void*, int);
  using GetVersionFn    = int32_t (*)();

  CreateBuilderFn create_builder{ };
  CreateRuntimeFn create_runtime{ };
  CreateParserFn  create_parser{ };
  int32_t version{ };
  std::string origin;
  std::string problem;

  bool ok() const { return create_builder && create_runtime && create_parser; }

  std::string version_string() const {
    if (version <= 0)
      return "unknown";
    const auto major = (version >= 10000 ? version / 10000 : version / 1000);
    const auto minor = (version >= 10000 ? (version / 100) % 100 : (version / 100) % 10);
    return std::to_string(major) + "." + std::to_string(minor) +
           "." + std::to_string(version % 100);
  }
};

// Looks beside the extension first, then lets Windows search PATH, then tries
// the folder named by TENSORRT_DIR.
HMODULE load_beside_or_from_path(const wchar_t* name) {
  auto self = HMODULE{ };
  if (GetModuleHandleExW(
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
        GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        reinterpret_cast<LPCWSTR>(&load_beside_or_from_path), &self)) {
    wchar_t path[MAX_PATH]{ };
    if (const auto length = GetModuleFileNameW(self, path, MAX_PATH);
        length > 0 && length < MAX_PATH) {
      auto dir = std::wstring(path, length);
      if (const auto slash = dir.find_last_of(LR"(\/)"); slash != std::wstring::npos) {
        dir.resize(slash + 1);
        if (auto* beside = LoadLibraryExW((dir + name).c_str(), nullptr,
                                          LOAD_WITH_ALTERED_SEARCH_PATH))
          return beside;
      }
    }
  }

  // Not beside us: fall back to the machine's, via the normal search order.
  if (auto* found = LoadLibraryW(name))
    return found;

  // Last resort: TENSORRT_DIR without it being on PATH.
  if (const auto* dir = _wgetenv(L"TENSORRT_DIR")) {
    auto folder = std::wstring(dir);
    if (!folder.empty() && folder.back() != L'\\' && folder.back() != L'/')
      folder += L'\\';
    return LoadLibraryExW((folder + name).c_str(), nullptr,
                          LOAD_WITH_ALTERED_SEARCH_PATH);
  }
  return nullptr;
}

const TensorRTLibrary& tensorrt() {
  static const auto library = [] {
    auto result = TensorRTLibrary{ };

    auto* nvinfer = load_beside_or_from_path(L"nvinfer_10.dll");
    if (!nvinfer) {
      result.problem =
        "nvinfer_10.dll not found beside the extension, on PATH, or in "
        "TENSORRT_DIR. Install TensorRT 10 and add its folder to the system "
        "PATH, then restart PIXERA.";
      return result;
    }

    wchar_t origin[MAX_PATH]{ };
    if (GetModuleFileNameW(nvinfer, origin, MAX_PATH)) {
      const auto wide = std::wstring(origin);
      result.origin = std::string(wide.begin(), wide.end());
    }

    const auto get_version = reinterpret_cast<TensorRTLibrary::GetVersionFn>(
      reinterpret_cast<void*>(GetProcAddress(nvinfer, "getInferLibVersion")));
    result.version = (get_version ? get_version() : 0);

    result.create_builder = reinterpret_cast<TensorRTLibrary::CreateBuilderFn>(
      reinterpret_cast<void*>(GetProcAddress(nvinfer, "createInferBuilder_INTERNAL")));
    result.create_runtime = reinterpret_cast<TensorRTLibrary::CreateRuntimeFn>(
      reinterpret_cast<void*>(GetProcAddress(nvinfer, "createInferRuntime_INTERNAL")));

    auto* parser_lib = load_beside_or_from_path(L"nvonnxparser_10.dll");
    if (!parser_lib) {
      result.problem = "nvonnxparser_10.dll not found next to nvinfer_10.dll";
      return result;
    }
    result.create_parser = reinterpret_cast<TensorRTLibrary::CreateParserFn>(
      reinterpret_cast<void*>(GetProcAddress(parser_lib, "createNvOnnxParser_INTERNAL")));

    if (!result.ok())
      result.problem = "TensorRT is present but does not export the expected "
                       "entry points; it may be too old for this build";
    return result;
  }();
  return library;
}

// TensorRT keeps a single logger for the whole process: the first one it is
// given is used from then on, and any later one is ignored with a warning. A
// logger owned by a processor would therefore live on inside TensorRT after
// that processor had been replaced and destroyed, and the next warning would be
// written through freed memory.
//
// So there is exactly one, for the life of the process. Whichever processor is
// currently building attaches its sink, and detaches on destruction. The sink is
// called with the lock held, so a detach waits for any message in flight
// rather than racing it.
class TrtLogger final : public nvinfer1::ILogger {
public:
  static TrtLogger& instance() noexcept {
    static TrtLogger logger;
    return logger;
  }

  void attach(const void* owner, std::function<void(const std::string&)> sink) {
    const auto lock = std::lock_guard(m_mutex);
    m_owner = owner;
    m_sink = std::move(sink);
  }

  void detach(const void* owner) noexcept {
    const auto lock = std::lock_guard(m_mutex);
    if (m_owner == owner) {
      m_owner = nullptr;
      m_sink = nullptr;
    }
  }

  void log(Severity severity, const char* message) noexcept override try {
    // Info and verbose from the builder are voluminous and mostly tactic
    // selection; warnings and errors are what an operator needs.
    if (severity > Severity::kWARNING || !message)
      return;
    const auto text = std::string("TensorRT: ") + message;
    const auto lock = std::lock_guard(m_mutex);
    if (m_sink)
      m_sink(text);
    else
      trace(text);
  }
  catch (...) {
  }

private:
  TrtLogger() = default;

  std::mutex m_mutex;
  const void* m_owner{ };
  std::function<void(const std::string&)> m_sink;
};

template<typename T>
struct TrtDeleter {
  void operator()(T* p) const noexcept { delete p; }
};

} // namespace

//-------------------------------------------------------------------------

struct TensorRTProcessor::Impl {
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IExecutionContext> context;

  std::string input_name, output_name;
  CUdeviceptr device_input{ };
  CUdeviceptr device_output{ };
  size_t input_bytes{ }, output_bytes{ };
  CUstream stream{ };

  ~Impl() {
    const auto& cu = cuda();
    if (cu.ok()) {
      if (device_input) cu.mem_free(device_input);
      if (device_output) cu.mem_free(device_output);
      if (stream) cu.stream_destroy(stream);
    }
    // Order matters: contexts before engines before runtime.
    context.reset();
    engine.reset();
    runtime.reset();
  }
};

TensorRTProcessor::TensorRTProcessor(Config config, Logger log_info, Logger log_error)
    : m_config(std::move(config)),
      m_log_info(std::move(log_info)),
      m_log_error(std::move(log_error)),
      m_impl(std::make_unique<Impl>()) {
}

TensorRTProcessor::~TensorRTProcessor() {
  if (m_loader.joinable())
    m_loader.join();
  // After the join, so no build on the loader thread can still be logging.
  TrtLogger::instance().detach(this);
}

std::string TensorRTProcessor::library_version() const {
  const auto& trt = tensorrt();
  return (trt.version > 0 ? trt.version_string() : std::string());
}

bool TensorRTProcessor::is_loading() const noexcept {
  return m_loading_started.load() && !m_session_ready.load() && !m_session_failed.load();
}

std::string TensorRTProcessor::status() const {
  if (m_session_failed.load()) {
    const auto lock = std::lock_guard(m_status_mutex);
    return "Failed: " + m_failure;
  }
  if (is_loading()) {
    const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(
      std::chrono::steady_clock::now() - m_load_started_at).count();
    const auto elapsed = (seconds < 60
      ? std::to_string(seconds) + "s"
      : std::to_string(seconds / 60) + "m " + std::to_string(seconds % 60) + "s");
    return (m_engine_was_cached.load()
      ? "Loading engine, " + elapsed + " (video is passing through)"
      : "Building TensorRT engine, " + elapsed + " (video is passing through)");
  }
  if (!m_session_ready.load())
    return "Ready";

  const auto ms = static_cast<int>(m_reported_ms.load() + 0.5);
  return "Ready on TensorRT " + library_version() + ", " + std::to_string(ms) + " ms/frame";
}

std::string TensorRTProcessor::timing_summary() const {
  const auto ms = [](double v) { return std::to_string(static_cast<int>(v + 0.5)); };
  return "prepare " + ms(m_last_preprocess_ms) +
         " ms, model " + ms(m_last_inference_ms) +
         " ms, output " + ms(m_last_postprocess_ms) + " ms";
}

FrameMeasurements TensorRTProcessor::measurements() const noexcept {
  const auto lock = std::lock_guard(m_measure_mutex);
  return m_measurements;
}

void TensorRTProcessor::start_loading() noexcept try {
  if (m_loading_started.exchange(true))
    return;
  m_load_started_at = std::chrono::steady_clock::now();
  m_loader = std::thread([this]() noexcept {
    if (build_engine())
      m_session_ready.store(true);
    else
      m_session_failed.store(true);
  });
}
catch (const std::exception& ex) {
  m_session_failed.store(true);
  if (m_log_error)
    m_log_error(std::string("could not start engine build: ") + ex.what());
}

bool TensorRTProcessor::configure(const FrameView&) noexcept {
  start_loading();
  return true;
}

namespace {

// Engines are specific to a model, a TensorRT version and a GPU architecture,
// so all three go in the name. Without the architecture a cache copied
// between machines would be silently wrong; without the version an upgraded
// TensorRT would refuse a cached engine with an unhelpful error.
std::string engine_cache_file(const std::string& model_path, int size_override,
    int effort, int32_t trt_version) {
  auto sm = std::string("unknown");
  if (const auto& cu = cuda(); cu.ok()) {
    auto device = CUdevice{ };
    auto major = 0, minor = 0;
    if (cu.device_get(&device, 0) == 0 &&
        cu.device_attribute(&major, kComputeCapabilityMajor, device) == 0 &&
        cu.device_attribute(&minor, kComputeCapabilityMinor, device) == 0)
      sm = "sm" + std::to_string(major) + std::to_string(minor);
  }

  auto stem = std::filesystem::path(model_path).stem().string();
  auto hash = std::hash<std::string>{ }(model_path);

  return engine_cache_directory() + R"(\)" + stem + "_" +
         std::to_string(hash) + "_" + std::to_string(size_override) +
         "_e" + std::to_string(effort) +
         "_trt" + std::to_string(trt_version) + "_" + sm + ".trtengine";
}

} // namespace

bool TensorRTProcessor::build_engine() noexcept try {
  // TensorRT needs a current context on this thread. Retaining the device's
  // primary context is the driver-API equivalent of what the CUDA runtime
  // does implicitly on first use, and shares the context with anything else
  // in the process using CUDA.
  if (const auto& cu = cuda(); cu.ok()) {
    auto device = CUdevice{ };
    auto context = CUcontext{ };
    if (cu.init(0) == 0 && cu.device_get(&device, 0) == 0 &&
        cu.primary_context(&context, device) == 0)
      cu.set_context(context);
  }

  const auto& trt = tensorrt();
  if (!trt.ok()) {
    const auto lock = std::lock_guard(m_status_mutex);
    m_failure = trt.problem;
    if (m_log_error)
      m_log_error(trt.problem);
    return false;
  }

  if (m_log_info)
    m_log_info("TensorRT " + trt.version_string() + " from " + trt.origin +
               " (bound to the library's own API version, so this build is "
               "not tied to one TensorRT release)");

  const auto cache_file = engine_cache_file(m_config.model_path,
    m_config.size_override, m_config.trt_build_effort, trt.version);
  m_engine_was_cached.store(std::filesystem::exists(cache_file));

  // Named in full because deleting this file is how a rebuild is asked for.
  // There is deliberately no parameter for that: a rebuild takes minutes and
  // throws away a good engine, which is not something the timeline should be
  // able to animate or a stray drag should be able to trigger.
  if (m_log_info)
    m_log_info("engine file for this model, GPU and settings: " + cache_file);

  return load_or_compile(cache_file);
}
catch (const std::exception& ex) {
  const auto lock = std::lock_guard(m_status_mutex);
  m_failure = ex.what();
  if (m_log_error)
    m_log_error(std::string("engine build failed: ") + ex.what());
  return false;
}

bool TensorRTProcessor::load_or_compile(const std::string& cache_file) noexcept try {
  const auto& trt = tensorrt();
  auto& logger = TrtLogger::instance();
  logger.attach(this, m_log_info);

  m_impl->runtime.reset(static_cast<nvinfer1::IRuntime*>(
    trt.create_runtime(&logger, trt.version)));
  if (!m_impl->runtime)
    throw std::runtime_error("could not create the TensorRT runtime");

  auto serialized = std::vector<char>();

  if (std::filesystem::exists(cache_file)) {
    auto in = std::ifstream(cache_file, std::ios::binary | std::ios::ate);
    if (in) {
      serialized.resize(static_cast<size_t>(in.tellg()));
      in.seekg(0);
      in.read(serialized.data(), static_cast<std::streamsize>(serialized.size()));
      if (m_log_info)
        m_log_info("reusing the engine compiled earlier for this model and GPU");
    }
  }

  if (serialized.empty()) {
    if (m_log_info)
      m_log_info("compiling a TensorRT engine for this model and GPU; "
                 "this takes minutes and only happens once");

    auto builder = std::unique_ptr<nvinfer1::IBuilder>(
      static_cast<nvinfer1::IBuilder*>(
        trt.create_builder(&logger, trt.version)));
    if (!builder)
      throw std::runtime_error("could not create the TensorRT builder");

    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
      builder->createNetworkV2(0));
    if (!network)
      throw std::runtime_error("could not create a network definition");

    auto parser = std::unique_ptr<nvonnxparser::IParser>(
      static_cast<nvonnxparser::IParser*>(
        trt.create_parser(network.get(), &logger,
                          NV_ONNX_PARSER_VERSION)));
    if (!parser)
      throw std::runtime_error("could not create the ONNX parser");

    if (!parser->parseFromFile(m_config.model_path.c_str(),
                               static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
      auto detail = std::string();
      for (auto i = 0; i < parser->getNbErrors(); ++i)
        detail += (detail.empty() ? "" : "; ") + std::string(parser->getError(i)->desc());
      throw std::runtime_error(
        "TensorRT could not parse this model" +
        (detail.empty() ? std::string() : ": " + detail) +
        ". There is no fallback for unsupported operators, so every "
        "operator in the model must be supported by TensorRT.");
    }

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
      throw std::runtime_error("could not create a builder config");

    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    config->setBuilderOptimizationLevel(std::clamp(m_config.trt_build_effort, 0, 5));

    // Dynamic input dimensions have to be pinned to something before an
    // engine can exist. The Input Size Override parameter sets them, and the
    // model's own shape is used when it is already fixed.
    auto* input = network->getInput(0);
    if (!input)
      throw std::runtime_error("the model has no input tensor");

    auto dims = input->getDimensions();
    const auto dynamic = std::any_of(dims.d, dims.d + dims.nbDims,
      [](int64_t d) { return d < 0; });

    if (dynamic) {
      const auto wanted = (m_config.size_override > 0 ? m_config.size_override : 512);
      auto* profile = builder->createOptimizationProfile();
      auto fixed = dims;
      for (auto i = 0; i < fixed.nbDims; ++i)
        if (fixed.d[i] < 0)
          fixed.d[i] = (i == 0 ? 1 : wanted);

      profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, fixed);
      profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, fixed);
      profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, fixed);
      config->addOptimizationProfile(profile);

      if (m_log_info)
        m_log_info("this model has dynamic input dimensions; pinning them to " +
                   std::to_string(wanted) + ". Set Input Size Override if the "
                   "model expects a different size.");
    }

    auto built = std::unique_ptr<nvinfer1::IHostMemory>(
      builder->buildSerializedNetwork(*network, *config));
    if (!built)
      throw std::runtime_error("TensorRT could not build an engine for this model");

    serialized.assign(static_cast<const char*>(built->data()),
                      static_cast<const char*>(built->data()) + built->size());

    if (auto out = std::ofstream(cache_file, std::ios::binary))
      out.write(serialized.data(), static_cast<std::streamsize>(serialized.size()));
  }

  m_impl->engine.reset(m_impl->runtime->deserializeCudaEngine(
    serialized.data(), serialized.size()));
  if (!m_impl->engine)
    throw std::runtime_error("could not deserialize the engine");

  m_impl->context.reset(m_impl->engine->createExecutionContext());
  if (!m_impl->context)
    throw std::runtime_error("could not create an execution context");

  // Work out which tensor is which, and how big.
  for (auto i = 0; i < m_impl->engine->getNbIOTensors(); ++i) {
    const auto* name = m_impl->engine->getIOTensorName(i);
    const auto mode = m_impl->engine->getTensorIOMode(name);
    if (mode == nvinfer1::TensorIOMode::kINPUT && m_impl->input_name.empty())
      m_impl->input_name = name;
    else if (mode == nvinfer1::TensorIOMode::kOUTPUT && m_impl->output_name.empty())
      m_impl->output_name = name;
  }
  if (m_impl->input_name.empty() || m_impl->output_name.empty())
    throw std::runtime_error("the engine does not have one input and one output");

  const auto in_dims = m_impl->context->getTensorShape(m_impl->input_name.c_str());
  const auto out_dims = m_impl->context->getTensorShape(m_impl->output_name.c_str());

  const auto elements = [](const nvinfer1::Dims& d) {
    auto n = size_t{ 1 };
    for (auto i = 0; i < d.nbDims; ++i)
      n *= static_cast<size_t>(std::max<int64_t>(d.d[i], 1));
    return n;
  };

  if (in_dims.nbDims == 4) {
    // Same channels-last test as the ONNX path: a trailing 3 with a second
    // dimension too large to be a channel count.
    m_input_is_nhwc = (in_dims.d[3] == 3 && in_dims.d[1] > 4);
    m_in_height = static_cast<size_t>(m_input_is_nhwc ? in_dims.d[1] : in_dims.d[2]);
    m_in_width  = static_cast<size_t>(m_input_is_nhwc ? in_dims.d[2] : in_dims.d[3]);
  }
  else {
    throw std::runtime_error("only 4-dimensional image inputs are supported");
  }

  m_output_kind = OutputKind::Unsupported;
  if (out_dims.nbDims == 4) {
    const auto d1 = static_cast<size_t>(out_dims.d[1]);
    const auto d3 = static_cast<size_t>(out_dims.d[3]);
    m_output_is_nhwc = ((d3 == 1 || d3 == 3) && d1 > 4);
    m_out_channels = (m_output_is_nhwc ? d3 : d1);
    m_out_height = static_cast<size_t>(m_output_is_nhwc ? out_dims.d[1] : out_dims.d[2]);
    m_out_width  = static_cast<size_t>(m_output_is_nhwc ? out_dims.d[2] : out_dims.d[3]);
    m_output_kind = (m_out_channels == 1 ? OutputKind::SingleChannelMap
                   : m_out_channels == 3 ? OutputKind::ColourImage
                                         : OutputKind::Unsupported);
  }
  else if (out_dims.nbDims == 3) {
    m_out_height = static_cast<size_t>(out_dims.d[1]);
    m_out_width  = static_cast<size_t>(out_dims.d[2]);
    m_out_channels = 1;
    m_output_kind = OutputKind::SingleChannelMap;
  }

  m_input_tensor.resize(elements(in_dims));
  m_output_tensor.resize(elements(out_dims));
  if (m_output_kind == OutputKind::SingleChannelMap)
    m_map.resize(m_out_width * m_out_height);

  m_impl->input_bytes = m_input_tensor.size() * sizeof(float);
  m_impl->output_bytes = m_output_tensor.size() * sizeof(float);

  const auto& cu = cuda();
  if (!cu.ok())
    throw std::runtime_error(
      "nvcuda.dll could not be loaded or is missing entry points; the NVIDIA "
      "display driver provides it, so this usually means no NVIDIA GPU");

  if (cu.mem_alloc(&m_impl->device_input, m_impl->input_bytes) != 0 ||
      cu.mem_alloc(&m_impl->device_output, m_impl->output_bytes) != 0)
    throw std::runtime_error("could not allocate GPU buffers; check available VRAM");

  if (cu.stream_create(&m_impl->stream, 0) != 0)
    throw std::runtime_error("could not create a CUDA stream");

  m_impl->context->setTensorAddress(m_impl->input_name.c_str(),
    reinterpret_cast<void*>(m_impl->device_input));
  m_impl->context->setTensorAddress(m_impl->output_name.c_str(),
    reinterpret_cast<void*>(m_impl->device_output));

  if (m_log_info) {
    const auto shape = [](const nvinfer1::Dims& d) {
      auto s = std::string("[");
      for (auto i = 0; i < d.nbDims; ++i)
        s += (i ? "," : "") + std::to_string(d.d[i]);
      return s + "]";
    };
    m_log_info("engine ready, input " + shape(in_dims) +
               (m_input_is_nhwc ? " NHWC" : " NCHW") +
               " -> output " + shape(out_dims));
    if (m_output_kind == OutputKind::Unsupported)
      m_log_info("this output shape is not directly renderable; "
                 "frames will pass through");
  }
  return true;
}
catch (const std::exception& ex) {
  const auto lock = std::lock_guard(m_status_mutex);
  m_failure = ex.what();
  if (m_log_error)
    m_log_error(std::string("engine build failed: ") + ex.what());
  return false;
}

void TensorRTProcessor::preprocess(const FrameView& input) noexcept {
  const auto reader = PixelReader{ input.data, input.pitch,
                                   input.bytes_per_pixel, input.bgr_order };
  const auto plane = m_in_width * m_in_height;
  const auto sx = static_cast<float>(input.width) / static_cast<float>(m_in_width);
  const auto sy = static_cast<float>(input.height) / static_cast<float>(m_in_height);

  parallel_rows(m_in_height, [&](size_t y_begin, size_t y_end) {
  for (auto y = y_begin; y < y_end; ++y) {
    const auto src_y = std::min(static_cast<size_t>(y * sy),
                                input.height ? input.height - 1 : 0);
    for (auto x = size_t{ }; x < m_in_width; ++x) {
      const auto src_x = std::min(static_cast<size_t>(x * sx),
                                  input.width ? input.width - 1 : 0);
      float rgb[3];
      reader.read(src_x, src_y, rgb);

      for (auto c = 0; c < 3; ++c) {
        auto v = rgb[c];
        switch (m_config.normalisation) {
          case Normalisation::ZeroToOne: break;
          case Normalisation::ImageNet:
            v = (v - kImageNetMean[c]) / kImageNetStd[c];
            break;
          case Normalisation::MinusOneToOne:
            v = v * 2.0f - 1.0f;
            break;
          case Normalisation::ZeroTo255:
            v = v * 255.0f;
            break;
        }
        const auto i = y * m_in_width + x;
        if (m_input_is_nhwc)
          m_input_tensor[i * 3 + c] = v;
        else
          m_input_tensor[c * plane + i] = v;
      }
    }
  }
  });
}

void TensorRTProcessor::decode_map(const float* data, size_t count) noexcept {
  const auto n = std::min(count, m_map.size());
  if (n == 0) return;

  auto activation = m_config.activation;
  if (activation == Activation::Auto) {
    auto lo = data[0], hi = data[0];
    for (auto i = size_t{ 1 }; i < n; ++i) {
      lo = std::min(lo, data[i]);
      hi = std::max(hi, data[i]);
    }
    if (lo >= 0.0f && hi <= 1.0f)
      activation = Activation::None;
    else if (lo < -0.5f)
      activation = Activation::Sigmoid;
    else
      activation = Activation::MinMax;
    m_resolved_activation = activation;
  }

  if (activation == Activation::MinMax) {
    auto lo = data[0], hi = data[0];
    for (auto i = size_t{ 1 }; i < n; ++i) {
      lo = std::min(lo, data[i]);
      hi = std::max(hi, data[i]);
    }
    const auto range = (hi - lo);
    const auto scale = (range > 1e-6f ? 1.0f / range : 0.0f);
    for (auto i = size_t{ }; i < n; ++i)
      m_map[i] = (data[i] - lo) * scale;
  }
  else if (activation == Activation::Sigmoid) {
    for (auto i = size_t{ }; i < n; ++i)
      m_map[i] = sigmoid(data[i]);
  }
  else {
    for (auto i = size_t{ }; i < n; ++i)
      m_map[i] = std::clamp(data[i], 0.0f, 1.0f);
  }
}

void TensorRTProcessor::measure_map() noexcept {
  if (m_map.empty() || m_out_width == 0 || m_out_height == 0)
    return;

  auto sum = 0.0, wx = 0.0, wy = 0.0;
  auto above_half = size_t{ };
  for (auto y = size_t{ }; y < m_out_height; ++y) {
    const auto* row = m_map.data() + y * m_out_width;
    for (auto x = size_t{ }; x < m_out_width; ++x) {
      const auto v = static_cast<double>(row[x]);
      sum += v; wx += v * x; wy += v * y;
      if (v > 0.5f) ++above_half;
    }
  }

  const auto pixels = static_cast<double>(m_map.size());
  auto measured = FrameMeasurements{ };
  measured.valid = true;
  measured.coverage = static_cast<double>(above_half) / pixels;
  measured.mean = sum / pixels;
  measured.centre_x = (sum > 0.0 ? wx / sum / static_cast<double>(m_out_width - 1) : 0.5);
  measured.centre_y = (sum > 0.0 ? wy / sum / static_cast<double>(m_out_height - 1) : 0.5);

  const auto lock = std::lock_guard(m_measure_mutex);
  m_measurements = measured;
}

void TensorRTProcessor::write_map(const FrameView& input, FrameBuffer& output) noexcept {
  const auto rows = std::min(input.height, output.height);
  const auto cols = std::min(input.width, output.width);
  const auto sx = static_cast<float>(m_out_width) / static_cast<float>(output.width);
  const auto sy = static_cast<float>(m_out_height) / static_cast<float>(output.height);
  const auto to_alpha = (m_config.target != OutputTarget::Greyscale);
  const auto to_grey = (m_config.target != OutputTarget::Alpha);

  parallel_rows(rows, [&](size_t y_begin, size_t y_end) {
  for (auto y = y_begin; y < y_end; ++y) {
    const auto* src_row = input.data + y * input.pitch;
    auto* dst_row = output.data + y * output.pitch;
    const auto fy = static_cast<float>(y) * sy;
    for (auto x = size_t{ }; x < cols; ++x) {
      const auto* src = src_row + x * input.bytes_per_pixel;
      auto* dst = dst_row + x * output.bytes_per_pixel;
      const auto v = sample_bilinear(m_map, m_out_width, m_out_height,
                                     static_cast<float>(x) * sx, fy);
      const auto bpp = output.bytes_per_pixel;
      if (to_grey) {
        for (auto c = 0; c < 3; ++c)
          write_channel(dst, bpp, c, v);
      }
      else if (dst != reinterpret_cast<uint8_t*>(const_cast<uint8_t*>(src))) {
        std::memcpy(dst, src, std::min(bpp, input.bytes_per_pixel));
      }
      write_channel(dst, bpp, 3, to_alpha ? v : 1.0f);
    }
  }
  });
}

void TensorRTProcessor::resolve_colour_scale(const float* data, size_t count) noexcept {
  if (m_colour_scale_resolved || count == 0)
    return;
  m_colour_scale_resolved = true;

  auto lo = data[0], hi = data[0];
  for (auto i = size_t{ 1 }; i < count; ++i) {
    lo = std::min(lo, data[i]);
    hi = std::max(hi, data[i]);
  }
  if (hi > 1.5f) {
    m_colour_scale = 1.0f / 255.0f;
  }
  else if (lo < -0.5f) {
    m_colour_scale = 0.5f;
    m_colour_bias = 0.5f;
  }
}

void TensorRTProcessor::write_colour(const float* data, const FrameView& input,
    FrameBuffer& output) noexcept {
  const auto plane = m_out_width * m_out_height;
  const auto sx = static_cast<float>(m_out_width) / static_cast<float>(output.width);
  const auto sy = static_cast<float>(m_out_height) / static_cast<float>(output.height);
  const auto rows = std::min(input.height, output.height);
  const auto cols = std::min(input.width, output.width);

  parallel_rows(rows, [&](size_t y_begin, size_t y_end) {
  for (auto y = y_begin; y < y_end; ++y) {
    auto* dst_row = output.data + y * output.pitch;
    const auto src_y = std::min(static_cast<size_t>(y * sy),
                                m_out_height ? m_out_height - 1 : 0);
    for (auto x = size_t{ }; x < cols; ++x) {
      auto* dst = dst_row + x * output.bytes_per_pixel;
      const auto src_x = std::min(static_cast<size_t>(x * sx),
                                  m_out_width ? m_out_width - 1 : 0);
      const auto i = src_y * m_out_width + src_x;
      for (auto c = size_t{ }; c < 3; ++c) {
        const auto v = (m_output_is_nhwc ? data[i * 3 + c] : data[c * plane + i]);
        write_channel(dst, output.bytes_per_pixel, static_cast<int>(c),
          std::clamp(v * m_colour_scale + m_colour_bias, 0.0f, 1.0f));
      }
      write_channel(dst, output.bytes_per_pixel, 3, 1.0f);
    }
  }
  });
}

bool TensorRTProcessor::process(const FrameView& input, FrameBuffer& output) noexcept try {
  if (!m_session_ready.load() || m_output_kind == OutputKind::Unsupported)
    return false;

  const auto t_pre = std::chrono::steady_clock::now();
  preprocess(input);
  m_last_preprocess_ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - t_pre).count();

  const auto t0 = std::chrono::steady_clock::now();

  const auto& cu = cuda();

  if (cu.memcpy_to_device(m_impl->device_input, m_input_tensor.data(),
                          m_impl->input_bytes, m_impl->stream) != 0)
    return false;

  // A CUstream and a cudaStream_t are the same pointer type.
  if (!m_impl->context->enqueueV3(reinterpret_cast<cudaStream_t>(m_impl->stream)))
    return false;

  if (cu.memcpy_to_host(m_output_tensor.data(), m_impl->device_output,
                        m_impl->output_bytes, m_impl->stream) != 0)
    return false;

  if (cu.stream_sync(m_impl->stream) != 0)
    return false;

  m_last_inference_ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - t0).count();

  const auto previous = m_reported_ms.load();
  m_reported_ms.store(previous > 0.0
    ? previous * 0.95 + m_last_inference_ms * 0.05
    : m_last_inference_ms);

  const auto t_post = std::chrono::steady_clock::now();
  if (m_output_kind == OutputKind::SingleChannelMap) {
    decode_map(m_output_tensor.data(), m_output_tensor.size());
    measure_map();
    write_map(input, output);
  }
  else {
    resolve_colour_scale(m_output_tensor.data(), m_output_tensor.size());
    write_colour(m_output_tensor.data(), input, output);
  }
  m_last_postprocess_ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - t_post).count();
  return true;
}
catch (const std::exception& ex) {
  if (m_log_error)
    m_log_error(std::string("inference failed: ") + ex.what());
  return false;
}

} // namespace
