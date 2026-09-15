
#include "VisionStream.h"
#include "PassthroughProcessor.h"
#include "Trace.h"
#include <array>
#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <format>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

namespace rxext::onnxvision {
namespace {

size_t get_bytes_per_pixel(Format format) {
  switch (format) {
    case Format::None: return 0;
    case Format::R8_UNORM: return 1;
    case Format::R8G8_UNORM: return 2;
    case Format::R8G8B8A8_UNORM: return 4;
    case Format::B8G8R8A8_UNORM: return 4;
    case Format::R16G16B16A16_SFLOAT: return 8;
    case Format::R32G32B32A32_SFLOAT: return 16;
  }
  return 0;
}

// Colour shown when the extension is loaded but idle. Deliberately
// distinctive, so "loaded, nothing connected" reads differently from a black
// layer, which usually means a failure.
constexpr uint32_t kIdleColor = 0xFF1E3A5F; // BGRA: deep blue

void copy_rows(const FrameView& input, FrameBuffer& output) {
  const auto rows = std::min(input.height, output.height);
  const auto row_bytes = std::min(input.width, output.width) * input.bytes_per_pixel;
  for (auto y = size_t{ }; y < rows; ++y)
    std::memcpy(output.data + y * output.pitch,
                input.data + y * input.pitch,
                row_bytes);
}

// PIXERA renders into our texture with red and blue in the opposite order to
// the one it uses when sampling what we publish, so an untouched frame comes
// back with skin tones blue. Correcting it needs CPU access to the pixels,
// which is why Passthrough mode cannot fix it and Readback can.
void swap_red_blue(FrameBuffer& frame) {
  const auto swap_elements = [&](auto* row, size_t count) {
    for (auto x = size_t{ }; x < count; ++x)
      std::swap(row[x * 4 + 0], row[x * 4 + 2]);
  };

  for (auto y = size_t{ }; y < frame.height; ++y) {
    auto* row = frame.data + y * frame.pitch;
    switch (frame.bytes_per_pixel) {
      case 4:  // 8 bit RGBA / BGRA
        swap_elements(row, frame.width);
        break;
      case 8:  // 16 bit float RGBA
        swap_elements(reinterpret_cast<uint16_t*>(row), frame.width);
        break;
      case 16: // 32 bit float RGBA
        swap_elements(reinterpret_cast<uint32_t*>(row), frame.width);
        break;
      default:
        return; // single or dual channel formats have nothing to swap
    }
  }
}

// Tells a real path from what PIXERA leaves behind in a string parameter once
// a project has been saved and reloaded: a bare number, or nothing.
bool looks_like_a_path(std::string_view text) {
  return (!text.empty() &&
          text.find_first_not_of("0123456789.-+eE") != std::string_view::npos);
}

// A model path supplied as a stream setting, which is how a script configures
// a run without a UI.
//
// Deliberately not the SDK's reserved "filename" setting: PIXERA populates
// that with the path the generic extension was loaded from, so reading it as
// the model made the extension try to load its own DLL. A .dll is rejected
// outright for the same reason, since a stale project may still carry one.
std::string setting_model_path(const ValueSet& settings) {
  auto path = std::string(settings.get("model", ""));
  if (path.size() >= 4 &&
      _stricmp(path.c_str() + path.size() - 4, ".dll") == 0)
    return { };
  return path;
}

// Where the model path for one layer is remembered between sessions.
//
// PIXERA cannot carry it: a string parameter is saved as a number, and the
// settings we declare never come back at all, so the path is lost the moment
// a project is reopened. Keeping it ourselves, keyed on the layer identity
// the host does give us, is the only route that survives a save.
//
// Alongside the engine cache, under LOCALAPPDATA: writable without admin
// rights and per user, so a machine shared between operators behaves.
std::string layer_store_path(std::string_view instance_id) {
  if (instance_id.empty())
    return { };

  // The id is a signed integer as it arrives, but it is about to become a
  // file name, so nothing exotic gets through.
  auto key = std::string();
  for (const auto c : instance_id)
    if (std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_')
      key += c;
  if (key.empty())
    return { };

  auto base = std::string();
  if (const auto* local = std::getenv("LOCALAPPDATA"))
    base = local;
  else if (const auto* profile = std::getenv("USERPROFILE"))
    base = std::string(profile) + R"(\AppData\Local)";
  else
    return { };

  const auto dir = base + R"(\ExtOnnxVision\layers)";
  auto error = std::error_code{ };
  std::filesystem::create_directories(dir, error);
  if (error)
    return { };
  return dir + R"(\)" + key + ".txt";
}

std::string read_remembered_model(std::string_view instance_id) {
  const auto file = layer_store_path(instance_id);
  if (file.empty())
    return { };
  auto in = std::ifstream(file);
  if (!in)
    return { };
  auto path = std::string();
  std::getline(in, path);
  // A remembered path is only useful if the file is still there; a model
  // moved or deleted between sessions should read as "nothing set" rather
  // than as a load failure every time the project opens.
  if (path.empty() || !std::filesystem::exists(path))
    return { };
  return path;
}

void remember_model(std::string_view instance_id, const std::string& path) {
  const auto file = layer_store_path(instance_id);
  if (file.empty())
    return;
  if (auto out = std::ofstream(file, std::ios::trunc))
    out << path;
}

} // namespace

VisionStream::VisionStream(std::shared_ptr<d3d11::Device> d3d_device,
    ValueSet settings)
    : m_settings(std::move(settings)),
      m_d3d_device(std::move(d3d_device)) {
}

VisionStream::~VisionStream() = default;

bool VisionStream::initialize() noexcept try {
  host().log_info(std::format("ExtOnnxVision: initializing stream '{}'",
    m_settings.get(SettingNames::handle)));
  trace("initialize: enter");

  // Every setting the host handed this stream, verbatim.
  //
  // Worth the noise once per stream as a diagnostic. It includes the
  // instance_id the model path is remembered against.
  {
    auto listed = std::string();
    for (const auto& value : m_settings.values)
      listed += (listed.empty() ? "" : ", ") +
                std::string(value.name) + "='" + std::string(value.value) + "'";
    trace("stream settings: " + (listed.empty() ? "(none)" : listed));
  }

  // Input slot. PIXERA shows this as a parameter on the layer; dragging a
  // Resource onto it is what routes content into the extension.
  m_source = add_parameter<ParameterTexture>("source");
  m_source->set_property(PropertyNames::name, "Source");

  // Process is the working mode and the sensible default: with no model set it
  // still corrects channel order, so a freshly dropped layer shows correct
  // video rather than the blue image Passthrough gives. The other two are
  // diagnostics. A stream setting can override the default, which is how a
  // script selects a mode without a UI.
  m_mode = add_parameter<ParameterInt>("mode",
    m_settings.get<int>("mode", ModeReadback));
  m_mode->set_property(PropertyNames::name, "Mode");
  m_mode->set_property<std::vector<std::string>>(PropertyNames::enum_names,
    { "Idle", "Passthrough (raw)", "Process" });

  m_swap_rb = add_parameter<ParameterBool>("swap_rb", true);
  m_swap_rb->set_property(PropertyNames::name, "Swap Red/Blue");

  // Model configuration lives on the layer's parameters, which is the only
  // place PIXERA both shows a control and passes its value back. Each takes
  // the matching stream setting as its default, so a script can configure a
  // run without a UI.

  m_normalisation = add_parameter<ParameterInt>("normalisation",
    m_settings.get<int>("normalisation", 1));
  m_normalisation->set_property(PropertyNames::name, "Input Range");
  m_normalisation->set_property<std::vector<std::string>>(PropertyNames::enum_names,
    { "0 to 1", "ImageNet", "-1 to 1", "0 to 255" });

  m_activation = add_parameter<ParameterInt>("activation",
    m_settings.get<int>("activation", 0));
  m_activation->set_property(PropertyNames::name, "Output Range");
  m_activation->set_property<std::vector<std::string>>(PropertyNames::enum_names,
    { "Auto", "Already 0 to 1", "Logits (sigmoid)", "Rescale min/max" });

  m_target = add_parameter<ParameterInt>("target",
    m_settings.get<int>("target", 0));
  m_target->set_property(PropertyNames::name, "Result As");
  m_target->set_property<std::vector<std::string>>(PropertyNames::enum_names,
    { "Alpha (key)", "Greyscale (view)", "Both" });

  m_size = add_parameter<ParameterInt>("size", m_settings.get<int>("size", 0));
  m_size->set_property(PropertyNames::name, "Input Size Override");
  m_size->set_property(PropertyNames::min_value, 0);
  m_size->set_property(PropertyNames::max_value, 2048);

  m_trt_effort = add_parameter<ParameterInt>("trt_effort",
    m_settings.get<int>("trt_effort", 3));
  m_trt_effort->set_property(PropertyNames::name, "TensorRT Build Effort");
  m_trt_effort->set_property(PropertyNames::min_value, 0);
  m_trt_effort->set_property(PropertyNames::max_value, 5);

  // Registered last so it appears after the settings it depends on: setting
  // a path is what starts a load, so everything else should be chosen first.
  //
  // PIXERA's parameter list is built for timeline values, so it gives a string
  // parameter a numeric drag box and writes a number into the project on save.
  // The path therefore has to be typed rather than dragged, and survives a
  // reload only because the extension remembers it; see ensure_processor().
  m_model_path = add_parameter<ParameterString>("model",
    setting_model_path(m_settings));
  m_model_path->set_property(PropertyNames::name, "Model File");

  // Retries a failed load, or picks up a model file replaced on disk under
  // the same name. It does not rebuild the engine; see the readme. Either
  // direction counts, so it works whether the host draws it as a button or a
  // checkbox that stays where it was left.
  m_reload = add_parameter<ParameterBool>("reload", false);
  m_reload->set_property(PropertyNames::name, "Reload Model");



  // Read-only state, so an operator can see what the extension is doing
  // without opening a log file.
  m_status = add_output_parameter<ParameterString>("status");
  m_status->set_property(PropertyNames::name, "Status");

  // A model that finds a subject knows more than the shape of it: how much of
  // the frame it fills, and where its weight sits. Published as numbers so a
  // PIXERA layer can use them for something other than a matte. Doubles rather
  // than strings deliberately — the parameter list is built for numeric
  // timeline values, and these are the ones it handles properly.
  const auto add_measurement = [this](const char* id, const char* label) {
    auto* p = add_output_parameter<ParameterValue>(id);
    p->set_property(PropertyNames::name, label);
    p->set_property(PropertyNames::min_value, 0.0);
    p->set_property(PropertyNames::max_value, 1.0);
    return p;
  };
  m_out_coverage = add_measurement("coverage", "Subject Coverage");
  m_out_centre_x = add_measurement("centre_x", "Subject Centre X");
  m_out_centre_y = add_measurement("centre_y", "Subject Centre Y");
  m_out_mean = add_measurement("mean", "Result Mean");

  // Result, published back to PIXERA.
  m_sampler = add_output_parameter<ParameterTexture>(ParameterNames::sampler);

  m_processor = std::make_unique<PassthroughProcessor>();

  publish_placeholder();

  // Log exactly what was registered, so a missing control in the host can be
  // told apart from a parameter that was never added.
  {
    auto list = std::string();
    const auto count = get_parameter_count();
    for (auto i = size_t{ }; i < count; ++i)
      if (auto* p = get_parameter(i))
        list += (list.empty() ? "" : ", ") + std::string(p->name()) +
                "[" + std::string(p->get_property(PropertyNames::direction)) + "]";
    trace("registered " + std::to_string(count) + " parameters: " + list);
    host().log_info("ExtOnnxVision: registered " + std::to_string(count) +
                    " parameters: " + list);
  }

  trace("initialize: done");
  return true;
}
catch (const std::exception& ex) {
  host().log_error(std::format("ExtOnnxVision: initialization failed: {}", ex.what()));
  return false;
}

void VisionStream::publish_placeholder() noexcept {
  auto desc = TextureDesc{ };
  desc.width = 2;
  desc.height = 2;
  desc.format = Format::B8G8R8A8_UNORM;
  m_placeholder = host().create_texture(desc);
  m_sampler->set_texture(TextureRef(m_placeholder));

  auto colors = std::array<uint32_t, 4>{
    kIdleColor, kIdleColor, kIdleColor, kIdleColor };
  auto buffer = BufferDesc{ };
  buffer.data = colors.data();
  buffer.pitch = 2 * sizeof(uint32_t);
  buffer.size = 2 * buffer.pitch;
  host().upload_texture(TextureRef(m_placeholder), buffer, true,
    []() noexcept { });
}

VisionStream::Mode VisionStream::mode() const noexcept {
  if (!m_mode)
    return ModePassthrough;
  const auto value = m_mode->value();
  if (value < ModeIdle || value > ModeReadback)
    return ModeIdle;
  return static_cast<Mode>(value);
}

bool VisionStream::update_settings(ValueSet settings) noexcept {
  // Accept every change in place; ensure_processor() notices and reloads.
  // Returning false would make the host tear down and recreate the stream.
  //
  // Traced so the log shows whether, and when, the host calls it.
  const auto before = setting_model_path(m_settings);
  m_settings = std::move(settings);
  const auto after = setting_model_path(m_settings);
  trace("update_settings: model " +
        (before == after ? "unchanged (" + after + ")"
                         : "'" + before + "' -> '" + after + "'"));
  return true;
}

ValueSet VisionStream::get_state() noexcept {
  auto state = ValueSet();
  const auto desc = (m_sampler ? m_sampler->texture().desc() : TextureDesc{ });
  state.set(StateNames::resolution_x, desc.width);
  state.set(StateNames::resolution_y, desc.height);
  state.set(StateNames::format, get_format_name(desc.format));
  // The texture we publish is bottom-up relative to what PIXERA expects, so
  // without this the image appears upside down. ExtSampleD3D's Output does
  // the same thing.
  state.set(StateNames::scale_y, -1);
  return state;
}

bool VisionStream::update() noexcept try {
  // Unconditional, so the log distinguishes "the host stopped calling us" from
  // "we are running but producing nothing". Every other trace here is
  // state-dependent and goes quiet in exactly the case worth diagnosing.
  if (++m_ticks % 120 == 1)
    trace("update tick " + std::to_string(m_ticks) +
          ", frames " + std::to_string(m_frames) +
          ", stage " + std::to_string(static_cast<int>(m_stage.load())) +
          ", processor " + (m_processor ? m_processor->name() : "none") +
          ", pending " + (m_pending.valid() ? "yes" : "no"));

  // The host advertises the incoming geometry as properties on the source
  // parameter. Allocate a matching texture for it to render into.
  update_input_texture(*m_source, [&](const TextureDesc& desc) {
    log_geometry(desc);

    // Own the texture PIXERA renders into, so its contents can be read back
    // on the CPU without download_texture(), which does not work here.
    auto shared_desc = desc;
    m_reader.reset();
    if (m_d3d_device) try {
      m_reader = std::make_unique<TextureReader>(*m_d3d_device,
        desc.width, desc.height, desc.format);
      shared_desc.share_handle = {
        HandleType::D3D11_IMAGE, m_reader->share_handle() };
      trace("created shared source texture");
    }
    catch (const std::exception& ex) {
      host().log_error(std::format(
        "ExtOnnxVision: could not create shared texture: {}", ex.what()));
      trace(std::string("shared texture failed: ") + ex.what());
    }

    m_source->set_texture(host().create_texture(shared_desc));
    ensure_output_texture(desc);
    m_processor_configured = false;
  });

  ensure_processor();

  // Adopt a processor that finished building, without ever waiting on it.
  if (m_pending.valid() &&
      m_pending.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
    trace("pending processor is ready, adopting");
    if (auto built = m_pending.get()) {
      if (m_processor && m_processor->is_loading())
        m_retired.push_back(std::move(m_processor));
      m_processor = std::move(built);
      m_processor_configured = false;
      trace("processor adopted");
    }
  }

  drain_log();
  publish_status();
  publish_measurements();

  const auto current_mode = mode();
  if (current_mode != m_logged_mode) {
    m_logged_mode = current_mode;
    const auto name = (current_mode == ModeIdle ? "Idle" :
                       current_mode == ModePassthrough ? "Passthrough" : "Readback");
    host().log_info(std::format("ExtOnnxVision: mode is {}", name));
    trace(std::string("mode is ") + name);
  }

  if (current_mode == ModeIdle || !m_source->texture()) {
    m_sampler->set_texture(TextureRef(m_placeholder));
    return true;
  }

  if (current_mode == ModePassthrough) {
    // Publish the source texture straight back, with no system memory round
    // trip. Diagnostic only: without CPU access to the pixels the channel
    // order cannot be corrected, so this looks blue.
    m_sampler->set_texture(TextureRef(m_source->texture()));
    return true;
  }

  // ModeReadback below: read the frame, run the processor, publish the result.
  if (m_output_texture)
    m_sampler->set_texture(TextureRef(m_output_texture));

  if (m_stage.load() == Stage::ReadyToUpload)
    begin_upload();

  if (m_stage.load() == Stage::Idle)
    begin_download();

  return true;
}
catch (const std::exception& ex) {
  host().log_error(std::format("ExtOnnxVision: update failed: {}", ex.what()));
  return false;
}

namespace {

// Enum settings may arrive as an index or as the display name, depending on
// how the host serialises them, so accept both.
int read_enum_setting(const ValueSet& settings, string_view id,
    std::initializer_list<const char*> names, int fallback) {
  const auto raw = settings.get(id, "");
  if (raw.empty())
    return fallback;
  if (raw.find_first_not_of("0123456789") == std::string::npos)
    return std::atoi(raw.c_str());
  auto index = 0;
  for (const auto* name : names) {
    if (raw == name)
      return index;
    ++index;
  }
  return fallback;
}

} // namespace

void VisionStream::ensure_processor() noexcept try {
  // Three sources, in the order of how deliberate each one is.
  //
  // The declared setting first, for a script or a future host that actually
  // delivers it. Then the parameter, which is what an operator edits and the
  // only route that reaches a running stream — but only if it still holds
  // something path-shaped, since a reopened project leaves a number there.
  // Finally what we remembered for this layer last session, which is what
  // makes a path survive a save at all.
  auto wanted_model = setting_model_path(m_settings);
  auto from_memory = false;

  if (wanted_model.empty())
    if (auto from_parameter = (m_model_path ? m_model_path->value() : std::string());
        looks_like_a_path(from_parameter))
      wanted_model = std::move(from_parameter);

  if (wanted_model.empty()) {
    wanted_model = read_remembered_model(m_settings.get(SettingNames::instance_id, ""));
    from_memory = !wanted_model.empty();
    if (from_memory && m_model_path) {
      // Put it back on the parameter too, so the layer shows what it is
      // actually running rather than the number the project restored.
      m_model_path->set_value(wanted_model);
      host().send_event(EventCategory::ValueChanged, "model");
    }
  }

  // Each parameter was constructed with the corresponding stream setting as
  // its default, so the parameter is the single source of truth here. Reading
  // settings again would override a deliberate choice of a zero-valued option
  // such as Compute = Auto.
  const auto value_of = [](ParameterInt* p, int fallback) {
    return (p ? p->value() : fallback);
  };

  const auto wanted_normalisation = value_of(m_normalisation, 1);
  const auto wanted_activation = value_of(m_activation, 0);
  const auto wanted_target = value_of(m_target, 0);
  const auto wanted_size = value_of(m_size, 0);
  const auto wanted_effort = value_of(m_trt_effort, 3);

  const auto options = wanted_normalisation + wanted_activation * 8 +
                       wanted_target * 64 + wanted_size * 512 +
                       wanted_effort * 2048;

  // A change of the reload toggle forces a rebuild even when nothing else
  // moved, which is the point: it is there for when the host has not passed a
  // settings change through, so the extension cannot see the difference.
  // Reloading re-reads the settings and reuses the compiled engine, so it is
  // quick. Rebuilding the engine is deliberately not a parameter: it takes
  // minutes and discards work, which is the wrong shape for a control that
  // the timeline can animate and a hand can nudge. Deleting the engine file
  // named in the log is how that is asked for; a reload then rebuilds.
  auto forced = false;
  if (m_reload && m_reload->value() != m_reload_seen) {
    m_reload_seen = m_reload->value();
    forced = true;
    trace("reload requested");
  }

  // Remember a path the operator chose, not one we just read back, and only
  // once it is going to be used.
  if (!from_memory && !wanted_model.empty() && wanted_model != m_loaded_model)
    remember_model(m_settings.get(SettingNames::instance_id, ""), wanted_model);

  if (!forced &&
      wanted_model == m_loaded_model &&
      options == m_loaded_options)
    return;

  // Retire rather than destroy. The processor's destructor joins its loader
  // thread, and a
  // TensorRT compile takes minutes; doing that here would block the host's
  // render thread and freeze the layer.
  if (m_processor && m_processor->is_loading())
    m_retired.push_back(std::move(m_processor));
  m_retired.erase(
    std::remove_if(m_retired.begin(), m_retired.end(),
      [](const std::unique_ptr<FrameProcessor>& p) { return !p->is_loading(); }),
    m_retired.end());

  m_loaded_model = wanted_model;
  m_loaded_options = options;
  m_processor_configured = false;

  if (wanted_model.empty()) {
    m_processor = std::make_unique<PassthroughProcessor>();
    host().log_info("ExtOnnxVision: no model set, passing frames through");
    trace("no model set");
    return;
  }

  static const auto normalisations = std::array<Normalisation, 4>{
    Normalisation::ZeroToOne, Normalisation::ImageNet, Normalisation::MinusOneToOne,
    Normalisation::ZeroTo255 };
  static const auto activations = std::array<Activation, 4>{
    Activation::Auto, Activation::None, Activation::Sigmoid, Activation::MinMax };
  static const auto targets = std::array<OutputTarget, 3>{
    OutputTarget::Alpha, OutputTarget::Greyscale, OutputTarget::Both };

  auto config = Processor::Config{ };
  config.model_path = wanted_model;
  config.normalisation = normalisations[std::clamp(wanted_normalisation, 0, 3)];
  config.activation = activations[std::clamp(wanted_activation, 0, 3)];
  config.target = targets[std::clamp(wanted_target, 0, 2)];
  config.size_override = wanted_size;
  config.trt_build_effort = wanted_effort;

  host().log_info(std::format("ExtOnnxVision: loading model {} on TensorRT",
    wanted_model));
  trace("loading model " + wanted_model +
        (from_memory ? " (remembered for this layer from a previous session)"
                     : ""));

  // Keep passing frames through with the cheap processor while the real one is
  // built. Constructing the real one loads inference libraries and, on the
  // TensorRT path, may compile an engine — none of which belongs on the
  // host's render thread.
  m_processor = std::make_unique<PassthroughProcessor>();
  m_processor_configured = false;

  trace("dispatching processor construction to a worker");
  m_pending = std::async(std::launch::async,
    [this, config = std::move(config)]() mutable {
      // These callbacks fire on the loading thread, so they queue rather than
      // calling the host directly.
      auto processor = std::make_unique<Processor>(std::move(config),
        [this](const std::string& m) noexcept { queue_log(false, m); trace(m); },
        [this](const std::string& m) noexcept { queue_log(true, m); trace("ERROR " + m); });
      processor->begin_loading();
      return processor;
    });
}
catch (const std::exception& ex) {
  host().log_error(std::format("ExtOnnxVision: creating processor failed: {}", ex.what()));
  m_processor = std::make_unique<PassthroughProcessor>();
}

void VisionStream::drain_log() noexcept {
  auto pending = std::vector<std::pair<bool, std::string>>();
  {
    const auto lock = std::lock_guard(m_log_mutex);
    pending.swap(m_log_queue);
  }
  for (const auto& [is_error, text] : pending) {
    if (is_error)
      host().log_error("ExtOnnxVision: " + text);
    else
      host().log_info("ExtOnnxVision: " + text);
  }
}

void VisionStream::queue_log(bool is_error, const std::string& text) noexcept try {
  const auto lock = std::lock_guard(m_log_mutex);
  if (m_log_queue.size() < 200)
    m_log_queue.emplace_back(is_error, text);
}
catch (...) {
}

// Pushes the last frame's measurements onto the output parameters. Only when
// they have actually moved: each change costs the host an event, and these
// would otherwise fire four times a frame.
void VisionStream::publish_measurements() noexcept {
  if (!m_processor || !m_out_coverage)
    return;
  const auto m = m_processor->measurements();
  if (!m.valid)
    return;

  const auto publish = [&](ParameterValue* p, const char* id,
                           double value, double previous) {
    // A hundredth of a percent: below anything an operator could see, and far
    // enough above sensor noise that a static shot goes quiet.
    if (p && std::abs(value - previous) > 0.0001) {
      p->set_value(value);
      host().send_event(EventCategory::ValueChanged, id);
    }
  };

  publish(m_out_coverage, "coverage", m.coverage, m_published_measurements.coverage);
  publish(m_out_centre_x, "centre_x", m.centre_x, m_published_measurements.centre_x);
  publish(m_out_centre_y, "centre_y", m.centre_y, m_published_measurements.centre_y);
  publish(m_out_mean, "mean", m.mean, m_published_measurements.mean);
  m_published_measurements = m;
}

void VisionStream::publish_status() noexcept {
  if (!m_status || !m_processor)
    return;
  auto text = m_processor->status();
  if (text == m_published_status)
    return;

  // A frame time that varies by a millisecond changes this text on nearly
  // every frame. Republish immediately when the state itself changes, but rate
  // limit updates that only move the number, or the log fills with it.
  const auto state_of = [](const std::string& t) {
    const auto comma = t.find_last_of(',');
    return (comma == std::string::npos ? t : t.substr(0, comma));
  };
  const auto now = std::chrono::steady_clock::now();
  if (state_of(text) == state_of(m_published_status) &&
      now - m_status_published_at < std::chrono::seconds(1))
    return;

  m_status_published_at = now;
  m_published_status = text;
  m_status->set_value(text.data(), text.size());
  // Without this the host has no reason to re-read the parameter, so the
  // status can sit visibly stale while the extension is busy.
  host().send_event(EventCategory::ValueChanged, "status");

  // Also log transitions, so a long compile leaves a trail even if nobody is
  // watching the parameter.
  host().log_info("ExtOnnxVision: " + text);
  trace("status: " + text);
}

void VisionStream::log_geometry(const TextureDesc& desc) noexcept {
  if (desc.width == m_logged_width &&
      desc.height == m_logged_height &&
      desc.format == m_logged_format)
    return;

  m_logged_width = desc.width;
  m_logged_height = desc.height;
  m_logged_format = desc.format;

  const auto message = std::format(
    "ExtOnnxVision: source is {}x{} {}, {} bytes per pixel",
    desc.width, desc.height,
    std::string(get_format_name(desc.format)),
    get_bytes_per_pixel(desc.format));
  host().log_info(message);
  trace(message);

  if (get_bytes_per_pixel(desc.format) == 0)
    host().log_warning(
      "ExtOnnxVision: unrecognised source format, frames will be skipped");
}

void VisionStream::ensure_output_texture(const TextureDesc& source_desc) noexcept {
  const auto current = m_output_texture.desc();
  if (current.width == source_desc.width &&
      current.height == source_desc.height &&
      current.format == source_desc.format)
    return;

  auto desc = TextureDesc{ };
  desc.width = source_desc.width;
  desc.height = source_desc.height;
  desc.format = source_desc.format;
  m_output_texture = host().create_texture(desc);

  const auto lock = std::lock_guard(m_mutex);
  m_width = desc.width;
  m_height = desc.height;
  m_format = desc.format;
  m_bytes_per_pixel = get_bytes_per_pixel(desc.format);
  m_pitch = m_width * m_bytes_per_pixel;
  m_output_bytes.assign(m_pitch * m_height, uint8_t{ });
}

void VisionStream::begin_download() noexcept {
  if (m_bytes_per_pixel == 0 || !m_reader)
    return;

  // Read the texture PIXERA rendered into, on this thread. The D3D immediate
  // context is not thread safe, so the map must happen here rather than on a
  // worker, but it is only a copy: the slow work is done asynchronously below.
  auto pitch = size_t{ };
  auto ok = false;
  {
    const auto lock = std::lock_guard(m_mutex);
    const auto t0 = std::chrono::steady_clock::now();
    ok = m_reader->read(m_input_bytes, pitch);
    m_readback_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t0).count();
    if (ok)
      m_pitch = pitch;
  }

  if (!ok) {
    if (!m_warned_read_failed) {
      m_warned_read_failed = true;
      host().log_error("ExtOnnxVision: reading the source texture failed");
      trace("read failed");
    }
    return;
  }

  if (!m_reported_read) {
    m_reported_read = true;
    const auto message = std::format(
      "ExtOnnxVision: read {} bytes from source, pitch {}",
      m_input_bytes.size(), pitch);
    host().log_info(message);
    trace(message);
  }

  m_stage.store(Stage::Processing);
  host().async([this]() noexcept {
    run_processor();
    m_stage.store(Stage::ReadyToUpload);
  });
}

void VisionStream::capture_input(const BufferDesc& buffer) noexcept {
  const auto lock = std::lock_guard(m_mutex);
  if (!buffer.data || buffer.size == 0) {
    m_input_bytes.clear();
    return;
  }
  const auto bytes = static_cast<const uint8_t*>(buffer.data);
  m_input_bytes.assign(bytes, bytes + buffer.size);
  // The host's pitch is authoritative; it may exceed width * bytes_per_pixel.
  m_pitch = (buffer.pitch ? buffer.pitch : m_width * m_bytes_per_pixel);
}

void VisionStream::run_processor() noexcept {
  const auto lock = std::lock_guard(m_mutex);
  if (m_input_bytes.empty() || m_output_bytes.empty())
    return;

  auto view = FrameView{ };
  view.data = m_input_bytes.data();
  view.width = m_width;
  view.height = m_height;
  view.pitch = m_pitch;
  view.bytes_per_pixel = m_bytes_per_pixel;
  view.format_name = std::string(get_format_name(m_format));
  view.bgr_order = (m_swap_rb && m_swap_rb->value());

  // Never read past what was actually downloaded.
  const auto available_rows = (view.pitch ? m_input_bytes.size() / view.pitch : 0);
  if (available_rows < view.height)
    view.height = available_rows;

  auto out = FrameBuffer{ };
  out.data = m_output_bytes.data();
  out.width = m_width;
  out.height = m_height;
  out.pitch = m_width * m_bytes_per_pixel;
  out.bytes_per_pixel = m_bytes_per_pixel;

  if (!m_processor_configured) {
    m_processor_configured = m_processor->configure(view);
    if (!m_processor_configured)
      host().log_warning(std::format(
        "ExtOnnxVision: processor {} rejected {} frames, passing through",
        m_processor->name(), view.format_name));
  }

  auto processed = false;
  if (m_processor_configured)
    processed = m_processor->process(view, out);

  // Degrade to passthrough rather than showing a black layer.
  if (!processed)
    copy_rows(view, out);

  if (m_swap_rb && m_swap_rb->value())
    swap_red_blue(out);

  ++m_frames;
  m_inference_ms = m_processor->last_inference_ms();
  if (m_processor->is_loading() && m_frames % 60 == 1)
    trace("still publishing frames while loading, frame " +
          std::to_string(m_frames));
  if (m_inference_ms > 0.0 && m_frames % 30 == 1) {
    host().monitor_value("inference_ms", m_inference_ms);
    trace("timing: readback " +
          std::to_string(static_cast<int>(m_readback_ms + 0.5)) + " ms, " +
          m_processor->timing_summary());
  }

  if (!m_reported_first_frame) {
    m_reported_first_frame = true;
    const auto message = std::format(
      "ExtOnnxVision: first frame processed, {}x{}, {} bytes downloaded",
      m_width, m_height, m_input_bytes.size());
    host().log_info(message);
    trace(message);
  }
}

void VisionStream::begin_upload() noexcept {
  auto buffer = BufferDesc{ };
  {
    const auto lock = std::lock_guard(m_mutex);
    if (m_output_bytes.empty() || !m_output_texture) {
      m_stage.store(Stage::Idle);
      return;
    }
    buffer.data = m_output_bytes.data();
    buffer.pitch = m_width * m_bytes_per_pixel;
    buffer.size = m_output_bytes.size();
  }

  if (!m_reported_upload) {
    m_reported_upload = true;
    trace("first upload of processed frame");
  }

  // upload_copy = true means the host takes its own copy before returning, so
  // the buffer is free immediately and there is nothing to wait for. Gating
  // the next frame on the completion callback deadlocked the pipeline in
  // PIXERA, where that callback is not always invoked: the stage stayed at
  // Uploading indefinitely and the layer froze after one frame.
  host().upload_texture(TextureRef(m_output_texture), buffer, true,
    []() noexcept { });
  m_stage.store(Stage::Idle);
}

} // namespace
