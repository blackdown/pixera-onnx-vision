
#include "Device.h"
#include "VisionStream.h"
#include "Trace.h"
#include <format>

namespace rxext::onnxvision {

Device::Device(ValueSet settings)
    : m_settings(std::move(settings)) {
}

bool Device::initialize() noexcept try {
  // Listed once, verbatim, as a diagnostic: it shows exactly what the host
  // handed this device.
  {
    auto listed = std::string();
    for (const auto& value : m_settings.values)
      listed += (listed.empty() ? "" : ", ") +
                std::string(value.name) + "='" + std::string(value.value) + "'";
    trace("device settings: " + (listed.empty() ? "(none)" : listed));
  }

  const auto adapter_luid = m_settings.get(SettingNames::adapter_luid);
  m_d3d_device = std::make_shared<d3d11::Device>(adapter_luid);
  host().log_info(std::format(
    "ExtOnnxVision: device initialized on adapter {}",
    adapter_luid.empty() ? std::string("(default)") : std::string(adapter_luid)));
  return true;
}
catch (const std::exception& ex) {
  host().log_error(std::format("ExtOnnxVision: device init failed: {}", ex.what()));
  return false;
}

string Device::get_property(string_view name) noexcept {
  if (name == PropertyNames::name)
    return string("ONNX Vision");

  return { };
}

vector<ValueSet> Device::enumerate_stream_settings() noexcept {
  // One vision stream for now. Additional entries here would appear as
  // separate selectable streams, e.g. to run different models side by side.
  auto streams = vector<ValueSet>();
  auto& stream = streams.emplace_back();
  stream.set(SettingNames::name, "Vision");
  stream.set(SettingNames::handle, "vision");
  return streams;
}

InputStream* Device::create_input_stream(ValueSet settings) noexcept try {
  // Device settings are merged into the stream's, with the stream's own values
  // winning. A script configuring a run without a UI supplies settings this
  // way; PIXERA's Setup tab does not.
  for (const auto& value : m_settings.values)
    if (settings.get(value.name, "").empty())
      settings.set(value.name, value.value);
  return new VisionStream(m_d3d_device, std::move(settings));
}
catch (const std::exception& ex) {
  host().log_error(std::format("ExtOnnxVision: creating stream failed: {}", ex.what()));
  return nullptr;
}

} // namespace
