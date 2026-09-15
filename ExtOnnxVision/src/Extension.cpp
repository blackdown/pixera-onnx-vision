
#include "Extension.h"
#include "Device.h"
#include "Trace.h"
#include <format>

namespace rxext::onnxvision {

constexpr auto kVersion = "0.3.0";

bool Extension::initialize() noexcept {
  // Nothing heavy happens here. TensorRT is located and loaded the first time
  // a model is set, on that model's own loading thread, so adding the extension
  // to a project costs nothing, and a machine without TensorRT still loads it
  // and reports the problem where it is useful: against the model.
  host().log_info(std::format("ExtOnnxVision {} loaded (built {})",
    kVersion, __DATE__));
  trace(std::format("ExtOnnxVision {} loaded", kVersion));
  return true;
}

string Extension::get_property(string_view name) noexcept {
  if (name == PropertyNames::name)
    return string("ONNX Vision");
  if (name == "version")
    return string(kVersion);
  if (name == PropertyNames::build_date)
    return string(__DATE__);

  // No stream_device_settings_desc. PIXERA renders the controls such a
  // description declares but does not pass their values back to the extension,
  // so every control there would do nothing. All configuration is on the
  // layer's parameters instead.
  return { };
}

StreamDevice* Extension::create_stream_device(ValueSet settings) noexcept try {
  return new Device(std::move(settings));
}
catch (const std::exception& ex) {
  host().log_error(std::format("ExtOnnxVision: creating device failed: {}", ex.what()));
  return nullptr;
}

} // namespace

//-------------------------------------------------------------------------

rxext::ExtensionP* rxext_open() {
  return new rxext::onnxvision::Extension();
}

void rxext_close(rxext::ExtensionP* extension) {
  delete static_cast<rxext::onnxvision::Extension*>(extension);
}
