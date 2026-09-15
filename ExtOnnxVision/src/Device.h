#pragma once

#include "rxext_client.h"
#include "d3d11/d3d11.h"
#include <memory>

namespace rxext::onnxvision {

class Device final : public rxext::StreamDevice {
public:
  explicit Device(ValueSet settings);

  bool initialize() noexcept override;
  string get_property(string_view name) noexcept override;
  vector<ValueSet> enumerate_stream_settings() noexcept override;
  InputStream* create_input_stream(ValueSet settings) noexcept override;

private:
  ValueSet m_settings;   // device settings, as supplied by the host
  // Owned D3D11 device, used to create textures PIXERA renders into and that
  // the extension can read back on the CPU. The adapter comes from the
  // adapter_luid setting PIXERA supplies, so this follows PIXERA's GPU choice.
  std::shared_ptr<d3d11::Device> m_d3d_device;
};

} // namespace
