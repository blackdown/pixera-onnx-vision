#pragma once

#include "rxext_client.h"

namespace rxext::onnxvision {

class Extension final : public rxext::Extension {
public:
  bool initialize() noexcept override;
  string get_property(string_view name) noexcept override;
  StreamDevice* create_stream_device(ValueSet settings) noexcept override;
};

} // namespace
