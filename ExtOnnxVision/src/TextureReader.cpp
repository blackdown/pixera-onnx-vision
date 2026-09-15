
#include "TextureReader.h"
#include <d3d11.h>
#include <stdexcept>
#include <cstring>

namespace rxext::onnxvision {

unsigned int get_dxgi_format(Format format) {
  switch (format) {
    case Format::None: break;
    case Format::R8_UNORM: return DXGI_FORMAT_R8_UNORM;
    case Format::R8G8_UNORM: return DXGI_FORMAT_R8G8_UNORM;
    case Format::R8G8B8A8_UNORM: return DXGI_FORMAT_R8G8B8A8_UNORM;
    case Format::B8G8R8A8_UNORM: return DXGI_FORMAT_B8G8R8A8_UNORM;
    case Format::R16G16B16A16_SFLOAT: return DXGI_FORMAT_R16G16B16A16_FLOAT;
    case Format::R32G32B32A32_SFLOAT: return DXGI_FORMAT_R32G32B32A32_FLOAT;
  }
  throw std::runtime_error("unsupported texture format");
}

TextureReader::TextureReader(d3d11::Device& device, size_t width, size_t height,
    Format format)
    : m_device(device),
      m_width(width),
      m_height(height),
      m_format(format),
      m_shared(device.device(), width, height,
        static_cast<DXGI_FORMAT>(get_dxgi_format(format)),
        d3d11::Texture::Default) {

  // Staging texture: CPU readable, not shared, no bind flags.
  auto desc = D3D11_TEXTURE2D_DESC{ };
  desc.Width = static_cast<UINT>(width);
  desc.Height = static_cast<UINT>(height);
  desc.Format = static_cast<DXGI_FORMAT>(get_dxgi_format(format));
  desc.ArraySize = 1;
  desc.SampleDesc = { 1, 0 };
  desc.MipLevels = 1;
  desc.Usage = D3D11_USAGE_STAGING;
  desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
  desc.BindFlags = 0;

  if (FAILED(device.device()->CreateTexture2D(&desc, nullptr, &m_staging)))
    throw std::runtime_error("creating staging texture failed");
}

TextureReader::~TextureReader() = default;

void* TextureReader::share_handle() const {
  return m_shared.share_handle();
}

bool TextureReader::read(std::vector<uint8_t>& out, size_t& pitch) noexcept try {
  auto* context = m_device.device_context();
  if (!context || !m_staging)
    return false;

  // GPU-side copy from the texture PIXERA rendered into, to one the CPU can map.
  context->CopyResource(m_staging, const_cast<d3d11::Texture&>(m_shared).texture());

  auto mapped = D3D11_MAPPED_SUBRESOURCE{ };
  if (FAILED(context->Map(m_staging, 0, D3D11_MAP_READ, 0, &mapped)))
    return false;

  pitch = mapped.RowPitch;
  out.resize(pitch * m_height);
  if (mapped.pData)
    std::memcpy(out.data(), mapped.pData, out.size());

  context->Unmap(m_staging, 0);
  return (mapped.pData != nullptr);
}
catch (...) {
  return false;
}

} // namespace
