#pragma once

#include "rxext_client.h"
#include "d3d11/d3d11.h"
#include "d3d11/ComPointer.h"
#include <cstdint>
#include <vector>

struct ID3D11Texture2D;

namespace rxext::onnxvision {

// Reads PIXERA content into system memory without HostContext::download_texture().
//
// download_texture() does not work from an InputStream: it crashes the RX test
// engine and silently never invokes its callback in PIXERA. See README
// "Known limitations".
//
// Instead the extension owns the texture. It creates a shared D3D11 texture,
// hands the share handle to PIXERA so PIXERA renders the connected Resource
// into it, then copies that into a staging texture it can map and read on the
// CPU. This is the same sharing mechanism ExtSampleD3D uses; only the readback
// is new.
class TextureReader {
public:
  TextureReader(d3d11::Device& device, size_t width, size_t height, Format format);
  ~TextureReader();

  TextureReader(const TextureReader&) = delete;
  TextureReader& operator=(const TextureReader&) = delete;

  // Share handle to give PIXERA via TextureDesc::share_handle.
  void* share_handle() const;

  size_t width() const { return m_width; }
  size_t height() const { return m_height; }
  Format format() const { return m_format; }

  // Copies the current contents into `out`, setting `pitch` to the row stride.
  // Must be called on the same thread that owns the D3D immediate context,
  // i.e. from update(). Returns false if the map failed.
  bool read(std::vector<uint8_t>& out, size_t& pitch) noexcept;

private:
  d3d11::Device& m_device;
  const size_t m_width;
  const size_t m_height;
  const Format m_format;
  d3d11::Texture m_shared;
  ComPointer<ID3D11Texture2D> m_staging;
};

// Maps an rxext Format onto its DXGI equivalent. Throws if unsupported.
unsigned int get_dxgi_format(Format format);

} // namespace
