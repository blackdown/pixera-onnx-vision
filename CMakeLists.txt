cmake_minimum_required(VERSION 3.14)
project(RXExt)

# Build only this extension, not the SDK's sample extensions.
set(OMIT_SAMPLE_PROJECTS true)

# Optionally deploy straight into a PIXERA installation. Set PIXERA_BUILD to
# the installed build folder, for example:
#
#   cmake -B build -DPIXERA_BUILD="C:/Program Files/AV Stumpfl/Pixera/<build>"
#
# Left unset, the extension is deployed into the SDK's bundled RX test engine.
if(DEFINED PIXERA_BUILD)
  set(PIXERA_DIR "${PIXERA_BUILD}")
  set(RX_DIR "${PIXERA_BUILD}/RX")
  set(DEPLOY_DIR "${PIXERA_BUILD}/data/rxExt/ExtOnnxVision")
endif()

include(render-bridge-sdk/RenderBridge/CMakeLists.txt)
set(EXTENSIONS_DIR ${CMAKE_CURRENT_SOURCE_DIR})

#--------------------------------------------------------------------------

# TensorRT headers, vendored from NVIDIA's Apache-2.0 repositories. Nothing is
# linked: nvinfer, nvonnxparser and the CUDA driver library nvcuda are resolved
# at run time, which lets one build work against whichever TensorRT 10 is
# installed and keeps the deployed extension to a single file.
#
# The cuda-shim directory supplies cuda_runtime_api.h, which TensorRT's headers
# include for two opaque handle types, so no CUDA toolkit is needed to build.
set(TENSORRT_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/third_party/tensorrt-headers")

rx_extension("ExtOnnxVision")
  target_include_directories(${PROJECT} PRIVATE
    ${PROJECT_DIR}/src
    ${TENSORRT_INCLUDE}
    ${TENSORRT_INCLUDE}/cuda-shim
  )
  # Windows headers define min/max as macros, which breaks std::min/std::max.
  target_compile_definitions(${PROJECT} PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
  # The SDK's own sources define those too; the redefinition is harmless.
  target_compile_options(${PROJECT} PRIVATE /wd4005)

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT})
