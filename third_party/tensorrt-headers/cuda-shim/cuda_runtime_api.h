// A stand-in for the CUDA toolkit's cuda_runtime_api.h.
//
// TensorRT's public headers include that file, but across the whole API they
// use exactly two things from it: cudaStream_t and cudaEvent_t, both opaque
// pointers. Nothing in this extension calls the CUDA runtime — GPU memory and
// the stream go through the driver API in nvcuda.dll, which ships with the
// display driver.
//
// Declaring those two types here means the CUDA toolkit is not needed to
// build, and cudart64_12.dll is not needed to deploy. The definitions match
// the toolkit's exactly: both are typedefs of the driver API's own handle
// types, which is why a stream created with cuStreamCreate can be handed
// straight to enqueueV3.
//
// If a future TensorRT release uses more of the CUDA runtime than this, the
// build breaks at compile time and names the missing symbol, rather than
// failing at run time.

#ifndef EXTONNX_CUDA_RUNTIME_API_SHIM_H
#define EXTONNX_CUDA_RUNTIME_API_SHIM_H

typedef struct CUstream_st* cudaStream_t;
typedef struct CUevent_st* cudaEvent_t;

#endif
