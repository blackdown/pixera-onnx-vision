# TensorRT headers

Unmodified copies of NVIDIA's public TensorRT headers, used only to compile
against. No TensorRT libraries are included in this repository.

| Files | Source | Licence |
|---|---|---|
| `NvInfer*.h` | [NVIDIA/TensorRT](https://github.com/NVIDIA/TensorRT) at tag `v10.13.3`, `include/` | Apache 2.0 |
| `NvOnnxParser.h` | [onnx/onnx-tensorrt](https://github.com/onnx/onnx-tensorrt) at branch `release/10.13-GA` | Apache 2.0 |

Copyright NVIDIA Corporation & Affiliates. The licence text is in
[`LICENSE`](LICENSE), and each header carries its own licence notice.

`v10.13.3` is the newest release NVIDIA has published headers for. The
extension does not rely on these matching the installed TensorRT: it resolves
TensorRT's entry points at run time and passes the version the installed
library reports, so it works against later TensorRT 10 releases.

## `cuda-shim/`

Not NVIDIA's. `cuda-shim/cuda_runtime_api.h` stands in for the CUDA toolkit
header of the same name, which TensorRT's headers include. Across the whole
TensorRT API they use two opaque handle types from it, and the shim declares
exactly those two, so no CUDA toolkit is needed to build. See the file for
details.
