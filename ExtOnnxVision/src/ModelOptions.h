#pragma once

namespace rxext::onnxvision {

// How pixel values are scaled before the model sees them. The right one is a
// property of how the model was trained and cannot be read from the file.
enum class Normalisation {
  ZeroToOne,      // 0..1, no further scaling
  ImageNet,       // (x - mean) / std, the usual torchvision convention
  MinusOneToOne,  // x * 2 - 1
  ZeroTo255,      // x * 255, as style transfer models expect
};

// What to do with a single-channel output map before displaying it.
enum class Activation {
  Auto,     // infer from the observed value range
  None,     // already 0..1
  Sigmoid,  // raw logits, e.g. BiRefNet
  MinMax,   // arbitrary range, e.g. depth; rescaled by its own min and max
};

// Where a single-channel result is written in the output frame.
enum class OutputTarget {
  Alpha,      // matte: colour passes through, map becomes alpha
  Greyscale,  // visualise the map itself, alpha left opaque
  Both,
};

} // namespace
