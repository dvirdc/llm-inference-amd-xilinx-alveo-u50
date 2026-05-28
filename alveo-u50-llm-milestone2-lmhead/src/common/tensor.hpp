// tensor.hpp -- intentionally tiny "tensor" type.
//
// We do NOT pull in Eigen / xtensor / Armadillo. For this milestone every
// tensor in the model is either:
//   * a flat std::vector<float>   (FP32 weights / activations),
//   * a flat std::vector<int8_t>  (quantized weights / activations), or
//   * a flat std::vector<int32_t> (kernel accumulator output).
//
// We just keep their dimensions in a small struct so call sites can carry
// shape information around without duplicating two ints next to every
// vector argument. Memory layout is always *row-major* and that fact is
// load-bearing for FPGA-vs-CPU bit-exact correctness (see docs/quantization.md).
//
// NOTE: this header is std::vector-only on purpose. Anything heavier should
// live in its own file -- e.g. QuantizedMatrix in src/llama/quantization.hpp.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace m2 {

// Lightweight 2-D shape descriptor.
struct Shape2D {
    int rows = 0;
    int cols = 0;
    constexpr size_t numel() const { return size_t(rows) * size_t(cols); }
};

// Helper: index a row-major 2-D buffer. Inline -- the compiler will collapse
// it into the same code you'd get from hand-written `buf[r*cols + c]`.
template <typename T>
inline T& at(std::vector<T>& v, const Shape2D& s, int r, int c) {
    return v[size_t(r) * size_t(s.cols) + size_t(c)];
}
template <typename T>
inline const T& at(const std::vector<T>& v, const Shape2D& s, int r, int c) {
    return v[size_t(r) * size_t(s.cols) + size_t(c)];
}

} // namespace m2
