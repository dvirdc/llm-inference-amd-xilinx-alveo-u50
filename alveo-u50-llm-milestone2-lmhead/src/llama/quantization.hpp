// quantization.hpp -- INT8 weight / activation quantization that matches
// the FPGA kernel's arithmetic *bit-exactly*.
//
// Strategy (very simple by design):
//   * Weights: per-row symmetric INT8. Each row gets its own scale.
//   * Activations: dynamic symmetric INT8 with a single scalar scale per
//     forward call. (We re-quantize the hidden vector every token.)
//   * Accumulator: INT32 (kernel uses int32, so we use int32 here).
//   * Dequantize after the GEMV by multiplying out int32_y * row_scale * x_scale.
//
// Why this is enough for CPU == FPGA bit-exact:
//   * The quantizer is deterministic FP32 -> INT8.
//   * The INT8*INT8 -> INT32 accumulation is associative within int32
//     (no overflow at our sizes -- see docs/quantization.md), so doing it
//     on CPU and on the kernel produces the same int32 sum.
//   * The dequant step is FP32, so we compare *before* dequantizing for
//     the strict bit-exact check.
//
// Why FP32 reference vs INT8 won't match exactly:
//   * Quantization loses precision. We report max/mean abs error and
//     top-k overlap as a perceptual proxy instead.

#pragma once

#include <cstdint>
#include <vector>

namespace m2 {

struct QuantizedMatrix {
    int                  rows = 0;
    int                  cols = 0;
    // Row-major, length rows*cols. data[r*cols + c] is INT8 weight.
    std::vector<int8_t>  data;
    // Length rows. row_scales[r] is the scalar that maps int8 row r back
    // to float: float ~= int8 * row_scales[r].
    std::vector<float>   row_scales;
};

struct QuantizedVector {
    int                  size  = 0;
    std::vector<int8_t>  data;
    float                scale = 1.0f;  // float ~= int8 * scale
};

// Per-row symmetric INT8 quantization. `matrix` is row-major
// [rows, cols]. We never throw on all-zero rows -- the scale falls back
// to 1.0 and all int8 entries stay 0 (the GEMV result is then 0 too).
QuantizedMatrix quantize_matrix_per_row_int8(const std::vector<float>& matrix,
                                             int rows, int cols);

// Dynamic symmetric INT8 quantization of an activation vector.
QuantizedVector quantize_vector_int8_dynamic(const std::vector<float>& vector);

// Pure CPU INT8 GEMV with INT32 accumulation.
//   y_int32[r] = sum_c weights.data[r*cols + c] * x.data[c]
// Layout must exactly match the kernel's m_axi reads (row-major weights,
// flat activation vector). This is the function that's required to be
// bit-exact with the FPGA output.
void gemv_cpu_int8(const QuantizedMatrix& weights,
                   const QuantizedVector& x,
                   std::vector<int32_t>&  y_int32);

// Multiplies through the scales to give the FP32 GEMV result you'd get
// from doing the matmul in FP32 *if the quantization were lossless*. In
// reality there's quantization error -- see compare.hpp.
void dequantize_gemv_output(const std::vector<int32_t>& y_int32,
                            const std::vector<float>&   row_scales,
                            float                       x_scale,
                            std::vector<float>&         y_fp32);

// FP32 reference matmul `y = W @ x`. Plain row-by-row dot products. Used
// as ground truth for the comparison reports.
void gemv_cpu_fp32(const std::vector<float>& weights,
                   const std::vector<float>& x,
                   int rows, int cols,
                   std::vector<float>& y);

} // namespace m2
