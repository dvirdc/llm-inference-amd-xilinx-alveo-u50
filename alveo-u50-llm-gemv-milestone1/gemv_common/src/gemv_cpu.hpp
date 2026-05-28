// gemv_cpu.hpp -- plain-C++ INT8 GEMV reference used for correctness checking.
//
// We never optimize this. The whole point is to be obviously correct so we can
// trust it as ground truth for the FPGA kernel.

#pragma once

#include <cstdint>
#include <vector>

namespace gemv {

// Computes y[row] = sum_{col} weights[row*cols + col] * x[col].
//   weights : signed int8, length rows*cols, row-major
//   x       : signed int8, length cols
//   y       : signed int32, length rows  (resized inside)
//
// Asserts in debug builds that sizes are consistent. Throws std::invalid_argument
// at runtime in release builds so the caller can fail cleanly.
void gemv_cpu_ref(const std::vector<int8_t>&  weights,
                  const std::vector<int8_t>&  x,
                  std::vector<int32_t>&       y,
                  int                         rows,
                  int                         cols);

} // namespace gemv
