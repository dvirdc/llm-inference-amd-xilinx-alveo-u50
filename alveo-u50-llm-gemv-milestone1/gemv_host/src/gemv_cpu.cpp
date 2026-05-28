#include "gemv_cpu.hpp"

#include <cassert>
#include <cstddef>
#include <stdexcept>

namespace gemv {

void gemv_cpu_ref(const std::vector<int8_t>&  weights,
                  const std::vector<int8_t>&  x,
                  std::vector<int32_t>&       y,
                  int                         rows,
                  int                         cols) {
    if (rows <= 0 || cols <= 0) {
        throw std::invalid_argument("gemv_cpu_ref: rows and cols must be positive");
    }
    const size_t expected_w = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    if (weights.size() != expected_w) {
        throw std::invalid_argument("gemv_cpu_ref: weights size != rows*cols");
    }
    if (x.size() != static_cast<size_t>(cols)) {
        throw std::invalid_argument("gemv_cpu_ref: x size != cols");
    }

    y.assign(static_cast<size_t>(rows), 0);

    // Triple-checked simple loop. INT8 * INT8 fits in 16 bits, the running
    // sum can grow up to rows*cols*128*128 which for rows=cols=4096 is
    // ~2.7e11 -- well outside int32. With our test data range of [-8, 7]
    // the bound shrinks to cols*64 which is at most 4096*64 = 2.6e5, so
    // int32 is safe by orders of magnitude.
    //
    // Promote both operands to int32 before multiplying so the product is
    // computed in 32-bit and we don't tempt narrow-multiply surprises.
    for (int row = 0; row < rows; ++row) {
        int32_t acc = 0;
        const int8_t* w_row = weights.data() + static_cast<size_t>(row) * cols;
        for (int col = 0; col < cols; ++col) {
            acc += static_cast<int32_t>(w_row[col]) *
                   static_cast<int32_t>(x[col]);
        }
        y[row] = acc;
    }
}

} // namespace gemv
