#include "quantization.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "../common/errors.hpp"

namespace m2 {

namespace {

// Symmetric INT8 keeps -127 as the negative bound -- not -128 -- so that
// `q = round(x / scale)` is symmetric around 0 and there's no asymmetry
// bias in the dot products. This matches what most production INT8 paths
// do (TensorRT, ONNXRuntime quant tools, etc.).
constexpr int kQMax = 127;

inline int8_t saturate_round(float v) {
    int r = static_cast<int>(std::lround(v));
    if (r >  kQMax) r =  kQMax;
    if (r < -kQMax) r = -kQMax;
    return static_cast<int8_t>(r);
}

} // anonymous

QuantizedMatrix quantize_matrix_per_row_int8(const std::vector<float>& matrix,
                                             int rows, int cols) {
    M2_CHECK(rows > 0 && cols > 0, ShapeError,
             "quantize_matrix: rows=" << rows << " cols=" << cols);
    M2_CHECK(matrix.size() == size_t(rows) * size_t(cols), ShapeError,
             "quantize_matrix: size " << matrix.size()
             << " != rows*cols " << (size_t(rows) * size_t(cols)));

    QuantizedMatrix qm;
    qm.rows = rows;
    qm.cols = cols;
    qm.data.assign(size_t(rows) * size_t(cols), 0);
    qm.row_scales.assign(rows, 1.0f);

    for (int r = 0; r < rows; ++r) {
        const float* row_ptr = matrix.data() + size_t(r) * size_t(cols);

        // amax across the row.
        float amax = 0.0f;
        for (int c = 0; c < cols; ++c) {
            float v = std::fabs(row_ptr[c]);
            if (v > amax) amax = v;
        }

        if (amax == 0.0f) {
            // All-zero row -- scale=1, int8 stays 0. Avoids div-by-zero.
            qm.row_scales[r] = 1.0f;
            continue;
        }
        float scale = amax / float(kQMax);
        qm.row_scales[r] = scale;
        const float inv  = 1.0f / scale;

        int8_t* qrow = qm.data.data() + size_t(r) * size_t(cols);
        for (int c = 0; c < cols; ++c) {
            qrow[c] = saturate_round(row_ptr[c] * inv);
        }
    }
    return qm;
}

QuantizedVector quantize_vector_int8_dynamic(const std::vector<float>& vector) {
    QuantizedVector qv;
    qv.size = static_cast<int>(vector.size());
    qv.data.assign(qv.size, 0);

    if (qv.size == 0) {
        qv.scale = 1.0f;
        return qv;
    }
    float amax = 0.0f;
    for (float v : vector) {
        float a = std::fabs(v);
        if (a > amax) amax = a;
    }
    if (amax == 0.0f) {
        qv.scale = 1.0f;
        return qv;
    }
    qv.scale = amax / float(kQMax);
    const float inv = 1.0f / qv.scale;
    for (int i = 0; i < qv.size; ++i) {
        qv.data[i] = saturate_round(vector[i] * inv);
    }
    return qv;
}

void gemv_cpu_int8(const QuantizedMatrix& weights,
                   const QuantizedVector& x,
                   std::vector<int32_t>&  y_int32) {
    M2_CHECK(weights.cols == x.size, ShapeError,
             "gemv_cpu_int8: weights.cols=" << weights.cols
             << " x.size=" << x.size);

    y_int32.assign(weights.rows, 0);
    for (int r = 0; r < weights.rows; ++r) {
        // Plain int32 accumulation -- same arithmetic the kernel performs.
        // Bound: |row sum| <= cols * 127 * 127. For cols=4096 that's
        // ~6.6e7, well under int32 range (~2.1e9). Safe.
        int32_t acc = 0;
        const int8_t* w_row = weights.data.data() + size_t(r) * size_t(weights.cols);
        for (int c = 0; c < weights.cols; ++c) {
            acc += int32_t(w_row[c]) * int32_t(x.data[c]);
        }
        y_int32[r] = acc;
    }
}

void dequantize_gemv_output(const std::vector<int32_t>& y_int32,
                            const std::vector<float>&   row_scales,
                            float                       x_scale,
                            std::vector<float>&         y_fp32) {
    M2_CHECK(y_int32.size() == row_scales.size(), ShapeError,
             "dequantize: y=" << y_int32.size()
             << " scales=" << row_scales.size());
    y_fp32.resize(y_int32.size());
    for (size_t i = 0; i < y_int32.size(); ++i) {
        y_fp32[i] = float(y_int32[i]) * row_scales[i] * x_scale;
    }
}

void gemv_cpu_fp32(const std::vector<float>& weights,
                   const std::vector<float>& x,
                   int rows, int cols,
                   std::vector<float>& y) {
    M2_CHECK(weights.size() == size_t(rows) * size_t(cols), ShapeError,
             "gemv_cpu_fp32: weights " << weights.size()
             << " vs rows*cols " << (size_t(rows) * size_t(cols)));
    M2_CHECK(x.size() == size_t(cols), ShapeError,
             "gemv_cpu_fp32: x " << x.size() << " vs cols " << cols);
    y.assign(rows, 0.0f);
    for (int r = 0; r < rows; ++r) {
        float acc = 0.0f;
        const float* row_ptr = weights.data() + size_t(r) * size_t(cols);
        for (int c = 0; c < cols; ++c) acc += row_ptr[c] * x[c];
        y[r] = acc;
    }
}

} // namespace m2
