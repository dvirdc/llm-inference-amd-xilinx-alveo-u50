// test_quantization.cpp -- per-row int8 weight quantization, dynamic int8
// activation quantization, INT8 GEMV with INT32 accumulator, dequant.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../src/llama/quantization.hpp"

namespace {
int g_pass = 0, g_fail = 0;
#define EXPECT_TRUE(x) do { if (x) { ++g_pass; } else { ++g_fail; \
  std::printf("  FAIL [%s:%d] %s\n", __FILE__, __LINE__, #x); } } while (0)
#define EXPECT_EQ(a, b) do { auto _a=(a); auto _b=(b); if (_a==_b) {++g_pass;} \
  else { ++g_fail; std::printf("  FAIL [%s:%d] %s != %s (%lld vs %lld)\n", \
  __FILE__, __LINE__, #a, #b, (long long)_a, (long long)_b); } } while (0)

void test_quantize_matrix_range() {
    std::printf("[test] quantize matrix stays in int8 range\n");
    std::vector<float> w = {1.0f, -2.0f, 0.5f,  3.0f, -100.0f, 50.0f};
    auto qm = m2::quantize_matrix_per_row_int8(w, 2, 3);
    EXPECT_EQ(qm.rows, 2);
    EXPECT_EQ(qm.cols, 3);
    for (auto v : qm.data) EXPECT_TRUE(v >= -127 && v <= 127);
    EXPECT_TRUE(qm.row_scales[0] > 0.0f);
    EXPECT_TRUE(qm.row_scales[1] > 0.0f);
}

void test_zero_row_does_not_divide_by_zero() {
    std::printf("[test] zero row -> scale=1, all zeros\n");
    std::vector<float> w = {0.0f, 0.0f, 0.0f,  1.0f, -1.0f, 0.5f};
    auto qm = m2::quantize_matrix_per_row_int8(w, 2, 3);
    EXPECT_EQ(qm.row_scales[0], 1.0f);
    for (int c = 0; c < 3; ++c) EXPECT_EQ(qm.data[c], (int8_t)0);
}

void test_zero_vector_does_not_divide_by_zero() {
    std::printf("[test] zero activation vector -> scale=1, all zeros\n");
    std::vector<float> v(8, 0.0f);
    auto qv = m2::quantize_vector_int8_dynamic(v);
    EXPECT_EQ(qv.size, 8);
    EXPECT_EQ(qv.scale, 1.0f);
    for (auto x : qv.data) EXPECT_EQ(x, (int8_t)0);
}

void test_int8_gemv_hand_checked() {
    std::printf("[test] CPU INT8 GEMV hand-checked\n");
    // weights = [[ 1,  2,  3],
    //            [-1,  0,  4]]  rows=2, cols=3
    // x       = [ 2, -1,  3]
    // expected y = [ 2 - 2 + 9,    -2 + 0 + 12 ] = [9, 10]
    m2::QuantizedMatrix qm;
    qm.rows = 2; qm.cols = 3;
    qm.data = {1, 2, 3, -1, 0, 4};
    qm.row_scales = {1.0f, 1.0f};
    m2::QuantizedVector qx; qx.size = 3; qx.data = {2, -1, 3}; qx.scale = 1.0f;
    std::vector<int32_t> y;
    m2::gemv_cpu_int8(qm, qx, y);
    EXPECT_EQ(y.size(), (size_t)2);
    EXPECT_EQ(y[0],  9);
    EXPECT_EQ(y[1], 10);
}

void test_dequant_roundtrip_close_to_fp32() {
    std::printf("[test] dequant roundtrip close to FP32 reference\n");
    // small random-ish problem; quantization error should be small relative
    // to magnitude.
    const int rows = 8, cols = 16;
    std::vector<float> w(rows * cols), x(cols);
    unsigned state = 1234;
    auto nf = [&]() {
        state = state * 1664525u + 1013904223u;
        // [-1, 1)
        return (float((state >> 8) & 0xFFFFFF) / 16777216.0f) * 2.0f - 1.0f;
    };
    for (auto& v : w) v = nf();
    for (auto& v : x) v = nf();

    std::vector<float> y_ref;
    m2::gemv_cpu_fp32(w, x, rows, cols, y_ref);

    auto qm = m2::quantize_matrix_per_row_int8(w, rows, cols);
    auto qx = m2::quantize_vector_int8_dynamic(x);
    std::vector<int32_t> y_i32;
    m2::gemv_cpu_int8(qm, qx, y_i32);
    std::vector<float> y_dq;
    m2::dequantize_gemv_output(y_i32, qm.row_scales, qx.scale, y_dq);

    float maxabs = 0.0f, refmax = 0.0f;
    for (int r = 0; r < rows; ++r) {
        maxabs = std::max(maxabs, std::fabs(y_dq[r] - y_ref[r]));
        refmax = std::max(refmax, std::fabs(y_ref[r]));
    }
    // Loose-ish: per-row symmetric int8 of values in [-1,1) gives ~1%
    // relative error in the worst case. Be generous on the threshold.
    EXPECT_TRUE(maxabs <= 0.1f * std::max(1.0f, refmax));
}

void test_saturation_at_127() {
    std::printf("[test] saturation pinned at -127 (not -128)\n");
    // amax == 100; -100/scale rounds toward -128 in naive form but we
    // clamp to -127 to keep symmetry. Confirm the negative bound is at
    // -127, not -128.
    std::vector<float> w = {-100.0f, 99.0f};
    auto qm = m2::quantize_matrix_per_row_int8(w, 1, 2);
    EXPECT_TRUE(qm.data[0] >= -127);
    // Spec: int8 max is +127 by definition; instead test the saturation
    // *floor*: after round, the most-negative entry must never be -128.
    EXPECT_TRUE(qm.data[0] != int8_t(-128));
}

} // anonymous

int main() {
    test_quantize_matrix_range();
    test_zero_row_does_not_divide_by_zero();
    test_zero_vector_does_not_divide_by_zero();
    test_int8_gemv_hand_checked();
    test_dequant_roundtrip_close_to_fp32();
    test_saturation_at_127();
    std::printf("\n[test_quantization] passed=%d failed=%d\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
