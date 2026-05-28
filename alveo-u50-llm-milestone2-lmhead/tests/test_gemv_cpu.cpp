// test_gemv_cpu.cpp -- ported from Milestone 1, validates the int8 CPU
// GEMV (which is also our bit-exact reference for the FPGA kernel).

#include <cstdint>
#include <cstdio>
#include <vector>

#include "../src/llama/quantization.hpp"

namespace {
int g_pass = 0, g_fail = 0;
#define EXPECT_EQ(a, b) do { auto _a=(a); auto _b=(b); if (_a==_b) {++g_pass;} \
  else { ++g_fail; std::printf("  FAIL [%s:%d] %s != %s (%lld vs %lld)\n", \
  __FILE__, __LINE__, #a, #b, (long long)_a, (long long)_b); } } while (0)
#define EXPECT_TRUE(x) do { if (x) {++g_pass;} else { ++g_fail; \
  std::printf("  FAIL [%s:%d] %s\n", __FILE__, __LINE__, #x); } } while (0)

void test_tiny() {
    std::printf("[test] tiny hand-worked GEMV (int8)\n");
    m2::QuantizedMatrix qm;
    qm.rows = 3; qm.cols = 3;
    qm.data = { 1,  2, 3,
               -1,  0, 4,
                5,  5, 5 };
    qm.row_scales = {1, 1, 1};
    m2::QuantizedVector qx;
    qx.size = 3; qx.data = {2, -1, 3}; qx.scale = 1.0f;

    std::vector<int32_t> y;
    m2::gemv_cpu_int8(qm, qx, y);
    EXPECT_EQ(y[0],  9);
    EXPECT_EQ(y[1], 10);
    EXPECT_EQ(y[2], 20);
}

void test_random_matches_independent_computation(int rows, int cols, unsigned seed) {
    std::printf("[test] random GEMV rows=%d cols=%d seed=0x%X\n", rows, cols, seed);
    std::vector<int8_t> w(rows*cols), x(cols);
    unsigned s = seed ? seed : 1;
    auto next = [&](){ s = s*1664525u + 1013904223u; return s; };
    for (auto& v : w) v = int8_t((next() >> 16) % 17 - 8); // [-8, 8]
    for (auto& v : x) v = int8_t((next() >> 16) % 17 - 8);

    m2::QuantizedMatrix qm; qm.rows=rows; qm.cols=cols; qm.data=w;
    qm.row_scales.assign(rows, 1.0f);
    m2::QuantizedVector qx; qx.size=cols; qx.data=x; qx.scale=1.0f;

    std::vector<int32_t> y_under_test;
    m2::gemv_cpu_int8(qm, qx, y_under_test);

    std::vector<int32_t> y_ref(rows, 0);
    for (int r = 0; r < rows; ++r) {
        int32_t acc = 0;
        for (int c = 0; c < cols; ++c)
            acc += int32_t(w[size_t(r)*cols + c]) * int32_t(x[c]);
        y_ref[r] = acc;
    }
    bool eq = (y_under_test == y_ref);
    EXPECT_TRUE(eq);
}

} // anonymous

int main() {
    test_tiny();
    test_random_matches_independent_computation(8, 8, 0xDEADBEEF);
    test_random_matches_independent_computation(64, 128, 0xC0FFEE);
    test_random_matches_independent_computation(1024, 64, 0x1234);
    std::printf("\n[test_gemv_cpu] passed=%d failed=%d\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
