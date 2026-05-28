// test_gemv_cpu.cpp -- standalone unit test for the CPU reference.
//
// Builds a single executable (no test framework) and returns 0 on PASS,
// 1 on FAIL. Easy to run from `make test`.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include "gemv_cpu.hpp"
#include "test_data.hpp"

namespace {

int  g_pass = 0;
int  g_fail = 0;

#define EXPECT_EQ(a, b) do {                                                  \
    auto _a = (a); auto _b = (b);                                             \
    if (_a == _b) { ++g_pass; }                                               \
    else {                                                                    \
        ++g_fail;                                                             \
        std::printf("  FAIL [%s:%d] %s != %s  (lhs=%lld rhs=%lld)\n",         \
                    __FILE__, __LINE__, #a, #b,                               \
                    static_cast<long long>(_a),                               \
                    static_cast<long long>(_b));                              \
    }                                                                         \
} while (0)

#define EXPECT_TRUE(x) do {                                                   \
    if (x) { ++g_pass; }                                                      \
    else { ++g_fail; std::printf("  FAIL [%s:%d] %s\n", __FILE__, __LINE__, #x); } \
} while (0)

// ---------------------------------------------------------------------------
// Test 1: hand-checked tiny GEMV.
//   weights = [[ 1, 2, 3],
//              [-1, 0, 4],
//              [ 5, 5, 5]]
//   x       = [ 2, -1, 3]
//   expected y = [ 2*1 + -1*2 + 3*3,     = 9
//                 -1*2 +  0*-1 + 4*3,    = 10
//                  5*2 + -1*5 +  3*5 ]   = 20
// ---------------------------------------------------------------------------
void test_tiny_handworked() {
    std::printf("[test] tiny hand-worked GEMV\n");
    std::vector<int8_t>  w = { 1, 2, 3,
                              -1, 0, 4,
                               5, 5, 5 };
    std::vector<int8_t>  x = { 2, -1, 3 };
    std::vector<int32_t> y;
    gemv::gemv_cpu_ref(w, x, y, 3, 3);
    EXPECT_EQ(y.size(), size_t{3});
    EXPECT_EQ(y[0],  9);
    EXPECT_EQ(y[1], 10);
    EXPECT_EQ(y[2], 20);
}

// ---------------------------------------------------------------------------
// Test 2: deterministic random GEMV reproduces an independently computed
// reference. We compute the reference two ways (via the function under test
// and via a hand-written nested loop here) and compare.
// ---------------------------------------------------------------------------
void test_deterministic_random(int rows, int cols, uint32_t seed) {
    std::printf("[test] deterministic random GEMV  rows=%d cols=%d seed=0x%X\n",
                rows, cols, seed);
    std::vector<int8_t> w, x;
    gemv::fill_random_int8(w, static_cast<size_t>(rows) * cols, seed ^ 0xA);
    gemv::fill_random_int8(x, static_cast<size_t>(cols),         seed ^ 0xB);

    std::vector<int32_t> y_under_test;
    gemv::gemv_cpu_ref(w, x, y_under_test, rows, cols);

    // Independent in-test computation.
    std::vector<int32_t> y_expected(rows, 0);
    for (int r = 0; r < rows; ++r) {
        int32_t acc = 0;
        for (int c = 0; c < cols; ++c) {
            acc += static_cast<int32_t>(w[size_t(r) * cols + c])
                 * static_cast<int32_t>(x[c]);
        }
        y_expected[r] = acc;
    }

    bool all_eq = true;
    for (int r = 0; r < rows; ++r) {
        if (y_under_test[r] != y_expected[r]) { all_eq = false; break; }
    }
    EXPECT_TRUE(all_eq);
}

// ---------------------------------------------------------------------------
// Test 3: invalid args throw.
// ---------------------------------------------------------------------------
void test_invalid_args_throw() {
    std::printf("[test] invalid args throw\n");
    std::vector<int8_t>  w(4, 1), x(2, 1);
    std::vector<int32_t> y;
    bool threw = false;
    try { gemv::gemv_cpu_ref(w, x, y, 0, 2); } catch (const std::exception&) { threw = true; }
    EXPECT_TRUE(threw);

    threw = false;
    try { gemv::gemv_cpu_ref(w, x, y, 2, 3); } catch (const std::exception&) { threw = true; } // wrong cols vs w size
    EXPECT_TRUE(threw);
}

} // namespace

int main() {
    test_tiny_handworked();
    test_deterministic_random(8,  8,    0xDEADBEEF);
    test_deterministic_random(64, 128,  0xC0FFEE);
    test_deterministic_random(1024, 64, 0x1234);
    test_invalid_args_throw();

    std::printf("\n[test_gemv_cpu] passed=%d  failed=%d\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
