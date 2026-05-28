// test_data_generation.cpp -- confirm fill_random_int8 is deterministic and
// stays within the requested range.

#include <cstdint>
#include <cstdio>
#include <vector>

#include "test_data.hpp"

namespace {

int g_pass = 0, g_fail = 0;

#define EXPECT_TRUE(x) do {                                                   \
    if (x) { ++g_pass; }                                                      \
    else { ++g_fail; std::printf("  FAIL [%s:%d] %s\n", __FILE__, __LINE__, #x); } \
} while (0)

void test_same_seed_same_bytes() {
    std::printf("[test] same seed -> same bytes\n");
    std::vector<int8_t> a, b;
    gemv::fill_random_int8(a, 4096, 0xC0FFEE);
    gemv::fill_random_int8(b, 4096, 0xC0FFEE);
    EXPECT_TRUE(a.size() == b.size());
    bool eq = (a == b);
    EXPECT_TRUE(eq);
}

void test_different_seed_different_bytes() {
    std::printf("[test] different seed -> different bytes\n");
    std::vector<int8_t> a, b;
    gemv::fill_random_int8(a, 4096, 1);
    gemv::fill_random_int8(b, 4096, 2);
    bool diff = (a != b);
    EXPECT_TRUE(diff);
}

void test_default_range() {
    std::printf("[test] default range [-8, 7]\n");
    std::vector<int8_t> v;
    gemv::fill_random_int8(v, 100000, 42);
    bool in_range = true;
    for (auto b : v) {
        if (b < gemv::kDefaultMin || b > gemv::kDefaultMax) { in_range = false; break; }
    }
    EXPECT_TRUE(in_range);
}

void test_custom_range() {
    std::printf("[test] custom range [-1, 1]\n");
    std::vector<int8_t> v;
    gemv::fill_random_int8(v, 100000, 7, -1, 1);
    bool in_range = true;
    bool saw_lo = false, saw_hi = false, saw_zero = false;
    for (auto b : v) {
        if (b < -1 || b > 1) { in_range = false; break; }
        if (b == -1) saw_lo = true;
        if (b ==  1) saw_hi = true;
        if (b ==  0) saw_zero = true;
    }
    EXPECT_TRUE(in_range);
    EXPECT_TRUE(saw_lo);
    EXPECT_TRUE(saw_hi);
    EXPECT_TRUE(saw_zero);
}

void test_full_int8_range() {
    std::printf("[test] full int8 range [-128, 127] works\n");
    std::vector<int8_t> v;
    gemv::fill_random_int8(v, 1000, 9, -128, 127);
    // Any int8 value is by definition within [-128, 127]; this test exists
    // to confirm fill_random_int8 doesn't reject or truncate at the
    // boundary. We check that we see at least one negative and one
    // positive value (i.e. sign handling works across the full range).
    bool saw_neg = false, saw_pos = false;
    for (auto b : v) {
        if (b < 0) saw_neg = true;
        if (b > 0) saw_pos = true;
    }
    EXPECT_TRUE(saw_neg);
    EXPECT_TRUE(saw_pos);
}

} // namespace

int main() {
    test_same_seed_same_bytes();
    test_different_seed_different_bytes();
    test_default_range();
    test_custom_range();
    test_full_int8_range();

    std::printf("\n[test_data_generation] passed=%d  failed=%d\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
