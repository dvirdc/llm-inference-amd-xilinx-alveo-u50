#include "test_data.hpp"

#include <cstdio>
#include <stdexcept>

namespace gemv {

namespace {

// Numerical Recipes LCG. Cheap, deterministic, good enough for filling
// test buffers with int8 noise. NOT for cryptography.
struct Lcg32 {
    uint32_t state;
    explicit Lcg32(uint32_t seed) : state(seed ? seed : 0x9E3779B9u) {}
    uint32_t next() {
        state = state * 1664525u + 1013904223u;
        return state;
    }
};

} // namespace

void fill_random_int8(std::vector<int8_t>& out,
                      size_t               n,
                      uint32_t             seed,
                      int8_t               vmin,
                      int8_t               vmax) {
    if (vmin > vmax) {
        throw std::invalid_argument("fill_random_int8: vmin > vmax");
    }
    out.resize(n);
    const int32_t span = static_cast<int32_t>(vmax) - static_cast<int32_t>(vmin) + 1;
    Lcg32 rng(seed);
    for (size_t i = 0; i < n; ++i) {
        // Take the high bits of the LCG output (they mix better) and map
        // them into [vmin, vmax].
        uint32_t r = rng.next() >> 16;
        out[i] = static_cast<int8_t>(vmin + static_cast<int32_t>(r % static_cast<uint32_t>(span)));
    }
}

void print_vec_int8(const char* label, const std::vector<int8_t>& v, size_t max_show) {
    const size_t n = v.size() < max_show ? v.size() : max_show;
    std::printf("%s [n=%zu]: ", label, v.size());
    for (size_t i = 0; i < n; ++i) std::printf("%4d ", static_cast<int>(v[i]));
    if (v.size() > n) std::printf("... ");
    std::printf("\n");
}

void print_vec_i32(const char* label, const std::vector<int32_t>& v, size_t max_show) {
    const size_t n = v.size() < max_show ? v.size() : max_show;
    std::printf("%s [n=%zu]: ", label, v.size());
    for (size_t i = 0; i < n; ++i) std::printf("%8d ", v[i]);
    if (v.size() > n) std::printf("... ");
    std::printf("\n");
}

} // namespace gemv
