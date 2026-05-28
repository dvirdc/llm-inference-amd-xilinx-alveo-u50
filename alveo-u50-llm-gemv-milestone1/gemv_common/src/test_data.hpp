// test_data.hpp -- deterministic INT8 test data generation.
//
// We intentionally restrict the random range to [-8, +7] by default so the
// CPU reference values stay small and easy to spot-check by eye, and so we
// have plenty of int32 headroom (see analysis in gemv_cpu.cpp).

#pragma once

#include <cstdint>
#include <vector>

namespace gemv {

// Default bounds for synthetic INT8 data. Tight enough to be obviously safe
// for int32 accumulation at all milestone-1 sizes, wide enough to exercise
// sign handling on the kernel side.
constexpr int8_t kDefaultMin = -8;
constexpr int8_t kDefaultMax =  7;

// Fill `out` with deterministic int8 values in [vmin, vmax] using a 32-bit
// LCG seeded by `seed`. Same (seed, size, vmin, vmax) tuple => same bytes.
//
// We roll our own RNG instead of std::mt19937 because the std distributions
// are implementation-defined across libstdc++/libc++, and we want bit-exact
// repeatability for cross-machine debugging.
void fill_random_int8(std::vector<int8_t>& out,
                      size_t               n,
                      uint32_t             seed,
                      int8_t               vmin = kDefaultMin,
                      int8_t               vmax = kDefaultMax);

// Pretty-print up to `max_show` elements of a vector (head only). Used when
// --verbose is on so a human can sanity-check small problem sizes.
void print_vec_int8(const char* label, const std::vector<int8_t>& v, size_t max_show = 16);
void print_vec_i32 (const char* label, const std::vector<int32_t>& v, size_t max_show = 16);

} // namespace gemv
