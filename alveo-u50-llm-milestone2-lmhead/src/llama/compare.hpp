// compare.hpp -- helpers for the three-way comparison FP32 / CPU INT8 /
// FPGA INT8. See docs/quantization.md for the rules.

#pragma once

#include <cstdint>
#include <vector>

namespace m2 {

struct LogitsCompareReport {
    float max_abs_error  = 0.0f;
    float mean_abs_error = 0.0f;
    int   top1_a         = -1;
    int   top1_b         = -1;
    bool  top1_match     = false;
    int   topk_overlap   = 0;   // |topK(a) ∩ topK(b)|
    int   k              = 5;
};

// Compare two FP32 logit vectors. Returns max/mean abs error, plus top1
// match and topk overlap (k=5 by default).
LogitsCompareReport compare_logits(const std::vector<float>& a,
                                   const std::vector<float>& b,
                                   int k = 5);

// Strict equality between two INT32 vectors -- the canonical "CPU INT8
// path matches FPGA INT8 path" check. Returns true on exact match, and
// when false also reports the index/values of the first mismatch.
bool exact_match_int32(const std::vector<int32_t>& a,
                       const std::vector<int32_t>& b,
                       int* first_diff_idx       = nullptr,
                       int32_t* first_diff_a_out = nullptr,
                       int32_t* first_diff_b_out = nullptr);

// Pretty-print a LogitsCompareReport to stderr.
void print_compare_report(const LogitsCompareReport& r, const char* label);

} // namespace m2
