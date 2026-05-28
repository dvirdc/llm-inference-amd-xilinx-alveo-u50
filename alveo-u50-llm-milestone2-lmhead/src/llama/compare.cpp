#include "compare.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstddef>

#include "../common/errors.hpp"

namespace m2 {

namespace {

// Indices of the k largest elements (descending by value). N small ~32k,
// k small ~5 -- partial_sort is overkill but trivially correct.
std::vector<int> topk_indices(const std::vector<float>& v, int k) {
    std::vector<int> idx(v.size());
    for (size_t i = 0; i < v.size(); ++i) idx[i] = static_cast<int>(i);
    if (k > static_cast<int>(idx.size())) k = static_cast<int>(idx.size());
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b) { return v[a] > v[b]; });
    idx.resize(k);
    return idx;
}

} // anonymous

LogitsCompareReport compare_logits(const std::vector<float>& a,
                                   const std::vector<float>& b,
                                   int k) {
    M2_CHECK(a.size() == b.size(), ShapeError,
             "compare_logits: " << a.size() << " vs " << b.size());
    LogitsCompareReport r;
    r.k = k;

    double sum_abs = 0.0;
    float maxabs = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float d = std::fabs(a[i] - b[i]);
        if (d > maxabs) maxabs = d;
        sum_abs += d;
    }
    r.max_abs_error  = maxabs;
    r.mean_abs_error = a.empty() ? 0.0f : float(sum_abs / double(a.size()));

    auto topa = topk_indices(a, k);
    auto topb = topk_indices(b, k);
    r.top1_a = topa.empty() ? -1 : topa[0];
    r.top1_b = topb.empty() ? -1 : topb[0];
    r.top1_match = (r.top1_a == r.top1_b);

    int overlap = 0;
    for (int x : topa) {
        if (std::find(topb.begin(), topb.end(), x) != topb.end()) ++overlap;
    }
    r.topk_overlap = overlap;
    return r;
}

bool exact_match_int32(const std::vector<int32_t>& a,
                       const std::vector<int32_t>& b,
                       int* first_diff_idx,
                       int32_t* first_diff_a_out,
                       int32_t* first_diff_b_out) {
    M2_CHECK(a.size() == b.size(), ShapeError,
             "exact_match_int32: " << a.size() << " vs " << b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            if (first_diff_idx)   *first_diff_idx   = int(i);
            if (first_diff_a_out) *first_diff_a_out = a[i];
            if (first_diff_b_out) *first_diff_b_out = b[i];
            return false;
        }
    }
    return true;
}

void print_compare_report(const LogitsCompareReport& r, const char* label) {
    std::fprintf(stderr,
        "  %s:\n"
        "    max_abs_error  : %g\n"
        "    mean_abs_error : %g\n"
        "    top1 a/b       : %d / %d  (%s)\n"
        "    top%d overlap   : %d / %d\n",
        label,
        r.max_abs_error, r.mean_abs_error,
        r.top1_a, r.top1_b, r.top1_match ? "MATCH" : "DIFFER",
        r.k, r.topk_overlap, r.k);
}

} // namespace m2
