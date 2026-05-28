// test_topk_compare.cpp -- exercise the compare_logits helper.

#include <cstdio>
#include <vector>

#include "../src/llama/compare.hpp"

namespace {
int g_pass = 0, g_fail = 0;
#define EXPECT_TRUE(x) do { if (x) {++g_pass;} else { ++g_fail; \
  std::printf("  FAIL [%s:%d] %s\n", __FILE__, __LINE__, #x); } } while (0)
#define EXPECT_EQ(a, b) do { auto _a=(a); auto _b=(b); if (_a==_b) {++g_pass;} \
  else { ++g_fail; std::printf("  FAIL [%s:%d] %s != %s (%lld vs %lld)\n", \
  __FILE__, __LINE__, #a, #b, (long long)_a, (long long)_b); } } while (0)

void test_identical_logits() {
    std::printf("[test] identical logits -> max_abs=0, top1 match, full overlap\n");
    std::vector<float> a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    auto r = m2::compare_logits(a, a, 5);
    EXPECT_EQ(r.max_abs_error, 0.0f);
    EXPECT_EQ(r.mean_abs_error, 0.0f);
    EXPECT_TRUE(r.top1_match);
    EXPECT_EQ(r.topk_overlap, 5);
}

void test_top1_differ() {
    std::printf("[test] top1 differs\n");
    std::vector<float> a = {5, 4, 3, 2, 1};
    std::vector<float> b = {1, 4, 3, 2, 5};
    auto r = m2::compare_logits(a, b, 3);
    EXPECT_TRUE(!r.top1_match);
    EXPECT_TRUE(r.max_abs_error > 0);
}

void test_partial_overlap() {
    std::printf("[test] partial topk overlap\n");
    std::vector<float> a = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    std::vector<float> b = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    auto r = m2::compare_logits(a, b, 5);
    // top5 of a = {0,1,2,3,4}, top5 of b = {9,8,7,6,5}. Overlap = 0.
    EXPECT_EQ(r.topk_overlap, 0);
}

} // anonymous

int main() {
    test_identical_logits();
    test_top1_differ();
    test_partial_overlap();
    std::printf("\n[test_topk_compare] passed=%d failed=%d\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
