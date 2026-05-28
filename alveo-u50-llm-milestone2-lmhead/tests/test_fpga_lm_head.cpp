// test_fpga_lm_head.cpp -- end-to-end FPGA test. Requires xclbin + model
// + tokenizer paths via environment variables; skips cleanly otherwise.
//
// Required env:
//   TEST_XCLBIN
//   TEST_LLAMA_CHECKPOINT
//   TEST_LLAMA_TOKENIZER

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

#include "../src/fpga/fpga_gemv_engine.hpp"
#include "../src/llama/compare.hpp"
#include "../src/llama/llama_forward_cpu.hpp"
#include "../src/llama/llama_model.hpp"
#include "../src/llama/quantization.hpp"
#include "../src/llama/tokenizer.hpp"

namespace {
int g_pass = 0, g_fail = 0;
#define EXPECT_TRUE(x) do { if (x) {++g_pass;} else { ++g_fail; \
  std::printf("  FAIL [%s:%d] %s\n", __FILE__, __LINE__, #x); } } while (0)

const char* env_or_null(const char* k) {
    const char* v = std::getenv(k);
    return (v && *v) ? v : nullptr;
}
} // anonymous

int main() {
    const char* xclbin = env_or_null("TEST_XCLBIN");
    const char* ckpt   = env_or_null("TEST_LLAMA_CHECKPOINT");
    const char* tokp   = env_or_null("TEST_LLAMA_TOKENIZER");
    if (!xclbin || !ckpt || !tokp) {
        std::printf("[test_fpga_lm_head] SKIPPED (set TEST_XCLBIN, "
                    "TEST_LLAMA_CHECKPOINT, TEST_LLAMA_TOKENIZER to run)\n");
        return 0;
    }
    try {
        auto model = m2::load_llama_model(ckpt);
        m2::Tokenizer tok(tokp, model.config.vocab_size);

        auto qw = m2::quantize_matrix_per_row_int8(
            model.weights.wcls, model.config.vocab_size, model.config.dim);

        m2::FpgaGemvEngine eng(xclbin, 0);
        eng.load_weight("lm_head", qw);

        // Run one transformer step to get a real hidden vector.
        m2::LlamaRunState state(model.config);
        auto tokens = tok.encode("Once upon a time", true, false);
        int t = tokens.empty() ? 1 : tokens[0];
        m2::llama_forward_cpu_until_lm_head(model, state, t, 0);

        auto qx = m2::quantize_vector_int8_dynamic(state.x);

        std::vector<int32_t> y_cpu, y_fpga;
        m2::gemv_cpu_int8(qw, qx, y_cpu);
        eng.run_int8("lm_head", qx, y_fpga);

        int idx = -1, va = 0, vb = 0;
        bool exact = m2::exact_match_int32(y_cpu, y_fpga, &idx, &va, &vb);
        if (!exact) {
            std::printf("  first mismatch row=%d cpu=%d fpga=%d\n", idx, va, vb);
        }
        EXPECT_TRUE(exact);

        std::vector<float> y_fp32_ref, y_fp32_fpga;
        m2::gemv_cpu_fp32(model.weights.wcls, state.x,
                          model.config.vocab_size, model.config.dim, y_fp32_ref);
        m2::dequantize_gemv_output(y_fpga, qw.row_scales, qx.scale, y_fp32_fpga);
        auto rep = m2::compare_logits(y_fp32_ref, y_fp32_fpga, 5);
        m2::print_compare_report(rep, "FP32 vs FPGA dequantized");
    } catch (const std::exception& e) {
        std::printf("  threw: %s\n", e.what());
        ++g_fail;
    }
    std::printf("\n[test_fpga_lm_head] passed=%d failed=%d\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
