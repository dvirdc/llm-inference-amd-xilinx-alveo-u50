// main_hybrid_lmhead.cpp -- runs Llama inference where the LM head is
// offloaded to the FPGA via the Milestone-1 gemv_int8 kernel.
//
// Two supported modes:
//
// 1) Generation mode (default):
//    decode `--steps` tokens; for each token compute the LM head on the
//    FPGA (if --use-fpga-lm-head) and (optionally, with
//    --compare-every-token) also run the CPU FP32 and CPU INT8 LM heads
//    and report differences.
//
// 2) Validation-only mode (--validate-lm-head-only):
//    run the CPU transformer for *one* token of the prompt, then
//    exercise all three LM-head paths and print the full comparison
//    report. Use this to confirm bit-exact CPU-INT8 vs FPGA-INT8 before
//    relying on the hybrid generation path.

#include <chrono>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

#include "../common/arg_parser.hpp"
#include "../common/errors.hpp"
#include "../fpga/fpga_gemv_engine.hpp"
#include "../llama/compare.hpp"
#include "../llama/llama_forward_cpu.hpp"
#include "../llama/llama_model.hpp"
#include "../llama/quantization.hpp"
#include "../llama/sampler.hpp"
#include "../llama/tokenizer.hpp"

namespace {

constexpr const char* kLmHeadName = "lm_head";

// Print the model + run banner up top so logs from different runs are
// distinguishable.
void print_banner(const std::string& xclbin, const std::string& ckpt,
                  const std::string& tokz, const m2::LlamaConfig& c) {
    std::printf("Milestone 2: U50 FPGA LM Head\n"
                "  checkpoint: %s\n"
                "  tokenizer : %s\n"
                "  xclbin    : %s\n",
                ckpt.c_str(), tokz.c_str(), xclbin.c_str());
    m2::print_llama_config(c);
}

// One-shot helper: given the post-final-RMSNorm hidden vector `hidden`,
// compute all three LM-head paths and compare them.
//   * Returns true iff CPU INT8 and FPGA INT8 produced bit-exact int32 outputs.
//   * Fills `fpga_logits_fp32` with the dequantized FPGA result so the
//     sampler can use it.
bool compare_three_paths(const std::vector<float>&        hidden,
                         const m2::QuantizedMatrix&       qw,
                         const m2::LlamaModel&            model,
                         m2::FpgaGemvEngine&              eng,
                         std::vector<float>&              fpga_logits_fp32,
                         m2::LogitsCompareReport&         fp32_vs_fpga,
                         float                            max_abs_warn,
                         bool                             verbose) {
    // CPU FP32 reference
    std::vector<float> y_fp32_ref;
    m2::gemv_cpu_fp32(model.weights.wcls, hidden,
                      model.config.vocab_size, model.config.dim,
                      y_fp32_ref);

    // Quantize activation, run CPU INT8 path and FPGA INT8 path.
    m2::QuantizedVector qx = m2::quantize_vector_int8_dynamic(hidden);

    std::vector<int32_t> y_int32_cpu, y_int32_fpga;
    m2::gemv_cpu_int8(qw, qx, y_int32_cpu);
    eng.run_int8(kLmHeadName, qx, y_int32_fpga);

    int idx = -1, va = 0, vb = 0;
    bool exact = m2::exact_match_int32(y_int32_cpu, y_int32_fpga, &idx, &va, &vb);
    if (!exact) {
        std::fprintf(stderr,
            "  CPU INT8 vs FPGA INT8: MISMATCH at row %d (cpu=%d fpga=%d)\n",
            idx, va, vb);
    } else if (verbose) {
        std::fprintf(stderr, "  CPU INT8 vs FPGA INT8: EXACT MATCH\n");
    }

    // Dequantize FPGA int32 -> fp32 logits so the sampler can use them.
    m2::dequantize_gemv_output(y_int32_fpga, qw.row_scales, qx.scale, fpga_logits_fp32);

    // Compare FP32 reference vs dequantized FPGA output.
    fp32_vs_fpga = m2::compare_logits(y_fp32_ref, fpga_logits_fp32, 5);
    if (verbose) m2::print_compare_report(fp32_vs_fpga, "FP32 vs FPGA dequantized");

    if (fp32_vs_fpga.max_abs_error > max_abs_warn) {
        std::fprintf(stderr,
            "  WARN: FP32 vs FPGA max_abs_error %g exceeds --max-abs-error-warning %g\n",
            fp32_vs_fpga.max_abs_error, max_abs_warn);
    }
    return exact;
}

} // anonymous

int main(int argc, char** argv) try {
    m2::ArgParser p(argv[0]);
    p.add_string("--xclbin",      "path to gemv_int8 xclbin");
    p.add_string("--checkpoint",  "path to llama2.c checkpoint");
    p.add_string("--tokenizer",   "path to tokenizer.bin");
    p.add_string("--prompt",      "prompt text", "Once upon a time");
    p.add_string("--steps",       "tokens to generate", "16");
    p.add_string("--temperature", "0.0 = greedy", "0.0");
    p.add_string("--seed",        "sampler seed", "0");
    p.add_string("--device-id",   "XRT device index", "0");
    p.add_toggle("--use-fpga-lm-head",     "route LM head through the FPGA");
    p.add_toggle("--validate-lm-head-only","one-shot LM-head 3-way compare, no generation");
    p.add_toggle("--compare-every-token",  "run all three paths each token");
    p.add_string("--max-abs-error-warning","FP32-vs-FPGA threshold beyond which we WARN", "1.0");
    p.add_string("--rows", "(validate mode) explicit vocab_size", "0");
    p.add_string("--cols", "(validate mode) explicit dim",        "0");
    p.add_toggle("--verbose", "extra logging");
    p.parse(argc, argv);

    p.require("--xclbin");
    p.require("--checkpoint");
    p.require("--tokenizer");

    const bool   verbose         = p.flag("--verbose");
    const bool   use_fpga        = p.flag("--use-fpga-lm-head");
    const bool   validate_only   = p.flag("--validate-lm-head-only");
    const bool   cmp_every_tok   = p.flag("--compare-every-token");
    const float  max_abs_warn    = p.get_float("--max-abs-error-warning");
    const int    device_index    = p.get_int("--device-id");

    // ---- load model + tokenizer ----
    m2::LlamaModel model = m2::load_llama_model(p.get("--checkpoint"));
    print_banner(p.get("--xclbin"), p.get("--checkpoint"), p.get("--tokenizer"), model.config);

    // Optional shape check against the user-supplied --rows/--cols when
    // they explicitly set them (validate-only flow).
    if (p.get_int("--rows") > 0)
        M2_CHECK(p.get_int("--rows") == model.config.vocab_size, ShapeError,
                 "--rows " << p.get_int("--rows") << " != vocab_size " << model.config.vocab_size);
    if (p.get_int("--cols") > 0)
        M2_CHECK(p.get_int("--cols") == model.config.dim, ShapeError,
                 "--cols " << p.get_int("--cols") << " != dim " << model.config.dim);

    m2::Tokenizer tok(p.get("--tokenizer"), model.config.vocab_size);

    // ---- quantize LM head ----
    std::fprintf(stderr, "\nQuantizing LM head...\n  rows: %d\n  cols: %d\n"
                         "  quantization: per-row int8\n",
                 model.config.vocab_size, model.config.dim);
    m2::QuantizedMatrix qw = m2::quantize_matrix_per_row_int8(
        model.weights.wcls, model.config.vocab_size, model.config.dim);
    std::fprintf(stderr, "  status: OK\n");

    // ---- open FPGA, push LM head weights into HBM ----
    m2::FpgaGemvEngine engine(p.get("--xclbin"), device_index);
    engine.load_weight(kLmHeadName, qw);

    m2::Sampler sampler(model.config.vocab_size,
                        p.get_float("--temperature"),
                        p.get_uint("--seed"));
    m2::LlamaRunState state(model.config);

    // ---- validate-only mode ----------------------------------------------
    if (validate_only) {
        // Run the CPU transformer for the very first token of the prompt
        // (BOS). That's enough to populate state.x with a meaningful
        // hidden vector for the LM-head comparison.
        std::vector<int> prompt_tokens = tok.encode(p.get("--prompt"), true, false);
        int token = prompt_tokens.empty() ? 1 : prompt_tokens[0];
        m2::llama_forward_cpu_until_lm_head(model, state, token, /*pos*/0);

        std::vector<float> fpga_logits;
        m2::LogitsCompareReport rep;
        bool exact = compare_three_paths(state.x, qw, model, engine,
                                         fpga_logits, rep, max_abs_warn, /*verbose*/true);

        std::printf("\nResult: %s\n", exact ? "PASS" : "FAIL");
        return exact ? 0 : 1;
    }

    // ---- generation mode --------------------------------------------------
    std::vector<int> prompt_tokens = tok.encode(p.get("--prompt"), true, false);
    if (prompt_tokens.empty()) {
        std::fprintf(stderr, "empty prompt after tokenization\n");
        return 2;
    }
    int token   = prompt_tokens[0];
    int next    = 0;
    int max_pos = std::min(p.get_int("--steps"), model.config.seq_len - 1);

    auto t0 = std::chrono::steady_clock::now();
    int exact_count = 0, total_count = 0;
    m2::LogitsCompareReport last_report;

    std::printf("\n");
    for (int pos = 0; pos < max_pos; ++pos) {
        // Run the CPU transformer all the way through. This populates:
        //   state.x      -- hidden vector going into the LM head
        //   state.logits -- CPU FP32 LM-head output (used as the reference)
        m2::llama_forward_cpu(model, state, token, pos);

        // Optionally replace state.logits with the FPGA-dequantized output.
        if (use_fpga) {
            std::vector<float> fpga_logits;
            if (cmp_every_tok) {
                bool e = compare_three_paths(state.x, qw, model, engine,
                                             fpga_logits, last_report,
                                             max_abs_warn, verbose);
                if (e) ++exact_count;
                ++total_count;
            } else {
                // Fast path: only quantize, run FPGA, dequant.
                m2::QuantizedVector qx = m2::quantize_vector_int8_dynamic(state.x);
                engine.run_dequantized(kLmHeadName, qx, fpga_logits);
            }
            state.logits.swap(fpga_logits);
        }

        if (pos + 1 < static_cast<int>(prompt_tokens.size())) {
            next = prompt_tokens[pos + 1]; // still in prompt
        } else {
            next = sampler.sample(state.logits);
        }
        std::string piece = tok.decode(token, next);
        std::printf("%s", piece.c_str()); std::fflush(stdout);
        token = next;
        if (token == 1) break;
    }
    std::printf("\n");

    auto t1 = std::chrono::steady_clock::now();
    double gen_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::fprintf(stderr,
        "\n[timing] generate: %.1f ms   tokens: %d   tok/s: %.1f\n",
        gen_ms, max_pos, gen_ms > 0 ? max_pos * 1000.0 / gen_ms : 0.0);

    if (cmp_every_tok && total_count > 0) {
        std::printf("\nLM-head bit-exact tokens: %d / %d\n", exact_count, total_count);
        std::printf("Result: %s\n", (exact_count == total_count) ? "PASS" : "FAIL");
        return (exact_count == total_count) ? 0 : 1;
    }
    return 0;
}
catch (const std::exception& e) {
    std::fprintf(stderr, "fatal: %s\n", e.what());
    return 1;
}
