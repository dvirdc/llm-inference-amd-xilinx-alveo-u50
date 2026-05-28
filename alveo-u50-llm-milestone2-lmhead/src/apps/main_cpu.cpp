// main_cpu.cpp -- pure CPU Llama inference. Does NOT touch XRT. Useful
// as a sanity check that:
//   * the model loader works
//   * the tokenizer works
//   * the CPU transformer forward is correctly ported from Karpathy
//
// Once this produces sensible text against stories15M.bin, main_hybrid_lmhead
// should produce token-for-token identical output (when --temperature 0
// and --use-fpga-lm-head are NOT set, or when the FPGA path is bit-exact).

#include <chrono>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

#include "../common/arg_parser.hpp"
#include "../llama/llama_forward_cpu.hpp"
#include "../llama/llama_model.hpp"
#include "../llama/sampler.hpp"
#include "../llama/tokenizer.hpp"

int main(int argc, char** argv) try {
    m2::ArgParser p(argv[0]);
    p.add_string("--checkpoint",  "path to .bin checkpoint");
    p.add_string("--tokenizer",   "path to tokenizer.bin");
    p.add_string("--prompt",      "prompt text", "Once upon a time");
    p.add_string("--steps",       "number of tokens to generate", "16");
    p.add_string("--temperature", "0.0 = greedy", "0.0");
    p.add_string("--seed",        "sampler seed", "0");
    p.add_toggle("--verbose",     "extra logging");
    p.parse(argc, argv);
    p.require("--checkpoint");
    p.require("--tokenizer");

    const std::string prompt = p.get("--prompt");
    const int   steps        = p.get_int("--steps");
    const float temperature  = p.get_float("--temperature");
    const unsigned seed      = p.get_uint("--seed");
    const bool verbose       = p.flag("--verbose");

    auto t0 = std::chrono::steady_clock::now();

    m2::LlamaModel model = m2::load_llama_model(p.get("--checkpoint"));
    if (verbose) m2::print_llama_config(model.config);

    m2::Tokenizer tok(p.get("--tokenizer"), model.config.vocab_size);

    std::vector<int> prompt_tokens = tok.encode(prompt, /*bos*/ true, /*eos*/ false);
    if (prompt_tokens.empty()) {
        std::fprintf(stderr, "empty prompt after tokenization\n");
        return 2;
    }
    m2::Sampler sampler(model.config.vocab_size, temperature, seed);
    m2::LlamaRunState state(model.config);

    auto t_load = std::chrono::steady_clock::now();

    // Decode loop. For pos=0 we feed the BOS, pos=1 the next prompt token,
    // and so on until prompt_tokens runs out; from then on we feed the
    // sampler's previous output (autoregressive generation).
    int token    = prompt_tokens[0];
    int next     = 0;
    int prev     = 0;
    int max_pos  = std::min(steps, model.config.seq_len - 1);

    std::printf("\n");
    for (int pos = 0; pos < max_pos; ++pos) {
        m2::llama_forward_cpu(model, state, token, pos);

        if (pos + 1 < static_cast<int>(prompt_tokens.size())) {
            next = prompt_tokens[pos + 1];   // still inside the prompt
        } else {
            next = sampler.sample(state.logits);
        }

        std::string piece = tok.decode(token, next);
        std::printf("%s", piece.c_str());
        std::fflush(stdout);

        prev  = token;
        token = next;
        (void)prev;

        if (token == 1) break; // BOS again -> stop (Karpathy convention)
    }
    std::printf("\n");

    auto t_end = std::chrono::steady_clock::now();
    if (verbose) {
        double load_ms = std::chrono::duration<double, std::milli>(t_load - t0).count();
        double gen_ms  = std::chrono::duration<double, std::milli>(t_end - t_load).count();
        std::fprintf(stderr, "\n[timing] load: %.1f ms   generate: %.1f ms   tok/s: %.1f\n",
                     load_ms, gen_ms,
                     gen_ms > 0 ? max_pos * 1000.0 / gen_ms : 0.0);
    }
    return 0;
}
catch (const std::exception& e) {
    std::fprintf(stderr, "fatal: %s\n", e.what());
    return 1;
}
