// inspect_model.cpp -- tiny utility that loads a Karpathy llama2.c
// checkpoint and prints its config plus tensor sizes. Useful for
// verifying you've placed the right file before kicking off a long FPGA
// build. No FPGA / XRT needed.

#include <cstdio>
#include <exception>

#include "../common/arg_parser.hpp"
#include "../llama/llama_model.hpp"

int main(int argc, char** argv) try {
    m2::ArgParser p(argv[0]);
    p.add_string("--checkpoint", "path to Karpathy llama2.c .bin file");
    p.parse(argc, argv);
    p.require("--checkpoint");

    m2::LlamaModel m = m2::load_llama_model(p.get("--checkpoint"));
    m2::print_llama_config(m.config);

    auto sz = [](const std::vector<float>& v) { return v.size(); };
    std::fprintf(stderr,
        "\nTensor sizes (float count):\n"
        "  token_embedding_table : %zu\n"
        "  wq                    : %zu\n"
        "  wk                    : %zu\n"
        "  wv                    : %zu\n"
        "  wo                    : %zu\n"
        "  w1                    : %zu\n"
        "  w2                    : %zu\n"
        "  w3                    : %zu\n"
        "  rms_att_weight        : %zu\n"
        "  rms_ffn_weight        : %zu\n"
        "  rms_final_weight      : %zu\n"
        "  wcls                  : %zu\n",
        sz(m.weights.token_embedding_table),
        sz(m.weights.wq), sz(m.weights.wk), sz(m.weights.wv), sz(m.weights.wo),
        sz(m.weights.w1), sz(m.weights.w2), sz(m.weights.w3),
        sz(m.weights.rms_att_weight), sz(m.weights.rms_ffn_weight),
        sz(m.weights.rms_final_weight), sz(m.weights.wcls));
    return 0;
}
catch (const std::exception& e) {
    std::fprintf(stderr, "fatal: %s\n", e.what());
    return 1;
}
