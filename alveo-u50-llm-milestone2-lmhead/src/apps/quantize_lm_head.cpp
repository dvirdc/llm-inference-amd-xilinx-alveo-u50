// quantize_lm_head.cpp -- standalone utility: load a llama2.c checkpoint,
// quantize the LM head (wcls) to per-row INT8, and write the int8 bytes
// + row_scales to two files.
//
// Useful for two things:
//   (a) Debugging: inspect the quantized weights without rebuilding the
//       inference binary.
//   (b) Future milestones where xclbin packaging may want to ingest the
//       weights at build time rather than at runtime.
//
// Output files:
//   <out_prefix>.int8.bin   -- raw int8 bytes (vocab*dim, row-major)
//   <out_prefix>.scales.bin -- vocab float32 row scales
// Both can be re-read with plain fread.

#include <cstdint>
#include <cstdio>
#include <exception>
#include <fstream>
#include <string>

#include "../common/arg_parser.hpp"
#include "../common/errors.hpp"
#include "../llama/llama_model.hpp"
#include "../llama/quantization.hpp"

namespace {

void write_bytes(const std::string& path, const void* data, size_t bytes) {
    std::ofstream f(path, std::ios::binary);
    if (!f) M2_THROW(IoError, "cannot open output: " << path);
    f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    if (!f) M2_THROW(IoError, "short write: " << path);
}

} // anonymous

int main(int argc, char** argv) try {
    m2::ArgParser p(argv[0]);
    p.add_string("--checkpoint",  "input llama2.c .bin");
    p.add_string("--out-prefix",  "output prefix (writes <p>.int8.bin and <p>.scales.bin)",
                                  "build/lm_head");
    p.parse(argc, argv);
    p.require("--checkpoint");

    m2::LlamaModel m  = m2::load_llama_model(p.get("--checkpoint"));
    m2::print_llama_config(m.config);

    m2::QuantizedMatrix qm = m2::quantize_matrix_per_row_int8(
        m.weights.wcls, m.config.vocab_size, m.config.dim);

    const std::string prefix = p.get("--out-prefix");
    write_bytes(prefix + ".int8.bin",   qm.data.data(),       qm.data.size());
    write_bytes(prefix + ".scales.bin", qm.row_scales.data(), qm.row_scales.size() * sizeof(float));

    std::fprintf(stderr,
                 "Wrote:\n"
                 "  %s.int8.bin    (%zu bytes)\n"
                 "  %s.scales.bin  (%zu bytes)\n",
                 prefix.c_str(), qm.data.size(),
                 prefix.c_str(), qm.row_scales.size() * sizeof(float));
    return 0;
}
catch (const std::exception& e) {
    std::fprintf(stderr, "fatal: %s\n", e.what());
    return 1;
}
