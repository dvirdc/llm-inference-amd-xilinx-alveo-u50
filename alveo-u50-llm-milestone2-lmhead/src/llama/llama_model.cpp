#include "llama_model.hpp"

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "../common/errors.hpp"
#include "../common/logging.hpp"

namespace m2 {

namespace {

// Read `n` floats. Throws if the stream is short -- never returns "partial".
void fread_f32(std::FILE* f, std::vector<float>& dst, size_t n, const char* tensor_name) {
    dst.resize(n);
    size_t got = std::fread(dst.data(), sizeof(float), n, f);
    if (got != n) {
        M2_THROW(FormatError,
                 "short read on tensor '" << tensor_name
                 << "' (expected " << n << " floats, got " << got << ")");
    }
}

// Loose sanity bounds. If a header field is way outside these we almost
// certainly read garbage (wrong endianness, wrong file, etc.).
bool plausible_dim   (int v) { return v > 0 && v <= 65536; }
bool plausible_layers(int v) { return v > 0 && v <= 1024;  }
bool plausible_heads (int v) { return v > 0 && v <= 1024;  }

} // anonymous

void print_llama_config(const LlamaConfig& c) {
    std::fprintf(stderr,
        "Model config:\n"
        "  dim         : %d\n"
        "  hidden_dim  : %d\n"
        "  n_layers    : %d\n"
        "  n_heads     : %d\n"
        "  n_kv_heads  : %d\n"
        "  vocab_size  : %d\n"
        "  seq_len     : %d\n"
        "  shared_cls  : %s\n",
        c.dim, c.hidden_dim, c.n_layers, c.n_heads, c.n_kv_heads,
        c.vocab_size, c.seq_len, c.shared_classifier ? "yes" : "no");
}

LlamaModel load_llama_model(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        M2_THROW(IoError, "cannot open checkpoint: " << path);
    }

    // ---------- header --------------------------------------------------
    // Seven signed int32s. `vocab_size` field is signed -- the sign carries
    // the "is the LM head tied to the embedding?" bit (positive == tied).
    int32_t header[7];
    if (std::fread(header, sizeof(int32_t), 7, f) != 7) {
        std::fclose(f);
        M2_THROW(FormatError, "checkpoint header too short: " << path);
    }

    LlamaConfig c{};
    c.dim         = header[0];
    c.hidden_dim  = header[1];
    c.n_layers    = header[2];
    c.n_heads     = header[3];
    c.n_kv_heads  = header[4];
    int raw_vocab = header[5];
    c.seq_len     = header[6];
    c.shared_classifier = (raw_vocab > 0);
    c.vocab_size  = std::abs(raw_vocab);

    if (!plausible_dim(c.dim) || !plausible_dim(c.hidden_dim) ||
        !plausible_layers(c.n_layers) ||
        !plausible_heads(c.n_heads) || !plausible_heads(c.n_kv_heads) ||
        c.vocab_size <= 0 || c.vocab_size > 4 * 1024 * 1024 ||
        c.seq_len <= 0    || c.seq_len    > 1 << 20) {
        std::fclose(f);
        M2_THROW(FormatError,
                 "implausible config in " << path
                 << " (dim=" << c.dim
                 << " hidden_dim=" << c.hidden_dim
                 << " layers=" << c.n_layers
                 << " heads=" << c.n_heads
                 << " kv_heads=" << c.n_kv_heads
                 << " vocab=" << c.vocab_size
                 << " seq_len=" << c.seq_len << ")");
    }
    if (c.dim % c.n_heads != 0) {
        std::fclose(f);
        M2_THROW(FormatError,
                 "dim (" << c.dim << ") not divisible by n_heads (" << c.n_heads << ")");
    }

    const int head_size = c.head_size();
    const size_t L = size_t(c.n_layers);

    // ---------- tensors -------------------------------------------------
    LlamaWeights w{};
    try {
        fread_f32(f, w.token_embedding_table,
                  size_t(c.vocab_size) * size_t(c.dim),                "token_embedding_table");
        fread_f32(f, w.rms_att_weight, L * size_t(c.dim),              "rms_att_weight");
        fread_f32(f, w.wq, L * size_t(c.dim) * size_t(c.n_heads    * head_size), "wq");
        fread_f32(f, w.wk, L * size_t(c.dim) * size_t(c.n_kv_heads * head_size), "wk");
        fread_f32(f, w.wv, L * size_t(c.dim) * size_t(c.n_kv_heads * head_size), "wv");
        fread_f32(f, w.wo, L * size_t(c.n_heads * head_size) * size_t(c.dim),    "wo");
        fread_f32(f, w.rms_ffn_weight, L * size_t(c.dim),              "rms_ffn_weight");
        fread_f32(f, w.w1, L * size_t(c.dim) * size_t(c.hidden_dim),   "w1");
        fread_f32(f, w.w2, L * size_t(c.hidden_dim) * size_t(c.dim),   "w2");
        fread_f32(f, w.w3, L * size_t(c.dim) * size_t(c.hidden_dim),   "w3");
        fread_f32(f, w.rms_final_weight, size_t(c.dim),                "rms_final_weight");

        if (c.shared_classifier) {
            // Tied embedding: wcls == token_embedding_table. Copy rather
            // than alias so call sites can treat the two independently
            // (e.g. quantize wcls without touching the embedding table).
            w.wcls = w.token_embedding_table;
        } else {
            fread_f32(f, w.wcls,
                      size_t(c.vocab_size) * size_t(c.dim), "wcls");
        }
    } catch (...) {
        std::fclose(f);
        throw;
    }

    // Diagnostic: warn (not error) if there are extra bytes. The legacy
    // Karpathy format embedded freq_cis tables between rms_final_weight
    // and wcls; that file would deserialize incorrectly with this loader
    // and we want the user to notice.
    long here = std::ftell(f);
    std::fseek(f, 0, SEEK_END);
    long end = std::ftell(f);
    if (here != end) {
        LOG_WARN("checkpoint '%s' has %ld trailing bytes after the modern v0 layout. "
                 "If this is a legacy file with freq_cis tables, this loader will give "
                 "the wrong weights -- re-export it with the modern script.",
                 path.c_str(), end - here);
    }

    std::fclose(f);
    return LlamaModel{c, std::move(w)};
}

} // namespace m2
