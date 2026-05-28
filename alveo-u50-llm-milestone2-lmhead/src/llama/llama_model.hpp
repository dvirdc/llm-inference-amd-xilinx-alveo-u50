// llama_model.hpp -- in-memory representation of a Karpathy llama2.c
// checkpoint, plus the loader.
//
// The on-disk format (modern, "v0" -- no precomputed RoPE table):
//   bytes 0..27  : seven int32s (dim, hidden_dim, n_layers, n_heads,
//                  n_kv_heads, vocab_size [signed!], seq_len)
//   then a sequence of float32 blobs, in this exact order:
//      token_embedding_table  [vocab_size, dim]
//      rms_att_weight         [n_layers, dim]
//      wq                     [n_layers, dim, n_heads*head_size]
//      wk                     [n_layers, dim, n_kv_heads*head_size]
//      wv                     [n_layers, dim, n_kv_heads*head_size]
//      wo                     [n_layers, n_heads*head_size, dim]
//      rms_ffn_weight         [n_layers, dim]
//      w1                     [n_layers, dim, hidden_dim]
//      w2                     [n_layers, hidden_dim, dim]
//      w3                     [n_layers, dim, hidden_dim]
//      rms_final_weight       [dim]
//      wcls                   [vocab_size, dim]   <-- only if vocab_size<0
//
// Layouts are *row-major*. Every matrix is the natural (out, in) layout
// where the OUTPUT dim is the slow index. That matches the GEMV the
// kernel does: y[row] = sum_c W[row, c] * x[c].

#pragma once

#include <string>
#include <vector>

#include "llama_config.hpp"

namespace m2 {

struct LlamaWeights {
    // [vocab_size * dim] -- per-token embedding table. Also re-used as wcls
    // when shared_classifier==true (we copy into wcls below so call sites
    // never have to special-case).
    std::vector<float> token_embedding_table;

    // [n_layers * dim] each.
    std::vector<float> rms_att_weight;
    std::vector<float> rms_ffn_weight;

    // [n_layers * dim * n_heads*head_size]
    std::vector<float> wq;
    // [n_layers * dim * n_kv_heads*head_size]
    std::vector<float> wk;
    std::vector<float> wv;
    // [n_layers * n_heads*head_size * dim]
    std::vector<float> wo;

    // [n_layers * dim * hidden_dim]
    std::vector<float> w1;
    // [n_layers * hidden_dim * dim]
    std::vector<float> w2;
    // [n_layers * dim * hidden_dim]
    std::vector<float> w3;

    // [dim]
    std::vector<float> rms_final_weight;

    // [vocab_size * dim] -- *always populated*, even when shared_classifier
    // is true (in that case it's a copy of token_embedding_table).
    std::vector<float> wcls;
};

struct LlamaModel {
    LlamaConfig  config;
    LlamaWeights weights;
};

// Loads a Karpathy-format checkpoint from `path`. Throws m2::IoError if the
// file cannot be opened, m2::FormatError if the header looks wrong or the
// expected number of floats is not present.
LlamaModel load_llama_model(const std::string& path);

// Convenience: print the config to stderr so the user sees what we read.
void print_llama_config(const LlamaConfig& c);

} // namespace m2
