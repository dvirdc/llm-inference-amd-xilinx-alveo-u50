// llama_config.hpp -- model dimensions parsed from the Karpathy llama2.c
// checkpoint header. See docs/model_format.md for the on-disk layout.

#pragma once

namespace m2 {

struct LlamaConfig {
    int dim         = 0;   // hidden / model dim          (e.g. 288 for stories15M)
    int hidden_dim  = 0;   // FFN inner dim (4*dim usually) (e.g. 768)
    int n_layers    = 0;   // transformer blocks           (e.g. 6)
    int n_heads     = 0;   // attention heads              (e.g. 6)
    int n_kv_heads  = 0;   // grouped-query KV heads       (== n_heads for llama2)
    int vocab_size  = 0;   // |V|                          (32000 for sentencepiece)
    int seq_len     = 0;   // max sequence length          (256 for stories15M)

    // Convenience: per-head size. Karpathy enforces dim % n_heads == 0.
    int head_size() const { return dim / n_heads; }

    // True when token_embedding_table is also used as the LM head (wcls).
    // The raw `vocab_size` field in the checkpoint is signed; positive
    // means tied (shared_classifier=1), negative means there is a separate
    // wcls block at the end of the file. We hide that detail here.
    bool shared_classifier = true;
};

} // namespace m2
