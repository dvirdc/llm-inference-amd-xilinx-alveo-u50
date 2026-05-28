// llama_forward_cpu.hpp -- CPU implementation of the Llama2 forward pass.
//
// This is a direct port of the inference loop from Karpathy's run.c,
// rewritten in C++ with std::vector and explicit comments naming each
// step's role in the transformer. The output we care about most in
// Milestone 2 is the *final hidden vector* (post-RMSNorm, pre-LM-head),
// because that's the vector that gets quantized and shipped to the FPGA.

#pragma once

#include <vector>

#include "llama_model.hpp"

namespace m2 {

// Per-forward scratch state. Holds:
//   * per-token activation buffers (re-used every step)
//   * the KV cache (allocated once for the full sequence)
//
// Constructing this object allocates O(seq_len * n_layers * n_kv_heads
// * head_size) bytes of cache -- ~1.5 MB for stories15M, harmless.
struct LlamaRunState {
    std::vector<float> x;        // [dim]      residual stream
    std::vector<float> xb;       // [dim]      scratch
    std::vector<float> xb2;      // [dim]      scratch
    std::vector<float> hb;       // [hidden]   FFN scratch
    std::vector<float> hb2;      // [hidden]   FFN scratch
    std::vector<float> q;        // [dim]      Q for current token
    std::vector<float> att;      // [n_heads * seq_len] attention weights
    std::vector<float> logits;   // [vocab]
    std::vector<float> key_cache;   // [n_layers * seq_len * n_kv_heads*head_size]
    std::vector<float> value_cache; // same shape

    explicit LlamaRunState(const LlamaConfig& c);
};

// Run one decode step.
//   token : the token id about to be embedded (input to the network)
//   pos   : 0-based position in the sequence (controls KV-cache indexing
//           and the causal mask)
//
// On return, `state.logits` is populated with the FP32 LM-head output
// (i.e. logits over the vocabulary). `state.x` holds the post-final-
// RMSNorm hidden vector -- this is the quantization input for the
// FPGA LM-head path.
void llama_forward_cpu(const LlamaModel& model,
                       LlamaRunState&    state,
                       int               token,
                       int               pos);

// Same as llama_forward_cpu, but stops just before the LM head, so the
// caller can inspect `state.x` (the post-final-norm hidden vector) and
// run their own LM head (CPU FP32 / CPU INT8 / FPGA INT8). Used by the
// validation harness in apps/main_hybrid_lmhead.cpp.
void llama_forward_cpu_until_lm_head(const LlamaModel& model,
                                     LlamaRunState&    state,
                                     int               token,
                                     int               pos);

} // namespace m2
