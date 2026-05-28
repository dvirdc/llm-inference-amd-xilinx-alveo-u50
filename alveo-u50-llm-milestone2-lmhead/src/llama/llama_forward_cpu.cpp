#include "llama_forward_cpu.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>

#include "../common/errors.hpp"
#include "quantization.hpp"  // for gemv_cpu_fp32

namespace m2 {

namespace {

// ----- elementwise helpers -------------------------------------------------

void rmsnorm(std::vector<float>& out,
             const std::vector<float>& x,
             const float* weight, int dim) {
    // RMSNorm: out = x * weight / sqrt(mean(x^2) + eps).
    // This is the *only* normalization Llama uses (no LayerNorm).
    float ss = 0.0f;
    for (int i = 0; i < dim; ++i) ss += x[i] * x[i];
    ss = ss / dim + 1e-5f;
    ss = 1.0f / std::sqrt(ss);
    for (int i = 0; i < dim; ++i) out[i] = weight[i] * (ss * x[i]);
}

void softmax_inplace(float* x, int n) {
    float maxv = x[0];
    for (int i = 1; i < n; ++i) if (x[i] > maxv) maxv = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) { x[i] = std::exp(x[i] - maxv); sum += x[i]; }
    for (int i = 0; i < n; ++i) x[i] /= sum;
}

// y = W @ x, where W is row-major [out_dim, in_dim]. This is the *only*
// matmul shape used here -- a vector against a matrix, a.k.a. GEMV. Every
// Q/K/V/O/W1/W2/W3/LM-head call below dispatches into here.
//
// Eventually each of these GEMVs is what we'll move to the FPGA in later
// milestones. For Milestone 2 the only one we replace is the LM head,
// which the host does explicitly (this function isn't called for it).
void matmul_gemv(float* y, const float* x, const float* w,
                 int in_dim, int out_dim) {
    for (int r = 0; r < out_dim; ++r) {
        float acc = 0.0f;
        const float* row_ptr = w + size_t(r) * size_t(in_dim);
        for (int c = 0; c < in_dim; ++c) acc += row_ptr[c] * x[c];
        y[r] = acc;
    }
}

} // anonymous

LlamaRunState::LlamaRunState(const LlamaConfig& c) {
    const int kv_dim = c.n_kv_heads * c.head_size();
    x.assign(c.dim, 0.0f);
    xb.assign(c.dim, 0.0f);
    xb2.assign(c.dim, 0.0f);
    hb.assign(c.hidden_dim, 0.0f);
    hb2.assign(c.hidden_dim, 0.0f);
    q.assign(c.dim, 0.0f);
    att.assign(size_t(c.n_heads) * size_t(c.seq_len), 0.0f);
    logits.assign(c.vocab_size, 0.0f);
    key_cache.assign(size_t(c.n_layers) * size_t(c.seq_len) * size_t(kv_dim), 0.0f);
    value_cache.assign(size_t(c.n_layers) * size_t(c.seq_len) * size_t(kv_dim), 0.0f);
}

// Shared body: runs everything up to (and including) the final RMSNorm,
// leaving state.x as the LM-head input. Returns by mutating state.
static void forward_body(const LlamaModel& model,
                         LlamaRunState&    s,
                         int               token,
                         int               pos) {
    const LlamaConfig& c = model.config;
    const int dim       = c.dim;
    const int head_size = c.head_size();
    const int kv_dim    = c.n_kv_heads * head_size;
    const int kv_mul    = c.n_heads / c.n_kv_heads;   // GQA broadcast factor
    const int hidden    = c.hidden_dim;

    M2_CHECK(token >= 0 && token < c.vocab_size, ShapeError,
             "forward: bad token " << token);
    M2_CHECK(pos   >= 0 && pos   < c.seq_len,    ShapeError,
             "forward: bad pos "   << pos);

    // 1) Embedding lookup. Copy row `token` of the embedding table into x.
    std::memcpy(s.x.data(),
                model.weights.token_embedding_table.data()
                    + size_t(token) * size_t(dim),
                sizeof(float) * size_t(dim));

    for (int l = 0; l < c.n_layers; ++l) {
        // ---- attention block ----
        // 2) RMSNorm
        rmsnorm(s.xb, s.x,
                model.weights.rms_att_weight.data() + size_t(l) * size_t(dim),
                dim);

        // 3) Q/K/V projection (three GEMVs). Q is [dim] (n_heads*head_size).
        //    K and V are [kv_dim] (n_kv_heads*head_size). We write K and V
        //    *directly* into the KV cache at slot `pos` for this layer --
        //    no separate intermediate buffer needed.
        const size_t kv_off = size_t(l) * size_t(c.seq_len) * size_t(kv_dim)
                            + size_t(pos) * size_t(kv_dim);
        float* k_dst = s.key_cache.data()   + kv_off;
        float* v_dst = s.value_cache.data() + kv_off;

        matmul_gemv(s.q.data(), s.xb.data(),
                    model.weights.wq.data() + size_t(l) * size_t(dim) * size_t(dim),
                    dim, dim);
        matmul_gemv(k_dst, s.xb.data(),
                    model.weights.wk.data() + size_t(l) * size_t(dim) * size_t(kv_dim),
                    dim, kv_dim);
        matmul_gemv(v_dst, s.xb.data(),
                    model.weights.wv.data() + size_t(l) * size_t(dim) * size_t(kv_dim),
                    dim, kv_dim);

        // 4) Rotary positional embedding (RoPE) on Q and K. Pairs of
        //    consecutive head-channels are rotated by an angle that
        //    depends on position. Karpathy's run.c version (no
        //    precomputed freq table -- we compute on the fly).
        for (int i = 0; i < dim; i += 2) {
            int head_dim = i % head_size;
            float freq = 1.0f / std::pow(10000.0f, float(head_dim) / float(head_size));
            float val = float(pos) * freq;
            float fcr = std::cos(val), fci = std::sin(val);
            // rotate Q
            float q0 = s.q[i], q1 = s.q[i + 1];
            s.q[i]     = q0 * fcr - q1 * fci;
            s.q[i + 1] = q0 * fci + q1 * fcr;
            // rotate K, but only the channels that exist in the KV head
            if (i < kv_dim) {
                float k0 = k_dst[i], k1 = k_dst[i + 1];
                k_dst[i]     = k0 * fcr - k1 * fci;
                k_dst[i + 1] = k0 * fci + k1 * fcr;
            }
        }

        // 5) Multi-head attention.
        //    For each head h, compute scores[t] = Q[h] . K_cache[t, h_kv] / sqrt(d),
        //    softmax over t in [0, pos], then xb[h] = sum_t scores[t] * V_cache[t, h_kv].
        //
        //    Grouped-query attention: kv_head = h / kv_mul.
        const float scale = 1.0f / std::sqrt(float(head_size));
        for (int h = 0; h < c.n_heads; ++h) {
            const int kv_h = h / kv_mul;
            float* q_h = s.q.data() + size_t(h) * size_t(head_size);
            float* att_row = s.att.data() + size_t(h) * size_t(c.seq_len);

            // scores
            for (int t = 0; t <= pos; ++t) {
                const float* k_t = s.key_cache.data()
                    + size_t(l) * size_t(c.seq_len) * size_t(kv_dim)
                    + size_t(t) * size_t(kv_dim)
                    + size_t(kv_h) * size_t(head_size);
                float dot = 0.0f;
                for (int i = 0; i < head_size; ++i) dot += q_h[i] * k_t[i];
                att_row[t] = dot * scale;
            }
            // softmax over [0, pos]
            softmax_inplace(att_row, pos + 1);

            // weighted sum of V
            float* xb_h = s.xb.data() + size_t(h) * size_t(head_size);
            std::memset(xb_h, 0, sizeof(float) * size_t(head_size));
            for (int t = 0; t <= pos; ++t) {
                const float* v_t = s.value_cache.data()
                    + size_t(l) * size_t(c.seq_len) * size_t(kv_dim)
                    + size_t(t) * size_t(kv_dim)
                    + size_t(kv_h) * size_t(head_size);
                float a = att_row[t];
                for (int i = 0; i < head_size; ++i) xb_h[i] += a * v_t[i];
            }
        }

        // 6) Output projection Wo. Then residual add into x.
        matmul_gemv(s.xb2.data(), s.xb.data(),
                    model.weights.wo.data() + size_t(l) * size_t(dim) * size_t(dim),
                    dim, dim);
        for (int i = 0; i < dim; ++i) s.x[i] += s.xb2[i];

        // ---- FFN block (SwiGLU) ----
        // 7) RMSNorm
        rmsnorm(s.xb, s.x,
                model.weights.rms_ffn_weight.data() + size_t(l) * size_t(dim),
                dim);

        // 8) Two parallel projections (w1 and w3), then SiLU(w1) * w3.
        matmul_gemv(s.hb.data(),  s.xb.data(),
                    model.weights.w1.data() + size_t(l) * size_t(dim) * size_t(hidden),
                    dim, hidden);
        matmul_gemv(s.hb2.data(), s.xb.data(),
                    model.weights.w3.data() + size_t(l) * size_t(dim) * size_t(hidden),
                    dim, hidden);
        for (int i = 0; i < hidden; ++i) {
            // SiLU(x) = x * sigmoid(x)
            float v = s.hb[i];
            v = v * (1.0f / (1.0f + std::exp(-v)));
            s.hb[i] = v * s.hb2[i];
        }
        // 9) Project back down. Residual add into x.
        matmul_gemv(s.xb.data(), s.hb.data(),
                    model.weights.w2.data() + size_t(l) * size_t(hidden) * size_t(dim),
                    hidden, dim);
        for (int i = 0; i < dim; ++i) s.x[i] += s.xb[i];
    }

    // 10) Final RMSNorm. The LM-head input is now sitting in state.x.
    rmsnorm(s.x, s.x, model.weights.rms_final_weight.data(), dim);
}

void llama_forward_cpu_until_lm_head(const LlamaModel& model,
                                     LlamaRunState&    state,
                                     int               token,
                                     int               pos) {
    forward_body(model, state, token, pos);
}

void llama_forward_cpu(const LlamaModel& model,
                       LlamaRunState&    state,
                       int               token,
                       int               pos) {
    forward_body(model, state, token, pos);

    // 11) LM head (FP32 reference path). For hybrid mode the caller will
    //     overwrite state.logits with the FPGA result -- but we always
    //     compute the FP32 path here so a single forward_cpu call still
    //     produces logits suitable for sampling, even when no FPGA is
    //     attached. The cost is one extra GEMV per token, which is
    //     negligible relative to the attention loop.
    gemv_cpu_fp32(model.weights.wcls, state.x,
                  model.config.vocab_size, model.config.dim,
                  state.logits);
}

} // namespace m2
