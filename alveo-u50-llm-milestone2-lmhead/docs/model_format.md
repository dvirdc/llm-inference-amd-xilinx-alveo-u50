# model_format.md

The on-disk format we read is the *modern* (post-2023) Karpathy llama2.c
"v0" format: pure binary, no precomputed RoPE tables, no magic header,
no version byte.

## Header (28 bytes)

Seven signed little-endian int32s in this exact order:

| Field        | Notes                                                       |
|--------------|-------------------------------------------------------------|
| `dim`        | Model / residual stream width                               |
| `hidden_dim` | FFN inner dim (typically ~4*dim)                            |
| `n_layers`   | Number of transformer blocks                                |
| `n_heads`    | Attention heads                                             |
| `n_kv_heads` | KV heads (== n_heads for llama2; smaller for GQA models)    |
| `vocab_size` | **Signed.** Positive == LM head tied to embedding;          |
|              | negative == separate wcls block at the end.                 |
|              | The actual vocab size is `abs(vocab_size)`.                 |
| `seq_len`    | Max sequence length                                         |

## Tensors (FP32, in this exact order)

All matrices are row-major. The dimension marked OUT below is the slow
(row) index. **This row-major convention is what makes our INT8 GEMV
kernel a drop-in: `y[row] = sum_c W[row, c] * x[c]` matches exactly.**

| Tensor                | Shape                                   | Notes                          |
|-----------------------|-----------------------------------------|--------------------------------|
| token_embedding_table | [vocab_size, dim]                        | Used as wcls when tied          |
| rms_att_weight        | [n_layers, dim]                          | One RMSNorm per layer (pre-att) |
| wq                    | [n_layers, dim_OUT, dim_IN]              | dim_OUT = n_heads*head_size     |
| wk                    | [n_layers, kv_dim_OUT, dim_IN]           | kv_dim_OUT = n_kv_heads*head_size|
| wv                    | [n_layers, kv_dim_OUT, dim_IN]           |                                |
| wo                    | [n_layers, dim_OUT, dim_IN]              | output projection               |
| rms_ffn_weight        | [n_layers, dim]                          | RMSNorm before FFN              |
| w1                    | [n_layers, hidden_dim_OUT, dim_IN]       | SwiGLU "gate"                   |
| w2                    | [n_layers, dim_OUT, hidden_dim_IN]       | Down-projection                 |
| w3                    | [n_layers, hidden_dim_OUT, dim_IN]       | SwiGLU "up"                     |
| rms_final_weight      | [dim]                                    | Final RMSNorm before LM head    |
| wcls                  | [vocab_size, dim]                        | Only if vocab_size < 0          |

## How we expose this

`src/llama/llama_model.cpp` parses the header, reads each tensor with a
single `fread` into a `std::vector<float>`, and -- when shared_classifier
is true -- copies token_embedding_table into wcls so callers never have
to special-case ties.

If the file has trailing bytes after this layout, we log a WARN but do
not error. That's almost certainly a *legacy* checkpoint with embedded
freq_cis tables; the safe fix is to re-export with the modern script.

## Tied embedding caveat (Milestone 2 specific)

When `shared_classifier == true`, mutating wcls (e.g. by quantizing it
to INT8 for the FPGA) would also mutate the embedding table -- breaking
inference. To prevent that, the loader **copies** rather than aliases:
`weights.wcls = weights.token_embedding_table`. The price is one extra
~9 MB allocation; for our sizes it's irrelevant.
