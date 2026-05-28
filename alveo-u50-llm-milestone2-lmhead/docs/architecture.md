# Architecture (Milestone 2)

## Why the LM head first

The LM head is the single GEMV that comes after every transformer step,
maps a `dim`-vector to a `vocab_size`-vector, and is the operation whose
output we sample from. Three reasons it's the right first thing to
offload:

1. **Single matrix, no dependencies on per-layer state.** It needs only
   the post-final-RMSNorm vector and the wcls matrix. We can offload it
   in isolation without rewriting the whole layer.
2. **It's a true GEMV** -- exactly the kernel we already built in
   Milestone 1. Zero new kernel work.
3. **Easy to bit-exact-verify.** We have a CPU INT8 reference; the FPGA
   must produce the same INT32 buffer. If it doesn't, we know within one
   token where the bug is.

## End-to-end data flow

```
            host (CPU)                                  Alveo U50
            ----------                                  ---------
prompt ---> tokenizer ---> tokens
                            |
                            v
            +---------------+
            | CPU transformer (every layer)            (this whole block
            |  RMSNorm                                  stays on CPU for
            |  Q/K/V projection                         Milestone 2)
            |  RoPE
            |  Attention (KV cache reads/writes)
            |  Output projection
            |  RMSNorm
            |  MLP (W1, W3, SiLU, W2)
            |  residual adds
            +---------------+
                            |
                            v
            final RMSNorm  --->  hidden vector x (dim float32)
                            |
                            v
            dynamic INT8 quantize  ---> qx (dim int8) + x_scale
                            |
                            v
            PCIe write of qx ----------> HBM[1]
                                                    +-----------------+
            HBM[0] holds the int8 wcls weights ---> | gemv_int8_1 CU  |
            (loaded ONCE at startup, stays put)     | (Milestone 1)   |
                                                    +-----------------+
                                                              |
            PCIe read of y_int32 <----------- HBM[2]          |
                            |                                 |
                            v
            y_fp32 = y_int32 * row_scales * x_scale
                            |
                            v
            sampler  --->  next token
```

## Memory buffers

| Buffer                       | Where             | Size                                 | Lifetime           |
|------------------------------|-------------------|--------------------------------------|--------------------|
| llama config + FP32 weights  | host RAM          | ~60 MB for stories15M                | program            |
| quantized LM head (int8)     | host RAM + HBM[0] | vocab*dim bytes (~9 MB)              | program            |
| quantized LM head row_scales | host RAM          | vocab*4 bytes                        | program            |
| per-token hidden vector x    | host RAM          | dim*4 bytes                          | per token          |
| dynamic-quant qx (int8)      | host RAM + HBM[1] | dim bytes                            | per token          |
| FPGA output y_int32          | host RAM + HBM[2] | vocab*4 bytes                        | per token          |

## Where PCIe overhead appears

Per token we move:
- `dim` bytes host -> HBM[1]  (~288 bytes for stories15M)
- `vocab*4` bytes HBM[2] -> host  (~128 KB for stories15M)

Both are small. The weight matrix (~9 MB) is moved exactly *once* per
program run, into HBM[0]. This is the central design property -- if we
moved the weights per token, we'd be PCIe-bound forever.

For a 7B model the LM head alone is ~2 GB. Same principle scales: HBM
fits it once, you ship a 4 KB hidden vector each token.

## Why this milestone is about correctness, not speed

For stories15M the LM head is one of ~37 matrix ops per token. Moving
just that one to the FPGA does not make decoding faster yet -- the
PCIe round-trip and the rest of the CPU transformer dominate.

We accept that. The deliverable for Milestone 2 is:

- Correct loader + tokenizer.
- Correct quantization (CPU INT8 == FPGA INT8 bit-exactly).
- Correct dequantization (FP32 ~= FPGA, top-1 token usually agrees).
- A `FpgaGemvEngine` abstraction that later milestones can pass *more*
  weights through with zero CPU-side changes.

Speed comes in M5+ when we move Q/K/V and the MLP weights too and the
CPU side stops being the bottleneck.
