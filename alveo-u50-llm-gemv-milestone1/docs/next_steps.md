# Next steps -- staged roadmap toward an LLM inference engine

Each milestone below is sized to be a single focused PR / branch that
ships a correct, measurable improvement. Don't combine milestones.

---

## M1 -- Simple INT8 GEMV  *(this repo)*

- One CU, one HBM channel per port, scalar loads, INT32 accumulation.
- Goal: end-to-end correctness, measurable timing, clean repo.
- Status: ✅

## M2 -- Vectorized INT8 GEMV with `ap_uint<512>`

- Change kernel pointer types so each AXI beat carries **64 INT8 lanes**.
- Inner loop computes 64 MACs/cycle instead of 1.
- Expected speedup at the kernel level: ~30-50× before any other change.
- Risk: alignment + tail handling for non-multiple-of-64 `cols`.

## M3 -- Multiple output rows per kernel invocation with unrolling

- Tile the row loop and `#pragma HLS UNROLL` it so e.g. 4 dot products
  share a single read of `x` (broadcast).
- Reduces redundant `x` reads, raises DSP utilization.

## M4 -- HBM-sharded weights

- Split `weights` across `HBM[0..7]` with 8 m_axi ports.
- Re-write the load to round-robin across ports.
- Goal: approach aggregate HBM read bandwidth.

## M5 -- Multiple compute units

- `nk=gemv_int8:N:gemv_int8_1...gemv_int8_N` -- N CUs partition the row
  dimension.
- Host runs N kernel invocations concurrently and stitches the result.
- Diminishing returns past the point where HBM is saturated.

## M6 -- INT4 packed weights

- Pack two int4 weights per byte.
- Half the bytes per token => ~2× the bandwidth-limited tok/s.
- Requires a tiny dequant path (or fused MAC against `(w << 4) >> 4`).

## M7 -- Fused RMSNorm + GEMV

- RMSNorm is cheap but memory-bound; fusing it into the GEMV load
  pipeline avoids a separate full read of the activation.
- Saves one full activation pass per layer.

## M8 -- Q/K/V projection kernel

- A specialization of GEMV with three output streams and the same `x`.
- Often combinable into a single fused kernel for one read of `x`.

## M9 -- RoPE

- Rotary position embedding -- a tiny per-head trig rotation.
- Cheap to compute; the question is just how/where to insert it in the
  pipeline (typically immediately after Q/K projections).

## M10 -- KV cache

- Append-only ring buffer per layer in HBM.
- Host manages allocation; kernel writes new K/V vectors at the current
  position.

## M11 -- Attention dot product

- `softmax(Q @ K^T / sqrt(d)) @ V` for one query and the full cached K/V.
- Two GEMVs and a small softmax. Memory-bound on the cache reads.

## M12 -- MLP (up + activation + down)

- Two GEMVs and a SiLU / GELU activation in between.
- The activation is element-wise and trivially fused into the second
  GEMV's input stream.

## M13 -- Tiny transformer block

- Stitch M7 + M8 + M9 + M10 + M11 + M12 into one decoder block.
- Host iterates the block N_layers times per token.
- This is the first point at which we can run a real model end-to-end.

---

## Things we are NOT doing yet (deliberately deferred)

- **Multi-batch / GEMM kernels.** Latency-first means batch-1; throughput
  comes later.
- **Mixed precision (FP16 / BF16) kernels.** All-INT8 (then INT4) is the
  bandwidth-efficient path.
- **Dynamic shapes in the kernel.** Stick to a small set of pre-built
  xclbins per shape until we know we need one-fits-all.
- **PCIe peer-to-peer or RDMA.** Single-card, host-driven for the whole
  roadmap above.
