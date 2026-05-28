# Architecture (milestone 1)

This document explains how the host and the FPGA cooperate to execute one
INT8 GEMV. Read it alongside `src/kernels/gemv_int8.cpp` and
`src/host/main.cpp`.

---

## 1. End-to-end data flow

```
                  ┌──────────────────────────────────────────────┐
                  │                 Host (x86)                    │
                  │                                                │
                  │ 1. parse args (--xclbin --rows --cols --seed)  │
                  │ 2. fill_random_int8(weights), fill_random_int8(x)
                  │ 3. gemv_cpu_ref() ───────► y_ref   (reference) │
                  │                                                │
                  │ 4. xrt::device(0) + load_xclbin                │
                  │ 5. xrt::kernel("gemv_int8")                    │
                  │ 6. xrt::bo bo_w, bo_x, bo_y on HBM banks       │
                  │ 7. memcpy host→BO map (no DMA yet)             │
                  │                                                │
                  │ 8. bo_w.sync(TO_DEVICE) ─┐                     │
                  │ 9. bo_x.sync(TO_DEVICE) ─┤  H2D                │
                  │10. run = kernel(bo_w, bo_x, bo_y, rows, cols)──┼──► PCIe → Alveo U50
                  │11. run.wait()                                  │
                  │12. bo_y.sync(FROM_DEVICE) ◄─                   │  D2H
                  │                                                │
                  │13. compare y_dev vs y_ref bit-exact            │
                  │14. print timings + PASS/FAIL                   │
                  └──────────────────────────────────────────────┘

                  ┌──────────────────────────────────────────────┐
                  │                  Alveo U50                    │
                  │                                                │
                  │  HBM[0]: weights  ────► gmem0 m_axi port       │
                  │  HBM[1]: x        ────► gmem1 m_axi port       │
                  │  HBM[2]: y        ◄──── gmem2 m_axi port       │
                  │                                                │
                  │           ┌───────────────────────┐            │
                  │           │   gemv_int8_1 (1 CU)  │            │
                  │           │                       │            │
                  │           │  for row in 0..rows:  │            │
                  │           │     acc = 0           │            │
                  │           │     for col in 0..cols│            │
                  │           │        acc += W*x     │  (II=1)    │
                  │           │     y[row] = acc      │            │
                  │           └───────────────────────┘            │
                  └──────────────────────────────────────────────┘
```

## 2. Memory buffers

Host-side:

| Buffer | Type | Size (rows=R, cols=C) | Notes |
|--------|------|-----------------------|-------|
| `h_weights` | `std::vector<int8_t>` | `R*C` bytes | row-major, generated from seed |
| `h_x`       | `std::vector<int8_t>` | `C` bytes   | input vector |
| `y_ref`     | `std::vector<int32_t>` | `4*R` bytes | CPU reference output |
| `y_dev`     | local copy of `bo_y` | `4*R` bytes | device output, compared to `y_ref` |

Device-side (XRT BOs, each in its own HBM bank):

| BO | Direction | Bank (default cfg) | Size |
|----|-----------|---------------------|------|
| `bo_w` | H2D | HBM[0] | `R*C` bytes |
| `bo_x` | H2D | HBM[1] | `C` bytes |
| `bo_y` | D2H | HBM[2] | `4*R` bytes |

The BO's host-mapped pointer (`xrt::bo::map<T*>()`) is a normal CPU
pointer; XRT allocates it as a pinned region that DMA can read from.
We `memcpy` into it on the host, then call `sync(TO_DEVICE)` which
triggers the DMA write across PCIe into HBM.

## 3. HBM mapping

The U50 has 8 GB of HBM2 organized as **32 pseudo-channels** of 256 MB
each (`HBM[0]` through `HBM[31]`). Each channel is ~14.5 GB/s peak read
bandwidth; aggregate peak is ~316 GB/s. For milestone 1 we use just three
channels (one per kernel port). The other 29 are idle -- a deliberate
choice so the first version is simple to debug. Sharding weights across
multiple banks is in milestone 2.

The mapping is set by `configs/u50_hbm.cfg`:

```
[connectivity]
nk=gemv_int8:1:gemv_int8_1
sp=gemv_int8_1.weights:HBM[0]
sp=gemv_int8_1.x:HBM[1]
sp=gemv_int8_1.y:HBM[2]
```

The kernel port names (`weights`, `x`, `y`) are the kernel argument names
from the C signature -- v++ resolves them through the HLS-generated
`kernel.xml`. The bundle names (`gmem0`, `gmem1`, `gmem2`) in the HLS
pragmas determine how many physical AXI master ports the kernel exposes;
distinct bundles = distinct ports = independent HBM channels.

## 4. Kernel call sequence

From `main.cpp`:

```cpp
auto run = krnl(bo_w, bo_x, bo_y, rows, cols);  // start CU, return immediately
run.wait();                                      // block until CU asserts done
```

Under the hood, XRT:
1. writes BO physical addresses + scalar args to the CU's AXI-Lite
   register file,
2. pulses the `ap_start` bit,
3. polls / waits on `ap_done`.

The CU's reading of weights and x is interleaved with writing to y; the
inner-loop `PIPELINE II=1` directive means once the pipeline is full, one
MAC happens per cycle.

## 5. Why GEMV matters for LLM inference

In transformer **decoding** (generating one token at a time):

| Op | Shape | Operation |
|----|-------|-----------|
| Q projection  | `[d_model] @ [d_model, d_q]` | GEMV |
| K projection  | `[d_model] @ [d_model, d_k]` | GEMV |
| V projection  | `[d_model] @ [d_model, d_v]` | GEMV |
| attn `Q @ K^T` | `[d_head] @ [d_head, T_cache]` per head | GEMV over cached K |
| attn `... @ V` | `[T_cache] @ [T_cache, d_head]` per head | GEMV over cached V |
| Output proj   | `[d_q*n_heads] @ [d_q*n_heads, d_model]` | GEMV |
| MLP up        | `[d_model] @ [d_model, d_ff]` | GEMV |
| MLP down      | `[d_ff] @ [d_ff, d_model]` | GEMV |
| LM head       | `[d_model] @ [d_model, vocab]` | GEMV |

Every one of these is a vector times a matrix. No GEMMs anywhere in
batch-1 decoding -- which is why a fast GEMV kernel is the right
foundation.

## 6. Why batch-1 decode is memory-bandwidth limited

For a single GEMV `y = W x` with W of shape `[M, K]`:

- compute  = `2*M*K` ops (one multiply + one add per element)
- bytes read = `M*K*sizeof(weight) + K*sizeof(x) + M*sizeof(y)`

For typical LLM dimensions, `M*K` dominates, so the **arithmetic intensity**
is roughly `2 ops / sizeof(weight) byte`. With INT8 weights that's 2
ops/byte; with INT4 that's 4 ops/byte. Either way it's an order of
magnitude below what an FPGA's DSP fabric could compute if fed -- so the
ceiling is set by HBM bandwidth.

Concretely, for a 7B-parameter model on a U50:
- weights to read per token: ~7 GB (INT8) / ~3.5 GB (INT4)
- U50 HBM peak: ~316 GB/s
- absolute ceiling: ~45 INT8 tok/s, ~90 INT4 tok/s

Hitting half of those numbers is the goal of later milestones. The point
of milestone 1 is to be correct first, so we have a known-good baseline
to measure against.

## 7. Current limitations

- Single CU, scalar (8-bit) loads -- no chance of saturating HBM yet.
- No quantization scales / zero-points. Works only because our test data
  range is tight enough that the bit-exact int-only path is meaningful.
- No streaming / dataflow: the outer row-loop runs serially and only the
  inner col-loop is pipelined. Each row pays a small pipeline fill cost.
- Host code does a CPU-side memcpy into the BO map before `sync()`. For
  large `R*C` this is non-negligible; we'll switch to async-allocated
  pinned buffers in a later milestone.
