# LLM Inference using AMD/Xilinx Alveo U50

Building toward transformer (Llama-style) LLM inference on an **AMD/Xilinx
Alveo U50** FPGA, one bounded milestone at a time. Each milestone is a
self-contained, buildable project with its own README, tests, docs, and
both a command-line (Makefile) and **Vitis Unified IDE** build flow.

The throughline is a single reusable building block: an **INT8 GEMV**
(matrix-vector multiply) HLS kernel. Batch-1 LLM token decoding is
dominated by GEMVs (Q/K/V projections, MLP projections, the LM head), so
getting one correct, HBM-connected, bit-exact INT8 GEMV is the foundation
everything else builds on.

## Milestones

### [`alveo-u50-llm-gemv-milestone1/`](alveo-u50-llm-gemv-milestone1/) — INT8 GEMV kernel
A clean, production-quality INT8 matrix-vector multiply on the U50:

- Vitis HLS `gemv_int8` kernel, INT8 inputs, INT32 accumulation.
- XRT C++ host: deterministic data, device buffers, kernel run, CPU
  reference comparison, per-stage timing, PASS/FAIL.
- HBM connectivity config, CPU-only unit tests, scripts, docs.
- Native Vitis IDE components (system + host + HLS kernel).

### [`alveo-u50-llm-milestone2-lmhead/`](alveo-u50-llm-milestone2-lmhead/) — FPGA LM head
Runs a Karpathy `llama2.c`-style tiny Llama model on CPU, with the final
**LM-head GEMV offloaded to the U50** via the Milestone 1 kernel:

- Karpathy checkpoint loader + SentencePiece-BPE tokenizer + sampler.
- Full CPU transformer forward (RMSNorm, GQA attention, RoPE, KV cache,
  SwiGLU MLP).
- Per-row INT8 weight quantization + dynamic INT8 activation quantization.
- `FpgaGemvEngine` wrapper that keeps weights resident in HBM and ships
  only the activation per token.
- Three-way validation: CPU FP32 vs CPU INT8 vs FPGA INT8, with
  **bit-exact** CPU-INT8 == FPGA-INT8 required, plus top-1/top-5 logit
  comparison.

## Hardware / toolchain

- AMD/Xilinx Alveo U50 (HBM, `xilinx_u50_gen3x16_xdma_5_202210_1`).
- AMD Vitis + XRT.
- C++17 host toolchain.

## Quick start

Each milestone has full instructions in its own README. The fastest thing
that needs no FPGA is the CPU-only test suite:

```bash
# Milestone 1
cd alveo-u50-llm-gemv-milestone1 && make -C tests

# Milestone 2
cd alveo-u50-llm-milestone2-lmhead && make test
```

FPGA builds (`make all TARGET=hw DEVICE=...`) and the Vitis IDE flow are
documented per-milestone.

## What is NOT in this repo

By design, the following are not committed (see `.gitignore`):

- Build outputs (`build/`, `*.xclbin`, `*.xo`).
- Model checkpoints / tokenizers (`*.bin`) — download per
  `alveo-u50-llm-milestone2-lmhead/models/README.md`.
- The cloned `llama2.c` reference — fetch with the provided script.
- Vitis IDE / clangd machine-specific caches.

## Roadmap

Next milestones (detailed in each project's `docs/next_steps.md`): move
Q/K/V and MLP projections to the FPGA, fuse QKV, INT4 packed weights,
512-bit packed loads, HBM sharding, multiple compute units, and ultimately
a full transformer block on the device.
