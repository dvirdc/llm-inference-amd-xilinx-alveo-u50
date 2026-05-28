# alveo-u50-llm-milestone2-lmhead

**Milestone 2** of the U50 LLM accelerator series: run Karpathy-style
Llama2 inference on CPU, with the **final LM-head GEMV offloaded to the
FPGA** using the Milestone 1 `gemv_int8` kernel.

## 1. What this milestone is

- Loads a Karpathy `llama2.c` checkpoint and tokenizer.
- Runs the full transformer forward on CPU.
- Quantizes the final `wcls` matrix per-row to INT8.
- Each token: dynamically quantizes the post-final-RMSNorm hidden
  vector, ships it to the FPGA, runs the INT8 GEMV, dequantizes the
  result back to FP32 logits, samples the next token.
- Three comparison paths every token (CPU FP32 / CPU INT8 / FPGA INT8)
  with **bit-exact** equality required between CPU INT8 and FPGA INT8.

## 2. What this milestone is NOT

- Not faster than CPU yet -- the LM head is one matmul of ~37 per
  token. Speed comes when we move Q/K/V and MLP weights in M3..M6.
- Not INT4. Not 512-bit packed loads. Not multi-CU. Not sharded HBM.
  All listed in `docs/next_steps.md`.
- Not a re-implementation of llama2.c. We port what we need; for the
  upstream reference run `bash scripts/fetch_karpathy_llama2c.sh`.

## 3. Architecture (text diagram)

```
prompt
  -> tokenizer
  -> CPU transformer (every layer)
       RMSNorm -> Q/K/V -> RoPE -> Attention -> Wo -> RMSNorm -> SwiGLU MLP
  -> final RMSNorm
  -> post-final hidden vector x  (FP32, length=dim)
  -> dynamic INT8 quantize  (yields qx and x_scale)
  -> PCIe write qx  ----------> HBM[1]
                                +-----------------+
  HBM[0] : int8 wcls (resident, loaded once) ---> |  gemv_int8 CU   |
                                                  +-----------------+
  HBM[2] : int32 y  <----------- PCIe read
  -> dequantize y_int32 * row_scales * x_scale -> y_fp32 logits
  -> sampler -> next token
```

## 4. Prerequisites

- Alveo U50 working under XRT (Milestone 1 confirmed this).
- AMD/Xilinx Vitis + XRT installed and sourced.
- A Karpathy llama2.c checkpoint (`stories15M.bin`) + tokenizer
  (`tokenizer.bin`).
- g++ with C++17.

## 5. Duplicate / set up

This project was duplicated from
`../alveo-u50-llm-gemv-milestone1`. If you're starting fresh:

```bash
cd alveo-u50-llm-milestone2-lmhead
bash scripts/copy_from_milestone1.sh ../alveo-u50-llm-gemv-milestone1
# (or: make copy-milestone1 MILESTONE1_DIR=../alveo-u50-llm-gemv-milestone1)
```

This copies four files: the kernel, the XRT utility class, the timer
header, and the HBM connectivity config.

## 6. Fetch Karpathy llama2.c (reference)

```bash
bash scripts/fetch_karpathy_llama2c.sh
```

Then download stories15M and the tokenizer into `models/` -- see
`models/README.md` for URLs.

## 7. Build

CPU-only path (no FPGA needed):

```bash
make test                                       # 4 CPU-only unit suites
make host                                       # 4 apps in build/
```

FPGA path:

```bash
source /opt/xilinx/xrt/setup.sh
source /tools/Xilinx/Vitis/<version>/settings64.sh
make all TARGET=sw_emu DEVICE=xilinx_u50_gen3x16_xdma_5_202210_1
make all TARGET=hw_emu DEVICE=xilinx_u50_gen3x16_xdma_5_202210_1
make all TARGET=hw     DEVICE=xilinx_u50_gen3x16_xdma_5_202210_1
```

## 8. Run

### CPU-only inference

```bash
./build/main_cpu \
  --checkpoint models/stories15M.bin \
  --tokenizer  models/tokenizer.bin \
  --prompt "Once upon a time" \
  --steps 16 --temperature 0.0
```

### Hybrid (FPGA LM head)

```bash
./build/main_hybrid_lmhead \
  --xclbin build/gemv_int8.hw.xclbin \
  --checkpoint models/stories15M.bin \
  --tokenizer  models/tokenizer.bin \
  --prompt "Once upon a time" \
  --steps 16 --temperature 0.0 \
  --use-fpga-lm-head --verbose
```

### Validation-only (one-shot 3-way LM-head compare)

```bash
./build/main_hybrid_lmhead \
  --xclbin build/gemv_int8.hw.xclbin \
  --checkpoint models/stories15M.bin \
  --tokenizer  models/tokenizer.bin \
  --validate-lm-head-only --verbose
```

### Software emulation

```bash
make emconfig TARGET=sw_emu DEVICE=...
cp build/emconfig.json .
bash scripts/run_hybrid_lmhead_sw_emu.sh
```

### Hardware emulation

```bash
make emconfig TARGET=hw_emu DEVICE=...
cp build/emconfig.json .
bash scripts/run_hybrid_lmhead_hw_emu.sh
```

### Real U50

```bash
bash scripts/run_hybrid_lmhead_hw.sh
```

## 9. Vitis IDE

See `vitis/README.md` and `vitis/debug_notes.md`. Short version: this
is a Makefile-driven project; import it as "Existing Makefile Project"
into Vitis Unified IDE, then point the debugger at
`build/main_hybrid_lmhead` using the example launch JSON files in
`vitis/launch_*_example.json`.

## 10. Expected output (validation mode)

```
Milestone 2: U50 FPGA LM Head
  checkpoint: models/stories15M.bin
  tokenizer : models/tokenizer.bin
  xclbin    : build/gemv_int8.hw.xclbin
Model config:
  dim         : 288
  hidden_dim  : 768
  n_layers    : 6
  n_heads     : 6
  n_kv_heads  : 6
  vocab_size  : 32000
  seq_len     : 256
  shared_cls  : yes

Quantizing LM head...
  rows: 32000
  cols: 288
  quantization: per-row int8
  status: OK

  CPU INT8 vs FPGA INT8: EXACT MATCH
  FP32 vs FPGA dequantized:
    max_abs_error  : 0.0145
    mean_abs_error : 0.0021
    top1 a/b       : 1722 / 1722  (MATCH)
    top5 overlap   : 5 / 5

Result: PASS
```

## 11. Validation logic

| Check                                | Required                              |
|--------------------------------------|---------------------------------------|
| CPU INT8 == FPGA INT8 (int32 buffer) | **bit-exact**                         |
| Top-1 token (FP32 vs FPGA)           | usually matches; logged when it doesn't|
| Top-5 overlap (FP32 vs FPGA)         | reported per token                    |
| max_abs / mean_abs FP32 - FPGA       | reported; warns above `--max-abs-error-warning` |

Failure of the bit-exact check is a *correctness* bug (we look at it).
Logit drift between FP32 and INT8 is expected; we report it so we
notice when it gets worse.

## 12. Troubleshooting

| Symptom                                                                  | Fix                                                |
|--------------------------------------------------------------------------|----------------------------------------------------|
| `XILINX_XRT not set`                                                     | `source /opt/xilinx/xrt/setup.sh`                  |
| `v++: command not found`                                                 | `source /tools/Xilinx/Vitis/<ver>/settings64.sh`   |
| `ERROR: DEVICE empty`                                                    | `make ... DEVICE=xilinx_u50_gen3x16_xdma_5_202210_1` |
| `xclbin not found`                                                       | Build the right target first                       |
| `load_xclbin(...) failed`                                                | Platform shell mismatch -- `xbutil examine`        |
| `xrt::kernel("gemv_int8") failed`                                        | xclbin missing CU -- check `nk=` in u50_hbm.cfg    |
| HBM connectivity error in v++ link                                       | port names don't match HLS bundles                 |
| `cannot open checkpoint`                                                 | Wrong path or file missing -- see `models/README.md` |
| `cannot open tokenizer`                                                  | Same; place `tokenizer.bin` next to the checkpoint |
| `Missing build/emconfig.json` (emulation only)                           | `make emconfig TARGET=<...> DEVICE=<...>; cp build/emconfig.json .` |
| `Permission denied` on `/dev/dri/renderD*`                               | `sudo usermod -aG render,xrt $USER` and re-login   |
| FP32 vs FPGA top-1 *frequently* differs                                  | Per-row symmetric INT8 may be too coarse for this  |
|                                                                          | model -- see `docs/quantization.md`                |

## 13. Next milestones

See `docs/next_steps.md`. Short list:

1. Layer-0 Wq on FPGA.
2. Wk and Wv.
3. Fused QKV kernel.
4. MLP on FPGA.
5. INT4 packed weights.
6. 512-bit packed loads.
7. HBM sharding across `HBM[0..7]`.
8. Multiple compute units.
9. Full tiny transformer block.
