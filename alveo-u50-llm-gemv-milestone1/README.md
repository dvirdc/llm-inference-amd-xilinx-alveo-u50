# alveo-u50-llm-gemv-milestone1

A clean, production-quality starter project that runs **INT8 matrix-vector
multiplication (GEMV)** on an **AMD/Xilinx Alveo U50** FPGA via XRT, and
compares the result against a CPU reference.

This is **milestone 1** of a longer effort to build a transformer-style LLM
inference engine on the U50. GEMV is the right starting point because the
hot loop of batch-1 LLM decode is dominated by GEMVs (Q/K/V projections,
output projection, MLP projections, LM head). Get GEMV right first, then
extend.

The repo is laid out as a **native Vitis IDE workspace** that mirrors the
working `vadd` example: one system project (`gemv/`) plus three components
(`gemv_host/`, `gemv_kernel/`, `gemv_common/`).

---

## 1. What this milestone does

- One **HLS kernel**, `gemv_int8`, that computes `y[row] = sum_col W[row,col] * x[col]`
  with INT8 inputs and INT32 accumulation.
- HBM connectivity for U50 (weights / x / y on separate HBM channels).
- A C++ host (`xrt::device/xrt::kernel/xrt::bo`) that:
  - generates deterministic INT8 test data,
  - runs a CPU reference,
  - copies buffers to the device, runs the kernel, copies them back,
  - bit-exact compares device output against the reference,
  - prints per-stage timing and `PASS` / `FAIL`.
- Two CPU-only unit tests (no FPGA needed):
  - `tests/test_gemv_cpu.cpp` (hand-checked + randomized GEMV)
  - `tests/test_data_generation.cpp` (determinism + range checks)
- Buildable and debuggable directly inside the Vitis IDE.

## 2. What this milestone does NOT do

- No 512-bit vectorized loads, no inner-loop unrolling.
- No HBM sharding across `HBM[0..7]`.
- No multiple compute units.
- No INT4 packed weights.
- No RMSNorm, no Q/K/V projection, no RoPE, no KV cache, no attention.
- No quantization-aware scales / zero-points (yet -- ranges are tight enough
  that pure integer math is bit-exact).

See [`docs/next_steps.md`](docs/next_steps.md) for the staged roadmap.

---

## 3. Repo layout

```
alveo-u50-llm-gemv-milestone1/        <-- open THIS folder in Vitis IDE
├── gemv/                              # System project (Vitis IDE: System)
│   ├── vitis-sys.json
│   ├── CMakeLists.txt
│   └── hw_link/
│       ├── CMakeLists.txt
│       └── gemv-link.cfg              # v++ link config: nk= and sp= for HBM
├── gemv_host/                         # Host component (Vitis IDE: HOST)
│   ├── vitis-comp.json
│   ├── CMakeLists.txt
│   ├── UserConfig.cmake
│   └── src/
│       ├── main.cpp
│       ├── xrt_utils.{hpp,cpp}
│       ├── gemv_cpu.cpp
│       └── test_data.cpp
├── gemv_kernel/                       # HLS kernel component (Vitis IDE: HLS)
│   ├── vitis-comp.json
│   ├── hls_config.cfg
│   └── src/
│       └── gemv_int8.cpp
├── gemv_common/                       # Shared headers
│   └── src/
│       ├── gemv_cpu.hpp
│       ├── test_data.hpp
│       └── timer.hpp
├── tests/
│   ├── Makefile                       # CPU-only unit tests
│   ├── test_gemv_cpu.cpp
│   └── test_data_generation.cpp
├── scripts/
│   └── env_check.sh
├── docs/
│   ├── architecture.md
│   └── next_steps.md
├── xrt.ini                            # XRT runtime config (auto-loaded)
└── README.md
```

---

## 4. Prerequisites

- Alveo U50 visible to `xbutil examine`.
- AMD/Xilinx Vitis (tuned against `xilinx_u50_gen3x16_xdma_5_202210_1`, but
  any U50 platform works -- adjust the `platform` field in
  `gemv/vitis-sys.json` and `gemv_host/vitis-comp.json` if needed).
- XRT 2021.2+ (provides the modern `xrt::*` native C++ API).
- A C++17 compiler (only needed for the CPU-only tests in `tests/`).

> The CPU-only tests do **not** require Vitis or XRT.

---

## 5. Environment setup

```bash
source /opt/xilinx/xrt/setup.sh
source /tools/Xilinx/Vitis/2022.2/settings64.sh   # adjust path/version

bash scripts/env_check.sh                          # sanity
```

`scripts/env_check.sh` prints which env vars are set, which tools are on
`PATH`, the installed platforms, and the devices XRT sees.

To confirm your U50 platform name:

```bash
platforminfo -l
xbutil examine
```

---

## 6. Open in Vitis IDE

```bash
# From a shell that has the Vitis env sourced:
cd /home/dvirdc/vitis_projects/alveo-u50-llm-gemv-milestone1
vitis -w .
# or just: vitis
# then File > Open Workspace > select this folder
```

Vitis IDE will detect the four components automatically:

| Component | Type | Source dir | Build output |
|-----------|------|------------|--------------|
| `gemv_kernel` | HLS | `gemv_kernel/src` | `gemv_kernel/gemv_kernel/gemv_int8.xo` |
| `gemv_host`   | HOST | `gemv_host/src` | `gemv_host/build/.../gemv_host` (executable) |
| `gemv`        | System | --- | `gemv/build/.../gemv.xclbin` |
| `gemv_common` | (headers only) | `gemv_common/src` | --- |

## 7. Build (inside Vitis IDE)

The IDE's **Flow Navigator** in the bottom-left drives the build:

1. Select target: **HW_EMU** (fast, ~5-20 min) or **HW** (full P&R, 1-4 hr).
2. Build **`gemv_kernel`** -> produces `gemv_int8.xo`.
3. Build **`gemv`** (system) -> v++ link with `hw_link/gemv-link.cfg`
   -> produces `gemv.xclbin`.
4. Build **`gemv_host`** -> produces the host executable.

You can also build everything from the system project's context menu
("Build" on `gemv`).

> The system project supports HW_EMU and HW. SW_EMU is not enabled for
> this platform (matches the working vadd setup).

## 8. Run / debug (inside Vitis IDE)

1. Right-click `gemv_host` > **Run As** (or **Debug As**) > **Vitis Host
   Application**.
2. In the launch dialog, set program args, e.g.:
   ```
   --xclbin ${workspace_loc}/gemv/build/hw_emu/gemv.xclbin --rows 128 --cols 128 --verbose
   ```
   (For HW runs swap `hw_emu` for `hw`.)
3. The IDE auto-sets `XCL_EMULATION_MODE` for emulation runs.
4. **Debug**: set breakpoints in `main.cpp` / `gemv_cpu.cpp` /
   `xrt_utils.cpp` and step through. The IDE attaches `gdb` to the host
   process.

> For **kernel** debugging, build with `--target hw_emu` and use the
> **Vitis Analyzer** (opens automatically after a run) to inspect
> waveforms, timeline, and connectivity.

## 9. Build / run from the command line

The Vitis IDE just drives CMake. You can do the same by hand:

```bash
# kernel .xo
cd gemv_kernel
v++ --target hw_emu --platform xilinx_u50_gen3x16_xdma_5_202210_1 \
    --config hls_config.cfg --mode hls -c -o gemv_kernel/gemv_int8.xo \
    src/gemv_int8.cpp

# link xclbin
cd ../gemv
cmake -B build/hw_emu -S . -DVITIS_TARGET=hw_emu \
      -DWORKSPACE_DIR=/home/dvirdc/vitis_projects/alveo-u50-llm-gemv-milestone1
cmake --build build/hw_emu

# host
cd ../gemv_host
cmake -B build -S .
cmake --build build
```

(The IDE does these for you; this is just the equivalent CLI.)

---

## 10. CPU-only unit tests

These don't need Vitis or XRT and are the fastest way to catch logic bugs.

```bash
make -C tests           # build + run both test executables
make -C tests clean
```

Expected output: `passed=N failed=0` for both suites.

---

## 11. Expected host output

```
GEMV INT8 test
  rows         : 1024
  cols         : 1024
  seed         : 0xC0FFEEAA
  xclbin       : .../gemv.xclbin
  device index : 0

--- Timing ---
  CPU reference              :    XX.XXX ms
  H2D transfer               :     X.XXX ms
  Kernel execution           :     X.XXX ms
  D2H transfer               :     X.XXX ms
  Total FPGA path            :     X.XXX ms

Result: PASS
```

Exit code is `0` on PASS, `1` on FAIL (with bit-exact mismatch details),
`2` on bad CLI args, `3` on any other fatal error.

---

## 12. Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `XILINX_XRT is not set` | XRT setup not sourced | `source /opt/xilinx/xrt/setup.sh` |
| `v++: command not found` | Vitis not sourced | `source /tools/Xilinx/Vitis/<ver>/settings64.sh` |
| `Platform not found` in Vitis IDE | platform name mismatch | Update `platform` in `gemv/vitis-sys.json` and `gemv_host/vitis-comp.json` |
| `load_xclbin(...) failed` | xclbin / shell mismatch | Confirm `xbutil examine` shows the same platform you built for |
| `xrt::kernel(gemv_int8) failed` | CU not in xclbin | Inspect `xclbinutil --info -i ...xclbin`; check `nk=` in `gemv/hw_link/gemv-link.cfg` |
| HBM connectivity error in v++ link | port names don't match HLS bundles | Inspect `gemv_kernel/gemv_kernel/.../kernel.xml` for actual port names |
| `Permission denied` opening `/dev/dri/renderD*` or `/dev/xclmgmt*` | user not in `xrt` group | `sudo usermod -aG render,xrt $USER` and re-login |
| 0 devices found on boot | Thunderbolt hotplug race (U50 over TB) | Replug the TB cable; this is a known machine-local quirk |
| `_ide/` keeps regenerating | Vitis IDE working files | This is normal; `_ide/logs` and `_ide/.wsdata` are git-ignored |

---

## 13. How this maps to LLM inference

For batch-1 (single-prompt) token decoding -- the case that matters most for
chat-style latency -- the dominant operation is **GEMV**, not GEMM:

- **Q, K, V projections**: each is `x @ W` with `x` of shape `[d_model]`
  and `W` of shape `[d_model, d_head*n_heads]`. That's a GEMV.
- **Output projection**: another GEMV.
- **MLP up- and down-projections**: GEMVs (with an activation in between).
- **LM head** (final logits): GEMV against the embedding matrix.
- **Attention**: for batch-1, `softmax(Q @ K^T / sqrt(d)) @ V` is itself
  two GEMVs over the KV cache.

Why **memory-bound** matters: batch-1 GEMV reads the *entire weight matrix
once per token*. For a 7B-parameter model that's ~7 GB of weights touched
per token. The U50's HBM gives ~316 GB/s peak -- so the theoretical
ceiling is ~45 INT8 tokens/sec, ~90 INT4 tokens/sec. Approaching those
numbers is exactly what later milestones target.

---

## 14. Next optimization steps

See [`docs/next_steps.md`](docs/next_steps.md). Short list:

1. **512-bit vectorized reads** with `ap_uint<512>` -- 64 INT8 lanes per
   AXI beat instead of 1.
2. **Inner-loop unrolling** + DSP packing.
3. **Multi-HBM sharding** (`HBM[0..7]` for weights).
4. **Multiple compute units** for row-parallel GEMV.
5. **INT4 packed weights** to halve memory bandwidth requirements.
6. **Fused RMSNorm + GEMV**.
7. **KV cache attention**.
