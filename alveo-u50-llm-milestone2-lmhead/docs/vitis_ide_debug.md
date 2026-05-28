# vitis_ide_debug.md

This project is Makefile-driven and CLI-buildable. Vitis IDE is *optional*,
useful when you want graphical breakpoints in the host code or you want
Vitis Analyzer to visualize the timeline of a hw_emu / hw run.

## Two ways to use it

### A) Workspace import (Vitis Unified IDE, 2023+)

1. Vitis -> File -> Open Workspace... -> select this directory.
2. The IDE will see no `vitis-comp.json` so it won't auto-detect this
   as a component-style workspace. That's fine.
3. Vitis -> File -> Import -> *Existing Makefile Project* -> point at
   this directory, set "Make target" to `host` (or `all TARGET=hw_emu DEVICE=...`).
4. The IDE will treat the Makefile as opaque and just `make` it.

### B) Plain `vitis_hls` + external debugger

If you only want host-side gdb/xgdb:

```
source /opt/xilinx/xrt/setup.sh
source /tools/Xilinx/Vitis/<ver>/settings64.sh
make all TARGET=hw_emu DEVICE=xilinx_u50_gen3x16_xdma_5_202210_1
make emconfig TARGET=hw_emu DEVICE=xilinx_u50_gen3x16_xdma_5_202210_1
cp build/emconfig.json .
XCL_EMULATION_MODE=hw_emu xgdb --args ./build/main_hybrid_lmhead \
   --xclbin build/gemv_int8.hw_emu.xclbin \
   --checkpoint models/stories15M.bin \
   --tokenizer  models/tokenizer.bin \
   --validate-lm-head-only --verbose
```

`xgdb` is just `gdb` with the Xilinx kernel-debug extensions registered.
Set breakpoints in `src/apps/main_hybrid_lmhead.cpp` or
`src/fpga/fpga_gemv_engine.cpp` as you would for any C++ program.

## Required environment variables

Set BEFORE launching Vitis or the debugger:

| Var                    | Why                                                       |
|------------------------|-----------------------------------------------------------|
| `XILINX_XRT`           | Lets the linker find libxrt_coreutil.so.2                 |
| `XILINX_VITIS`         | Lets the IDE find xgdb, Vitis Analyzer, simulators        |
| `PLATFORM_REPO_PATHS`  | If your platform XPFM lives in a non-default location     |
| `XCL_EMULATION_MODE`   | `sw_emu` / `hw_emu` for emulation; *unset* for real card  |

The launch JSON files in `vitis/launch_*_example.json` set
`XCL_EMULATION_MODE` correctly per target -- copy-paste them into the IDE.

## What you CAN'T debug like normal CPU code

The HLS kernel C++ source (`src/kernels/gemv_int8.cpp`) is *not* CPU
code at run time. Setting a breakpoint inside it from the host
debugger will not do what you expect.

For kernel-side debugging:

* **sw_emu**: `cosim` runs the kernel as plain C++ in the host process,
  so a host gdb breakpoint *does* work. This is the fastest dev loop.
* **hw_emu**: the kernel is compiled to RTL and runs in a Vivado
  simulator. Set waveform probes via `xrt.ini` `[Debug] device_trace`
  and open the `.run_summary` in Vitis Analyzer afterwards. No live
  stepping.
* **hw**: no debugger access into the kernel. You're limited to:
  - `--compare-every-token` to detect divergence vs the CPU INT8 path.
  - Vitis Analyzer trace if you enabled `device_trace=coarse`.
  - Adding printf to the *host* before/after the kernel call.

When something fails on hw but works on hw_emu, the usual culprits
(in this order) are: HBM connectivity (`u50_hbm.cfg`), platform shell
mismatch, or kernel frequency closure failure.

## Useful runtime knobs (xrt.ini)

* `[Debug] host_trace=true`        -- per-API host log, low overhead
* `[Debug] device_trace=coarse`    -- CU-level timeline, ~5% overhead
* `[Debug] stall_trace=coarse`     -- adds memory stall counters
* `[Debug] native_xrt_trace=true`  -- writes `xclbin.run_summary` for Analyzer

The shipped `xrt.ini` keeps these *off* so per-token timings reported
by `main_hybrid_lmhead` aren't inflated. Turn them on when investigating.
