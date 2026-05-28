# vitis/debug_notes.md

## What you can debug like normal CPU code

* All four `apps/*.cpp` -- step through, set breakpoints, watch variables.
* All `src/llama/*.cpp` -- the CPU transformer, tokenizer, sampler,
  quantizer. These are plain std::vector code.
* `src/fpga/fpga_gemv_engine.cpp` -- the XRT-using wrapper. Breakpoints
  before and after `kernel(bo_w, bo_x, bo_y, rows, cols)` work fine.

## What you CAN'T

* `src/kernels/gemv_int8.cpp` is *FPGA HLS code*. Outside of `sw_emu`
  it does not execute on the CPU at all and a host-side breakpoint
  inside it is a no-op.

| Mode       | Kernel runs as           | Kernel-side debugging                              |
|------------|--------------------------|----------------------------------------------------|
| `sw_emu`   | Plain C++ in host process| ✓ host gdb breakpoints work inside the kernel       |
| `hw_emu`   | RTL in Vivado sim        | Waveform via `device_trace`, no live stepping      |
| `hw`       | Bitstream on the U50     | None. Use `--compare-every-token` + Analyzer trace |

## When something fails on hw but works on hw_emu

Almost always one of three things:

1. **HBM connectivity**: edit `configs/u50_hbm.cfg`, re-link.
2. **Platform shell mismatch**: `xbutil examine` -- the shell flashed on
   the card must match the platform used to build.
3. **Timing closure**: drop `VPP_KERNEL_FREQ` from 250 to 200 and
   rebuild.

## Useful runtime profiling

In `xrt.ini`:

* `[Debug] host_trace=true` -- per-API host log; minimal overhead.
* `[Debug] device_trace=coarse` -- CU-level timeline; opens in
  Vitis Analyzer via `xclbin.run_summary`.
* `[Debug] stall_trace=coarse` -- memory stalls visible in Analyzer.

Be aware: with these enabled, the per-token timings printed by
`main_hybrid_lmhead` include tracing overhead and are not directly
comparable to a "production" run. Turn them off for benchmark numbers.

## Common reasons the binary won't even launch

| Symptom                                              | Fix                                              |
|-----------------------------------------------------|--------------------------------------------------|
| `libxrt_coreutil.so.2: cannot open shared object`    | `source /opt/xilinx/xrt/setup.sh`               |
| `Cannot find emconfig.json`                         | `make emconfig TARGET=<...> DEVICE=<...>` and `cp build/emconfig.json .` |
| `XCL_EMULATION_MODE set but using a real-card flow` | Unset it for `hw` runs                          |
| `xclbin not found: build/...sw_emu.xclbin`          | Build the right target: `make xclbin TARGET=sw_emu DEVICE=...` |
| `Permission denied opening /dev/dri/renderD*`       | `sudo usermod -aG render,xrt $USER` and re-login|
