# vitis/

Documentation for using **Vitis Unified IDE** with this project.

This project now supports TWO build flows side by side. Pick whichever
fits the task -- they read the same source tree.

## Flow A -- Vitis IDE (native components)

The project root contains three Vitis IDE components that the IDE
auto-detects when you open the workspace:

| Path             | Type       | What it builds                                  |
|------------------|------------|-------------------------------------------------|
| `lmhead/`        | System     | Links the kernel + cfg into `lmhead.xclbin`     |
| `lmhead_host/`   | HOST       | Builds the `lmhead_host` executable             |
|                  |            | (= `main_hybrid_lmhead` linked with all of llama/ + fpga/)|
| `lmhead_kernel/` | HLS kernel | Compiles `gemv_int8.cpp` -> `gemv_int8.xo`      |

The component dirs are *thin wrappers*: their CMake / hls_config files
point back at the existing `../src/` tree, so there is **no source
duplication**. Edit a `.cpp` once and both flows pick it up.

### Opening the workspace

```bash
source /opt/xilinx/xrt/setup.sh
source /tools/Xilinx/Vitis/<version>/settings64.sh
cd /home/dvirdc/vitis_projects/alveo-u50-llm-milestone2-lmhead
vitis -w .
```

Vitis IDE recognises the three components automatically. In the Flow
Navigator (bottom-left):

1. Pick target: **HW_EMU** (fast smoke test) or **HW** (real card).
2. Build `lmhead_kernel`  -> `lmhead_kernel/lmhead_kernel/gemv_int8.xo`
3. Build `lmhead` (system) -> `lmhead/build/<target>/hw_link/lmhead.xclbin`
4. Build `lmhead_host`     -> `lmhead_host/build/lmhead_host`

Or build everything in one go from the `lmhead` system project context menu.

### Running and debugging in the IDE

Four pre-baked launch configurations live in
[`lmhead/_ide/launch.json`](../lmhead/_ide/launch.json):

| Name                          | Args                                                  | XCL_EMULATION_MODE  |
|-------------------------------|-------------------------------------------------------|---------------------|
| `lmhead_hw_emu_validate_only` | `--validate-lm-head-only`                             | `hw_emu`            |
| `lmhead_hw_emu_generate`      | `--prompt "Once upon a time" --steps 4 --use-fpga...` | `hw_emu`            |
| `lmhead_hw_validate_only`     | `--validate-lm-head-only`                             | unset (real card)   |
| `lmhead_hw_generate`          | `--prompt "Once upon a time" --steps 32 --use-fpga...`| unset (real card)   |

They each set `XRT_INI_PATH` to point at
`lmhead/lmhead_host/runtime/hw_emu_xrt.ini` or `hw_xrt.ini` so per-mode
tracing settings are picked up correctly.

To launch: Run -> Debug Configurations -> pick one -> Debug. Set
breakpoints in any file under `src/apps/`, `src/llama/`, `src/fpga/`,
`src/host/`. See `debug_notes.md` for what is and isn't debuggable
inside the kernel.

## Flow B -- Makefile (CLI)

Still fully supported. Same source tree. Build artifacts land under
`build/` at the project root rather than per-component `build/` dirs.

```bash
make test
make all TARGET=sw_emu DEVICE=xilinx_u50_gen3x16_xdma_5_202210_1
make all TARGET=hw_emu DEVICE=xilinx_u50_gen3x16_xdma_5_202210_1
make all TARGET=hw     DEVICE=xilinx_u50_gen3x16_xdma_5_202210_1

./build/main_hybrid_lmhead --xclbin build/gemv_int8.hw.xclbin ...
```

The Makefile flow builds *all four* apps (`main_cpu`,
`main_hybrid_lmhead`, `inspect_model`, `quantize_lm_head`). The IDE
component only builds `main_hybrid_lmhead` because that's the one you
actually debug against the FPGA. The others are short CLI utilities --
use them from the Makefile or `./build/...`.

## Files in this folder

| File                         | What it is                                          |
|------------------------------|-----------------------------------------------------|
| `README.md`                  | This file                                           |
| `debug_notes.md`             | What you can and can't debug, common gotchas        |
| `launch_sw_emu_example.json` | Reference launch config for Flow B + sw_emu         |
| `launch_hw_emu_example.json` | Reference launch config for Flow B + hw_emu         |
| `launch_hw_example.json`     | Reference launch config for Flow B + real hardware  |

The `launch_*_example.json` files predate Flow A and document the
generic launch fields. Flow A users get a working launch.json
out of the box at `lmhead/_ide/launch.json` -- you don't need to copy
the example files unless you want to drive xgdb manually.

## Environment

In every Vitis IDE configuration the following must be set (the IDE
inherits them from the shell that started Vitis, so source the setup
scripts before running `vitis`):

```
XILINX_XRT=/opt/xilinx/xrt
XILINX_VITIS=/tools/Xilinx/Vitis/<version>
LD_LIBRARY_PATH=$XILINX_XRT/lib:$LD_LIBRARY_PATH
```

For emulation runs `XCL_EMULATION_MODE` is set automatically by the
launch configs in `lmhead/_ide/launch.json`.
