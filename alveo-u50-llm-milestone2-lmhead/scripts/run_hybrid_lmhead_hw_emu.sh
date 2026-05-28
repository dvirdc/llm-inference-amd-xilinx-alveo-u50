#!/usr/bin/env bash
# run_hybrid_lmhead_hw_emu.sh -- hybrid LM-head under hw_emu.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
[[ -n "${XILINX_XRT:-}" ]] || { echo "source /opt/xilinx/xrt/setup.sh first"; exit 1; }

export XCL_EMULATION_MODE=hw_emu
[[ -f build/emconfig.json ]] || {
    echo "Missing build/emconfig.json. Run: make emconfig TARGET=hw_emu DEVICE=<platform>"
    exit 1
}
cp -f build/emconfig.json ./emconfig.json

./build/main_hybrid_lmhead \
    --xclbin build/gemv_int8.hw_emu.xclbin \
    --checkpoint models/stories15M.bin \
    --tokenizer  models/tokenizer.bin \
    --prompt "${PROMPT:-Once upon a time}" \
    --steps "${STEPS:-4}" --temperature 0.0 \
    --use-fpga-lm-head --compare-every-token --verbose
