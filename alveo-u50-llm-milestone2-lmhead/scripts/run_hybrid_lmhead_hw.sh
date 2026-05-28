#!/usr/bin/env bash
# run_hybrid_lmhead_hw.sh -- run the hybrid LM-head against real U50 hardware.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
[[ -n "${XILINX_XRT:-}" ]] || { echo "source /opt/xilinx/xrt/setup.sh first"; exit 1; }
unset XCL_EMULATION_MODE || true

./build/main_hybrid_lmhead \
    --xclbin build/gemv_int8.hw.xclbin \
    --checkpoint models/stories15M.bin \
    --tokenizer  models/tokenizer.bin \
    --prompt "${PROMPT:-Once upon a time}" \
    --steps "${STEPS:-32}" --temperature 0.0 \
    --use-fpga-lm-head --verbose
