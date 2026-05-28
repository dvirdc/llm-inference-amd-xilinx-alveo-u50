#!/usr/bin/env bash
# copy_from_milestone1.sh -- copies the four files this milestone reuses
# from the existing Milestone 1 project.
#
# Usage:
#   MILESTONE1_DIR=../alveo-u50-llm-gemv-milestone1 bash scripts/copy_from_milestone1.sh
# or pass the dir as $1.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

M1="${1:-${MILESTONE1_DIR:-../alveo-u50-llm-gemv-milestone1}}"
if [[ ! -d "$M1" ]]; then
    echo "ERROR: MILESTONE1_DIR='$M1' not found." >&2
    echo "Pass it explicitly: bash $0 /path/to/alveo-u50-llm-gemv-milestone1" >&2
    exit 1
fi

cp "$M1/gemv_kernel/src/gemv_int8.cpp" src/kernels/gemv_int8.cpp
cp "$M1/gemv_host/src/xrt_utils.hpp"   src/host/xrt_utils.hpp
cp "$M1/gemv_host/src/xrt_utils.cpp"   src/host/xrt_utils.cpp
cp "$M1/gemv_common/src/timer.hpp"     src/host/timer.hpp
cp "$M1/gemv/hw_link/gemv-link.cfg"    configs/u50_hbm.cfg

echo "Copied from $M1 :"
echo "  - src/kernels/gemv_int8.cpp"
echo "  - src/host/xrt_utils.{hpp,cpp}"
echo "  - src/host/timer.hpp"
echo "  - configs/u50_hbm.cfg"
