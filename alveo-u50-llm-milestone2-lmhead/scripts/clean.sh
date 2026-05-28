#!/usr/bin/env bash
# clean.sh -- wipe build artifacts. Convenience wrapper around `make clean`.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
make clean
rm -rf .Xil _x v++_* xrt.log emconfig.json *.compile_summary *.link_summary *.run_summary
echo "Clean done."
