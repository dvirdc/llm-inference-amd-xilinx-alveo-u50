#!/usr/bin/env bash
# run_cpu.sh -- generate text with the CPU-only inference path.
# Honors PROMPT, STEPS, TEMP env vars. Defaults are good for stories15M.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PROMPT="${PROMPT:-Once upon a time}"
STEPS="${STEPS:-32}"
TEMP="${TEMP:-0.0}"
SEED="${SEED:-0}"

./build/main_cpu \
    --checkpoint models/stories15M.bin \
    --tokenizer  models/tokenizer.bin \
    --prompt "$PROMPT" --steps "$STEPS" --temperature "$TEMP" --seed "$SEED" \
    --verbose
