#!/usr/bin/env bash
# fetch_karpathy_llama2c.sh -- clones Karpathy's llama2.c into
# third_party/llama2.c for reference. We use it for:
#   * the run.c implementation as a comparison point,
#   * its export.py / tokenizer.py if you want to convert your own
#     PyTorch model to a llama2.c checkpoint,
#   * its README for model download links (stories15M.bin etc.).

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DEST="third_party/llama2.c"
if [[ -d "$DEST/.git" ]]; then
    echo "Already cloned at $DEST. Pulling latest..."
    git -C "$DEST" pull --ff-only || true
else
    rm -rf "$DEST"
    git clone --depth=1 https://github.com/karpathy/llama2.c "$DEST"
fi

echo ""
echo "Done. Next:"
echo "  * See $DEST/README.md for stories15M.bin download URL."
echo "  * Place stories15M.bin and tokenizer.bin in $ROOT/models/."
