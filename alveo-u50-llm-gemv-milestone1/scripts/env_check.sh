#!/usr/bin/env bash
# env_check.sh -- quick sanity check for the Vitis + XRT environment.
#
# Run with:   bash scripts/env_check.sh

set -u

echo "==> Environment variables"
for v in XILINX_XRT XILINX_VITIS XILINX_VIVADO XILINX_HLS; do
    val="${!v:-}"
    if [[ -z "$val" ]]; then
        echo "  $v: (not set)"
    else
        echo "  $v: $val"
    fi
done

echo ""
echo "==> Tool versions"
for tool in v++ vitis xrt-smi xbutil platforminfo emconfigutil; do
    if command -v "$tool" >/dev/null 2>&1; then
        path=$(command -v "$tool")
        ver=$("$tool" --version 2>&1 | head -n 1 || true)
        printf "  %-15s OK  %s   (%s)\n" "$tool" "$path" "$ver"
    else
        printf "  %-15s MISSING\n" "$tool"
    fi
done

echo ""
echo "==> Installed Vitis platforms"
if command -v platforminfo >/dev/null 2>&1; then
    platforminfo -l 2>/dev/null | sed 's/^/  /'
else
    echo "  platforminfo not on PATH -- source /tools/Xilinx/Vitis/<ver>/settings64.sh first"
fi

echo ""
echo "==> Devices visible to XRT"
if command -v xbutil >/dev/null 2>&1; then
    xbutil examine 2>/dev/null | sed 's/^/  /'
else
    echo "  xbutil not on PATH -- source /opt/xilinx/xrt/setup.sh first"
fi

echo ""
echo "If anything above is MISSING, source the right setup scripts:"
echo "  source /opt/xilinx/xrt/setup.sh"
echo "  source /tools/Xilinx/Vitis/<version>/settings64.sh"
