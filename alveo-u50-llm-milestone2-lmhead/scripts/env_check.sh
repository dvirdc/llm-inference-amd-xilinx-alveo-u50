#!/usr/bin/env bash
# env_check.sh -- quick sanity check for Vitis + XRT environment.
set -u

echo "==> Environment variables"
for v in XILINX_XRT XILINX_VITIS XILINX_VIVADO XILINX_HLS PLATFORM_REPO_PATHS; do
    val="${!v:-}"
    if [[ -z "$val" ]]; then echo "  $v: (not set)"
    else echo "  $v: $val"; fi
done

echo ""
echo "==> Tool versions"
for tool in v++ vitis xbutil xrt-smi platforminfo emconfigutil; do
    if command -v "$tool" >/dev/null 2>&1; then
        ver=$("$tool" --version 2>&1 | head -n1 || true)
        printf "  %-15s OK  %s\n" "$tool" "$ver"
    else
        printf "  %-15s MISSING\n" "$tool"
    fi
done

echo ""
echo "==> Installed platforms"
if command -v platforminfo >/dev/null 2>&1; then
    platforminfo -l 2>/dev/null | sed 's/^/  /'
else
    echo "  (platforminfo not on PATH)"
fi

echo ""
echo "==> XRT-visible devices"
if command -v xbutil >/dev/null 2>&1; then
    xbutil examine 2>/dev/null | sed 's/^/  /'
else
    echo "  (xbutil not on PATH)"
fi
