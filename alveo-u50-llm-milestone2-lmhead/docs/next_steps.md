# next_steps.md

The end-state is to run a complete tiny transformer block on the FPGA.
We get there one bounded milestone at a time so we can keep CPU
reference checks at every step.

## Milestone 2 -- FPGA LM head  *(this repo)*

* One CU (the Milestone 1 `gemv_int8`).
* One resident weight matrix (`wcls`).
* Goal: end-to-end token generation with FPGA-accelerated LM head.
* Speed: not faster than CPU yet -- this is the correctness scaffold.

## Milestone 3 -- Layer-0 Wq on FPGA

* Add a *second* resident weight to the FpgaGemvEngine: layer-0 Q proj.
* Replace the CPU Q-proj call in `llama_forward_cpu.cpp` with an engine
  call only when `--use-fpga-l0-wq` is set.
* Same correctness gates as Milestone 2 (CPU INT8 vs FPGA INT8 exact).

## Milestone 4 -- Wk and Wv (separately)

* Three resident weights now (Wq, Wk, Wv for layer 0). Three engine
  calls per token. PCIe still dwarfs compute -- not faster yet.

## Milestone 5 -- Fused QKV kernel

* New kernel: writes Q, K, V from one read of `x`. Saves two PCIe
  transfers of the activation per token. This is where speed starts
  to matter.

## Milestone 6 -- MLP (W1, W2, W3) on FPGA

* Either three separate kernels or one fused SwiGLU kernel.
* The MLP block accounts for ~2/3 of FLOPs in modern Llama; this is
  the moment the FPGA path likely overtakes the CPU.

## Milestone 7 -- INT4 packed weights

* Halve the bytes/token. Re-do the bit-exact gate with the int4 dequant
  rule rather than int8.

## Milestone 8 -- HBM sharding + multi-CU

* Spread `wcls` across `HBM[0..7]` with 8 m_axi ports. Add a second
  `gemv_int8` CU. Bandwidth scaling.

## Milestone 9 -- Full tiny transformer block on FPGA

* RMSNorm + QKV + RoPE + attention + Wo + RMSNorm + MLP in a single
  monolithic kernel. CPU is now the orchestrator only.

## Milestone 10 -- Bigger model + perf tuning

* Move from stories15M (288 dim) to llama2-7b (4096 dim). Measure
  tokens/sec. Hit ~half of HBM peak.
