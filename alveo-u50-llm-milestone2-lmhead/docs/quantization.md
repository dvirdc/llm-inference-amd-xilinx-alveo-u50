# quantization.md

## Three logits paths

For every token we have three possible LM-head paths:

| Path           | Math                                              | Where                          |
|----------------|---------------------------------------------------|--------------------------------|
| FP32 reference | y = W @ x   (all float32)                         | CPU                            |
| CPU INT8       | y_int32 = qW @ qx; y_fp32 = y_int32 * sw * sx     | CPU (`gemv_cpu_int8`)          |
| FPGA INT8      | y_int32 = qW @ qx; y_fp32 = y_int32 * sw * sx     | FPGA (`gemv_int8` kernel)      |

The CPU INT8 and FPGA INT8 paths share the *same quantized inputs* and
the *same integer accumulation rule*. The kernel and the host's
`gemv_cpu_int8` both compute:

```
y_int32[r] = sum_{c=0..cols-1} int32(qW[r,c]) * int32(qx[c])
```

with no rounding, no FP detour. That's why we require **bit-exact**
equality between CPU INT8 and FPGA INT8 int32 outputs.

## Quantization formulas

Per-row symmetric INT8 for weights:

```
for each row r:
    amax     = max_c |W[r,c]|
    if amax == 0: scale=1, all qW[r,*] = 0
    else:
        scale  = amax / 127
        qW[r,c] = clamp_to_127(round(W[r,c] / scale))
    row_scales[r] = scale
```

Dynamic symmetric INT8 for activations (run every token):

```
amax = max_i |x[i]|
if amax == 0: scale=1
else:        scale = amax / 127
qx[i] = clamp_to_127(round(x[i] / scale))
```

Dequantize (one multiply per output row, FP32):

```
y_fp32[r] = y_int32[r] * row_scales[r] * x_scale
```

## Why pin to ±127 instead of -128..+127

Asymmetric INT8 has a one-extra-bin negative range (-128), which makes
`round(x/scale)` produce a tiny bias toward negative values. Symmetric
clamping at ±127 eliminates that. Standard practice (TensorRT, ORT).

## Why CPU INT8 == FPGA INT8 bit-exactly

* Both code paths do `int32 += int8 * int8`. The product of two int8s
  fits in int16; the sum across `cols` cannot overflow int32 for our
  sizes (bound: cols * 127 * 127 = ~33M at cols=2048).
* Integer addition is associative, so any inner-loop schedule
  (sequential, unrolled, vectorized) produces the same int32.
* No FP arithmetic happens until we dequantize -- and we compare
  *before* that step.

If this ever fails: it's a layout bug (the host and the kernel
disagree on row-major vs column-major) or a buffer-mapping bug (we sent
the wrong HBM bank). Look there first.

## Why FP32 reference != INT8 (and that's OK)

Quantization is lossy. Per-row symmetric INT8 against weights drawn
from N(0, σ²) loses about 1% relative error in the dot product.
For LM-head logits this almost never changes the top-1 token, but
absolute logit values can differ noticeably -- the report shows
max_abs_error and mean_abs_error so you can track this.

The success criterion we use:
* Bit-exact CPU INT8 vs FPGA INT8.
* Top-1 token usually agrees between FP32 and dequantized FPGA.
* Top-5 overlap typically 4 or 5 of 5.

If top-1 *frequently* disagrees, the wcls weights have a high dynamic
range that per-row symmetric INT8 can't capture cleanly -- a future
milestone will switch to per-row INT8 with INT4 outliers or block-wise
quantization.
