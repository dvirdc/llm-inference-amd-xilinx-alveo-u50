// fpga_gemv_engine.hpp -- thin wrapper around the Milestone 1 gemv_int8
// kernel + XRT host code. The model code talks to this class, not to XRT
// directly. That gives us two important properties:
//
//   1. Llama inference code stays free of XRT types -- you can compile
//      apps/main_cpu without XRT installed.
//   2. When later milestones split GEMV across multiple CUs or move more
//      weights onto the device, only this file changes.
//
// Lifecycle:
//   * Construct once per process: opens the device, loads xclbin.
//   * `load_weight(name, qm)` copies a quantized matrix into a
//     device-resident HBM buffer and keeps it there. We pay this cost
//     once per program run, not per token.
//   * Per token: `run_int8` (or `run_dequantized`) -- sends the small
//     activation vector across PCIe, runs the kernel, fetches the small
//     output vector back. The big weight matrix stays put.
//
// The "sticky weights, moving activations" pattern is critical to ever
// being faster than CPU: PCIe is the slow link, and a 7B-param model's
// weights are ~7 GB -- you cannot move that per token.

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "../llama/quantization.hpp"

// Forward-declare XRT handles so this header itself is XRT-free. The
// implementation file includes the XRT headers.
namespace xrt { class device; class kernel; class bo; class uuid; }

namespace m2 {

class FpgaGemvEngine {
public:
    // Opens device #device_index and loads xclbin. Throws m2::FpgaError
    // if the device cannot be opened or the xclbin doesn't contain a
    // kernel named "gemv_int8".
    FpgaGemvEngine(const std::string& xclbin_path, int device_index = 0);
    ~FpgaGemvEngine();

    FpgaGemvEngine(const FpgaGemvEngine&)            = delete;
    FpgaGemvEngine& operator=(const FpgaGemvEngine&) = delete;

    // Upload a quantized matrix and keep it resident on the FPGA. Use
    // `name` later to refer to this weight. Calling load_weight twice
    // with the same name replaces the previous buffer.
    void load_weight(const std::string& name, const QuantizedMatrix& matrix);

    // y_int32 = W @ x  (INT32 accumulator, exactly like the kernel does).
    // Per-row scales are NOT applied here -- this is the path used for
    // bit-exact comparison with CPU INT8.
    void run_int8(const std::string&     weight_name,
                  const QuantizedVector& x,
                  std::vector<int32_t>&  y_int32);

    // Same as run_int8 but multiplies through the saved row_scales and
    // the provided x.scale to produce FP32 logits. Convenience for the
    // hybrid LM-head path.
    void run_dequantized(const std::string&     weight_name,
                         const QuantizedVector& x,
                         std::vector<float>&    y_fp32);

    // Test helpers / introspection.
    bool has_weight(const std::string& name) const;
    int  rows_of(const std::string& name) const;
    int  cols_of(const std::string& name) const;

private:
    // PImpl so the header stays free of XRT types.
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace m2
