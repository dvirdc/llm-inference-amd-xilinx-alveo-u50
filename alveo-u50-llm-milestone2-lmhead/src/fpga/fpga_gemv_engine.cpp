#include "fpga_gemv_engine.hpp"

#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <string>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "../common/errors.hpp"
#include "../common/logging.hpp"

namespace m2 {

// ---------------------------------------------------------------------------
// Pimpl: keep all XRT state here so the header stays XRT-free.
// ---------------------------------------------------------------------------
struct FpgaGemvEngine::Impl {
    // One device + one kernel handle per engine. Milestone 1's hw_link.cfg
    // builds one CU named gemv_int8_1; xrt::kernel binds to that by name.
    xrt::device device;
    xrt::uuid   uuid;
    xrt::kernel kernel;

    // For each weight matrix loaded:
    //   * one HBM-resident xrt::bo for the int8 weight data
    //   * one HBM-resident xrt::bo for the int32 output (sized rows*4)
    //     -- we keep these per-weight so back-to-back runs don't fight
    //     over the same output bo.
    //   * a cached row_scales vector for dequant
    struct WeightSlot {
        int                 rows = 0;
        int                 cols = 0;
        xrt::bo             bo_w;
        xrt::bo             bo_y;
        std::vector<float>  row_scales;
    };

    // Map from caller-supplied name -> slot.
    std::map<std::string, WeightSlot> registry;

    // Activation buffer is reused across all weights (size = max rows we
    // see). Reallocating an xrt::bo is cheap-ish but we'd rather not do
    // it every token, so we keep one and grow it if needed.
    xrt::bo bo_x;
    int     bo_x_capacity = 0;

    bool    verbose = false;

    void ensure_x_capacity(int cols);
};

namespace {

bool file_exists(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return f.good();
}

} // anonymous

void FpgaGemvEngine::Impl::ensure_x_capacity(int cols) {
    if (cols <= bo_x_capacity) return;
    // Round up to the next 4K to dodge per-token reallocation when sizes
    // are close (most LM heads share dim=288 etc., so this loop usually
    // runs once).
    int new_cap = ((cols + 4095) / 4096) * 4096;
    bo_x = xrt::bo(device, size_t(new_cap), kernel.group_id(1));
    bo_x_capacity = new_cap;
}

// ---------------------------------------------------------------------------
// FpgaGemvEngine
// ---------------------------------------------------------------------------
FpgaGemvEngine::FpgaGemvEngine(const std::string& xclbin_path, int device_index)
    : impl_(std::make_unique<Impl>()) {
    if (!file_exists(xclbin_path)) {
        M2_THROW(FpgaError, "xclbin not found: " << xclbin_path);
    }
    try {
        impl_->device = xrt::device(device_index);
    } catch (const std::exception& e) {
        M2_THROW(FpgaError,
                 "xrt::device(" << device_index << ") failed: " << e.what()
                 << "  -- run `xbutil examine` to confirm the U50 is visible.");
    }
    try {
        impl_->uuid = impl_->device.load_xclbin(xclbin_path);
    } catch (const std::exception& e) {
        M2_THROW(FpgaError,
                 "load_xclbin('" << xclbin_path << "') failed: " << e.what()
                 << "  -- ensure the xclbin matches the platform on the card.");
    }
    try {
        impl_->kernel = xrt::kernel(impl_->device, impl_->uuid, "gemv_int8");
    } catch (const std::exception& e) {
        M2_THROW(FpgaError,
                 "xrt::kernel(\"gemv_int8\") failed: " << e.what()
                 << "  -- check that the xclbin actually contains the gemv_int8 CU"
                 << " (xclbinutil --info -i <xclbin>).");
    }
    LOG_INFO("FpgaGemvEngine: xclbin loaded, kernel 'gemv_int8' bound.");
}

FpgaGemvEngine::~FpgaGemvEngine() = default;

void FpgaGemvEngine::load_weight(const std::string& name, const QuantizedMatrix& matrix) {
    M2_CHECK(matrix.rows > 0 && matrix.cols > 0, ShapeError,
             "load_weight: bad shape rows=" << matrix.rows << " cols=" << matrix.cols);
    M2_CHECK(matrix.data.size() == size_t(matrix.rows) * size_t(matrix.cols), ShapeError,
             "load_weight: data size=" << matrix.data.size()
             << " expected=" << (size_t(matrix.rows) * size_t(matrix.cols)));

    Impl::WeightSlot slot;
    slot.rows       = matrix.rows;
    slot.cols       = matrix.cols;
    slot.row_scales = matrix.row_scales;

    // Allocate the weight BO on the bank wired to the kernel's
    // `weights` (gmem0) port. group_id(0) returns that bank's tag.
    const size_t w_bytes = size_t(matrix.rows) * size_t(matrix.cols);
    const size_t y_bytes = size_t(matrix.rows) * sizeof(int32_t);
    try {
        slot.bo_w = xrt::bo(impl_->device, w_bytes, impl_->kernel.group_id(0));
        slot.bo_y = xrt::bo(impl_->device, y_bytes, impl_->kernel.group_id(2));
    } catch (const std::exception& e) {
        M2_THROW(FpgaError, "xrt::bo allocation failed: " << e.what());
    }

    // Copy the int8 weight data into the BO and push it to HBM. After
    // this point we never touch the host copy of the weights again --
    // every per-token call reads them straight from HBM.
    auto* w_map = slot.bo_w.map<int8_t*>();
    std::memcpy(w_map, matrix.data.data(), w_bytes);
    slot.bo_w.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Replace any existing entry under this name. The old WeightSlot's
    // xrt::bo will release its HBM allocation in its destructor.
    impl_->registry.erase(name);
    impl_->registry.emplace(name, std::move(slot));

    LOG_INFO("FpgaGemvEngine: loaded weight '%s' [rows=%d cols=%d, %.2f MB] into HBM.",
             name.c_str(), matrix.rows, matrix.cols, double(w_bytes) / (1024.0 * 1024.0));
}

void FpgaGemvEngine::run_int8(const std::string&     name,
                              const QuantizedVector& x,
                              std::vector<int32_t>&  y_int32) {
    auto it = impl_->registry.find(name);
    M2_CHECK(it != impl_->registry.end(), FpgaError,
             "run_int8: unknown weight '" << name << "' -- call load_weight first.");
    Impl::WeightSlot& s = it->second;
    M2_CHECK(x.size == s.cols, ShapeError,
             "run_int8: x.size=" << x.size << " != weight.cols=" << s.cols);

    // Make sure the shared activation BO is big enough for `cols`.
    impl_->ensure_x_capacity(s.cols);

    // Push activation. Only the first cols bytes are meaningful; the rest
    // of the BO can be uninitialized -- the kernel only reads `cols` entries.
    auto* x_map = impl_->bo_x.map<int8_t*>();
    std::memcpy(x_map, x.data.data(), size_t(s.cols));
    impl_->bo_x.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Launch the kernel: (weights, x, y, rows, cols)
    try {
        auto run = impl_->kernel(s.bo_w, impl_->bo_x, s.bo_y, s.rows, s.cols);
        run.wait();
    } catch (const std::exception& e) {
        M2_THROW(FpgaError, "kernel launch failed: " << e.what());
    }

    // Pull the int32 output back.
    s.bo_y.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    auto* y_map = s.bo_y.map<int32_t*>();
    y_int32.assign(y_map, y_map + s.rows);
}

void FpgaGemvEngine::run_dequantized(const std::string&     name,
                                     const QuantizedVector& x,
                                     std::vector<float>&    y_fp32) {
    std::vector<int32_t> y_i32;
    run_int8(name, x, y_i32);

    auto it = impl_->registry.find(name);
    const auto& s = it->second;
    y_fp32.resize(y_i32.size());
    for (size_t i = 0; i < y_i32.size(); ++i) {
        y_fp32[i] = float(y_i32[i]) * s.row_scales[i] * x.scale;
    }
}

bool FpgaGemvEngine::has_weight(const std::string& name) const {
    return impl_->registry.find(name) != impl_->registry.end();
}
int FpgaGemvEngine::rows_of(const std::string& name) const {
    auto it = impl_->registry.find(name);
    M2_CHECK(it != impl_->registry.end(), FpgaError, "rows_of: unknown '" << name << "'");
    return it->second.rows;
}
int FpgaGemvEngine::cols_of(const std::string& name) const {
    auto it = impl_->registry.find(name);
    M2_CHECK(it != impl_->registry.end(), FpgaError, "cols_of: unknown '" << name << "'");
    return it->second.cols;
}

} // namespace m2
