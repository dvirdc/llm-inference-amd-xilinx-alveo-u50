#include "xrt_utils.hpp"

#include <cstdio>
#include <fstream>
#include <stdexcept>

// NOTE: XRT 2.23+ prints deprecation warnings on the
// xrt::device::load_xclbin() and xrt::kernel(device, uuid, name) constructors,
// recommending xrt::hw_context-based code instead. We keep the older forms on
// purpose: (1) they match the working sibling vadd project, (2) they remain
// supported in current XRT, and (3) the spec explicitly says to isolate any
// XRT-version specifics here. Migrating to hw_context is on the M2+ list.

namespace gemv {

xrt::device open_device_and_load(const std::string& xclbin_path,
                                 unsigned int       device_idx,
                                 xrt::uuid&         out_uuid) {
    // Fail fast and loudly if the xclbin path is wrong -- otherwise XRT
    // gives a much less helpful error deep inside load_xclbin().
    {
        std::ifstream f(xclbin_path, std::ios::binary);
        if (!f.good()) {
            throw std::runtime_error("xclbin not found: " + xclbin_path);
        }
    }

    xrt::device dev;
    try {
        dev = xrt::device(device_idx);
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "xrt::device(" + std::to_string(device_idx) + ") failed: " + e.what()
            + "\nHint: run `xbutil examine` and confirm the U50 is visible and the user has access.");
    }

    try {
        out_uuid = dev.load_xclbin(xclbin_path);
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "load_xclbin('" + xclbin_path + "') failed: " + e.what()
            + "\nHint: the xclbin must match the platform/shell flashed on the card.");
    }

    return dev;
}

xrt::kernel make_kernel(const xrt::device& dev,
                        const xrt::uuid&   uuid,
                        const std::string& kernel_name,
                        bool               verbose) {
    xrt::kernel k;
    try {
        k = xrt::kernel(dev, uuid, kernel_name);
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "xrt::kernel(" + kernel_name + ") failed: " + e.what()
            + "\nHint: kernel name must match the extern \"C\" function in the .xo, "
              "and the CU must be present in the xclbin (check `xclbinutil --info`).");
    }

    if (verbose) {
        // group_id() tells us which memory bank (HBM channel index in
        // U50's case) the argument is connected to in the xclbin. Useful
        // for verifying that u50_hbm.cfg actually took effect.
        try {
            std::printf("[xrt_utils] kernel '%s' arg group_ids: w=%u  x=%u  y=%u\n",
                        kernel_name.c_str(),
                        k.group_id(0), k.group_id(1), k.group_id(2));
        } catch (const std::exception& e) {
            std::printf("[xrt_utils] group_id query failed: %s\n", e.what());
        }
    }
    return k;
}

} // namespace gemv
