// xrt_utils.hpp -- thin wrapper around the modern XRT C++ API so main.cpp
// can read top-to-bottom without being interrupted by XRT boilerplate.
//
// We use only the documented "native" XRT API:
//   xrt::device, xrt::xclbin, xrt::kernel, xrt::bo, xrt::run
// available in XRT 2021.2+ (the version shipped with the 202210_1 U50
// platform). If you need to run on an older XRT, replace the implementations
// in xrt_utils.cpp -- main.cpp does not change.

#pragma once

#include <cstdint>
#include <string>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

namespace gemv {

// Opens device index `device_idx` and loads the given .xclbin. Throws
// std::runtime_error on any failure (missing file, wrong platform, ...).
xrt::device open_device_and_load(const std::string& xclbin_path,
                                 unsigned int       device_idx,
                                 xrt::uuid&         out_uuid);

// Convenience: build an xrt::kernel for the kernel name we care about and
// log its memory group ids (helpful when debugging HBM connectivity).
xrt::kernel make_kernel(const xrt::device& dev,
                        const xrt::uuid&   uuid,
                        const std::string& kernel_name,
                        bool               verbose);

} // namespace gemv
