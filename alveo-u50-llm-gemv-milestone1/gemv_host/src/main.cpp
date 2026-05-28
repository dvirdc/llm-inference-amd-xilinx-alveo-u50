// ============================================================================
//  main.cpp  --  Host application for the INT8 GEMV milestone-1 kernel.
//
//  Flow:
//    1. Parse CLI args (--xclbin, --rows, --cols, --seed, --verbose).
//    2. Generate deterministic INT8 weights + x.
//    3. Compute the int32 CPU reference and time it.
//    4. Open device, load xclbin, build kernel.
//    5. Allocate xrt::bo buffers in the HBM channels picked by the linker
//       (group_id() per argument).
//    6. Copy weights/x into the BOs, sync H2D, run kernel, wait, sync D2H.
//    7. Bit-exact compare against the CPU reference.
//    8. Print timings + PASS/FAIL and return non-zero on FAIL.
// ============================================================================

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "gemv_cpu.hpp"
#include "test_data.hpp"
#include "timer.hpp"
#include "xrt_utils.hpp"

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

namespace {

struct Args {
    std::string  xclbin;
    int          rows        = 1024;
    int          cols        = 1024;
    uint32_t     seed        = 0xC0FFEEu;
    unsigned int device_idx  = 0;
    bool         verbose     = false;
};

void print_usage(const char* argv0) {
    std::printf(
        "Usage: %s --xclbin <path> [--rows N] [--cols N] [--seed N]\n"
        "          [--device-id N] [--verbose] [-h|--help]\n"
        "\n"
        "  --xclbin <path>   Path to compiled gemv_int8.xclbin (required)\n"
        "  --rows N          Output vector length        (default 1024)\n"
        "  --cols N          Input vector length         (default 1024)\n"
        "  --seed N          Deterministic RNG seed      (default 0xC0FFEE)\n"
        "  --device-id N     XRT device index            (default 0)\n"
        "  --verbose         Print extra diagnostic info\n"
        "  -h, --help        This help\n",
        argv0);
}

// Tiny self-contained arg parser. Keeps main.cpp dependency-free; no need to
// drag in the vadd-style sda::CmdLineParser + its logger here.
bool parse_args(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto need = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Error: %s requires a value\n", name);
                std::exit(2);
            }
            return argv[++i];
        };
        if      (k == "-h" || k == "--help")    { print_usage(argv[0]); std::exit(0); }
        else if (k == "--xclbin")               { a.xclbin     = need("--xclbin"); }
        else if (k == "--rows")                 { a.rows       = std::atoi(need("--rows")); }
        else if (k == "--cols")                 { a.cols       = std::atoi(need("--cols")); }
        else if (k == "--seed")                 { a.seed       = static_cast<uint32_t>(std::strtoul(need("--seed"), nullptr, 0)); }
        else if (k == "--device-id")            { a.device_idx = static_cast<unsigned int>(std::atoi(need("--device-id"))); }
        else if (k == "--verbose" || k == "-v") { a.verbose    = true; }
        else {
            std::fprintf(stderr, "Unknown arg: %s\n", k.c_str());
            print_usage(argv[0]);
            return false;
        }
    }
    if (a.xclbin.empty()) {
        std::fprintf(stderr, "Error: --xclbin is required\n\n");
        print_usage(argv[0]);
        return false;
    }
    if (a.rows <= 0 || a.cols <= 0) {
        std::fprintf(stderr, "Error: rows and cols must be positive\n");
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv) try {
    Args args;
    if (!parse_args(argc, argv, args)) return 2;

    const int      rows   = args.rows;
    const int      cols   = args.cols;
    const size_t   w_n    = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    const size_t   x_n    = static_cast<size_t>(cols);
    const size_t   y_n    = static_cast<size_t>(rows);

    std::printf("GEMV INT8 test\n");
    std::printf("  rows         : %d\n", rows);
    std::printf("  cols         : %d\n", cols);
    std::printf("  seed         : 0x%08X\n", args.seed);
    std::printf("  xclbin       : %s\n", args.xclbin.c_str());
    std::printf("  device index : %u\n", args.device_idx);

    // -------- 1. Test data ------------------------------------------------
    std::vector<int8_t> h_weights, h_x;
    // Use two different sub-seeds so weights and x aren't bit-correlated.
    gemv::fill_random_int8(h_weights, w_n, args.seed ^ 0x11111111u);
    gemv::fill_random_int8(h_x,       x_n, args.seed ^ 0x22222222u);

    if (args.verbose) {
        gemv::print_vec_int8("weights (head)", h_weights);
        gemv::print_vec_int8("x       (head)", h_x);
    }

    // -------- 2. CPU reference -------------------------------------------
    std::vector<int32_t> y_ref;
    double cpu_ms = 0.0;
    {
        gemv::Timer t; t.start();
        gemv::gemv_cpu_ref(h_weights, h_x, y_ref, rows, cols);
        cpu_ms = t.stop_ms();
    }
    if (args.verbose) gemv::print_vec_i32("y_ref (head)", y_ref);

    // -------- 3. Open device + load xclbin -------------------------------
    xrt::uuid uuid;
    xrt::device dev = gemv::open_device_and_load(args.xclbin, args.device_idx, uuid);
    xrt::kernel krnl = gemv::make_kernel(dev, uuid, "gemv_int8", args.verbose);

    // -------- 4. Allocate BOs in the HBM channels chosen by the linker ---
    // group_id(i) returns the memory bank tag for kernel argument i.
    // Argument indices match the kernel signature: 0=weights, 1=x, 2=y.
    const size_t bytes_w = w_n * sizeof(int8_t);
    const size_t bytes_x = x_n * sizeof(int8_t);
    const size_t bytes_y = y_n * sizeof(int32_t);

    xrt::bo bo_w(dev, bytes_w, krnl.group_id(0));
    xrt::bo bo_x(dev, bytes_x, krnl.group_id(1));
    xrt::bo bo_y(dev, bytes_y, krnl.group_id(2));

    // Map host views and copy input data.
    auto bo_w_map = bo_w.map<int8_t*>();
    auto bo_x_map = bo_x.map<int8_t*>();
    auto bo_y_map = bo_y.map<int32_t*>();
    std::memcpy(bo_w_map, h_weights.data(), bytes_w);
    std::memcpy(bo_x_map, h_x.data(),       bytes_x);
    std::memset(bo_y_map, 0, bytes_y); // make any partial-write bug obvious

    // -------- 5. H2D --> kernel --> D2H ----------------------------------
    gemv::Timer t_total; t_total.start();

    double h2d_ms = 0.0, krn_ms = 0.0, d2h_ms = 0.0;
    {
        gemv::Timer t; t.start();
        bo_w.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_x.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        h2d_ms = t.stop_ms();
    }
    {
        gemv::Timer t; t.start();
        auto run = krnl(bo_w, bo_x, bo_y, rows, cols);
        run.wait();
        krn_ms = t.stop_ms();
    }
    {
        gemv::Timer t; t.start();
        bo_y.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        d2h_ms = t.stop_ms();
    }
    const double total_ms = t_total.stop_ms();

    // -------- 6. Compare -------------------------------------------------
    std::vector<int32_t> y_dev(bo_y_map, bo_y_map + y_n);

    size_t mismatches = 0;
    const size_t kMaxPrint = 8;
    for (size_t i = 0; i < y_n; ++i) {
        if (y_dev[i] != y_ref[i]) {
            if (mismatches < kMaxPrint) {
                std::printf("  mismatch[%zu]: dev=%d  ref=%d  (diff=%d)\n",
                            i, y_dev[i], y_ref[i], y_dev[i] - y_ref[i]);
            }
            ++mismatches;
        }
    }

    // -------- 7. Summary -------------------------------------------------
    std::printf("\n--- Timing ---\n");
    std::printf("  %-26s : %10.3f ms\n", "CPU reference",        cpu_ms);
    std::printf("  %-26s : %10.3f ms\n", "H2D transfer",         h2d_ms);
    std::printf("  %-26s : %10.3f ms\n", "Kernel execution",     krn_ms);
    std::printf("  %-26s : %10.3f ms\n", "D2H transfer",         d2h_ms);
    std::printf("  %-26s : %10.3f ms\n", "Total FPGA path",      total_ms);

    if (mismatches == 0) {
        std::printf("\nResult: PASS\n");
        return 0;
    } else {
        std::printf("\nResult: FAIL (%zu / %zu mismatches)\n", mismatches, y_n);
        return 1;
    }
}
catch (const std::exception& e) {
    std::fprintf(stderr, "Fatal: %s\n", e.what());
    return 3;
}
