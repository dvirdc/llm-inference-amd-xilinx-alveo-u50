// timer.hpp -- tiny chrono wrapper used for both CPU and host-side FPGA timing.
//
// Two flavors:
//   Timer t;  t.start(); ...do work...; double ms = t.stop_ms();
//   ScopedTimer st("label");  ...work...  // prints "label: 1.234 ms" at scope exit
//
// Header-only on purpose -- nothing here is performance-critical and inlining
// keeps the call sites short.

#pragma once

#include <chrono>
#include <cstdio>
#include <string>

namespace gemv {

class Timer {
public:
    void start() {
        t0_ = clock::now();
    }
    // Returns elapsed milliseconds since the last start().
    double stop_ms() {
        auto t1 = clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0_).count();
    }

private:
    using clock = std::chrono::steady_clock;
    clock::time_point t0_{};
};

class ScopedTimer {
public:
    explicit ScopedTimer(std::string label) : label_(std::move(label)) {
        t_.start();
    }
    ~ScopedTimer() {
        std::printf("%-28s : %10.3f ms\n", label_.c_str(), t_.stop_ms());
    }

private:
    std::string label_;
    Timer       t_;
};

} // namespace gemv
