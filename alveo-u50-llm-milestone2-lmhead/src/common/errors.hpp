// errors.hpp -- common exception types + macros so failure modes are loud.
//
// Convention used by this project:
//   * Every function that can fail throws on failure -- no silent skips.
//   * The exception type identifies the *category* (IO / format / shape
//     mismatch / runtime); the .what() string gives the *details*.
//   * Top-level main()s catch std::exception and turn it into a nonzero
//     exit code. That's why M2_CHECK / M2_THROW always carry a message.

#pragma once

#include <stdexcept>
#include <string>
#include <sstream>

namespace m2 {

// File / checkpoint / tokenizer I/O.
class IoError       : public std::runtime_error { using runtime_error::runtime_error; };
// Checkpoint header / tokenizer layout violations -- the file is there but
// doesn't look like what we expect.
class FormatError   : public std::runtime_error { using runtime_error::runtime_error; };
// Tensor / vector sizes that don't line up. Almost always our bug, not the
// user's -- but we throw rather than abort so tests can catch it.
class ShapeError    : public std::runtime_error { using runtime_error::runtime_error; };
// XRT / kernel-side things that the user can fix (xclbin missing, wrong
// shell, kernel name mismatch).
class FpgaError     : public std::runtime_error { using runtime_error::runtime_error; };

} // namespace m2

// M2_THROW(IoError, "checkpoint not found: " << path);
//   -- composes a message stream-style and throws the named exception.
#define M2_THROW(ExcType, msg_expr) do {                                        \
    std::ostringstream _oss;                                                    \
    _oss << msg_expr;                                                           \
    throw ::m2::ExcType(_oss.str());                                            \
} while (0)

// M2_CHECK(cond, ShapeError, "rows="<<r<<" cols="<<c);
//   -- like assert(), but throws and stays compiled-in for release builds.
#define M2_CHECK(cond, ExcType, msg_expr) do {                                  \
    if (!(cond)) {                                                              \
        std::ostringstream _oss;                                                \
        _oss << "check failed (" << #cond << "): " << msg_expr;                 \
        throw ::m2::ExcType(_oss.str());                                        \
    }                                                                           \
} while (0)
