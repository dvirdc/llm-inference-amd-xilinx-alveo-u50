#include "sampler.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "../common/errors.hpp"

namespace m2 {

Sampler::Sampler(int vocab_size, float temperature, uint64_t seed)
    : vocab_size_(vocab_size),
      temperature_(temperature),
      rng_state_(seed ? seed : 0x9E3779B97F4A7C15ull) {
    M2_CHECK(vocab_size > 0, ShapeError, "vocab_size=" << vocab_size);
}

// xorshift64* -- a tiny PRNG suitable for reproducible sampling. Same
// generator as Karpathy's run.c so test cross-checks behave identically.
uint32_t Sampler::next_u32() {
    rng_state_ ^= rng_state_ >> 12;
    rng_state_ ^= rng_state_ << 25;
    rng_state_ ^= rng_state_ >> 27;
    return static_cast<uint32_t>((rng_state_ * 0x2545F4914F6CDD1Dull) >> 32);
}
float Sampler::next_unit_float() {
    return (next_u32() >> 8) / 16777216.0f; // 24-bit, in [0, 1)
}

int Sampler::sample_argmax(const std::vector<float>& logits) const {
    int   best_i = 0;
    float best_v = logits[0];
    for (int i = 1; i < vocab_size_; ++i) {
        if (logits[i] > best_v) { best_v = logits[i]; best_i = i; }
    }
    return best_i;
}

int Sampler::sample_mult(std::vector<float>& logits) {
    // softmax with temperature scaling. Subtracting the max is the usual
    // numerical-stability trick (keeps exponents <= 0).
    float maxv = logits[0];
    for (int i = 1; i < vocab_size_; ++i) maxv = std::max(maxv, logits[i]);

    float sum = 0.0f;
    for (int i = 0; i < vocab_size_; ++i) {
        logits[i] = std::exp((logits[i] - maxv) / temperature_);
        sum += logits[i];
    }
    if (sum <= 0.0f) {
        // softmax degenerated -- fall back to greedy so we never emit a
        // random token.
        return sample_argmax(logits);
    }
    // Inverse-CDF sample. logits is now the (unnormalized) PMF.
    float r = next_unit_float() * sum;
    float cdf = 0.0f;
    for (int i = 0; i < vocab_size_; ++i) {
        cdf += logits[i];
        if (r < cdf) return i;
    }
    return vocab_size_ - 1; // unreachable except for rounding edge cases
}

int Sampler::sample(std::vector<float>& logits) {
    M2_CHECK(static_cast<int>(logits.size()) == vocab_size_, ShapeError,
             "logits size=" << logits.size() << " expected " << vocab_size_);
    if (temperature_ == 0.0f) return sample_argmax(logits);
    return sample_mult(logits);
}

} // namespace m2
