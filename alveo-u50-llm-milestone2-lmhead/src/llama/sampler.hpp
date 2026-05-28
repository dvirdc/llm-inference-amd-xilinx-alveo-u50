// sampler.hpp -- pick the next token from logits.
//
//   * temperature == 0    => argmax (greedy).
//   * temperature  > 0    => softmax(logits / T) then multinomial sample.
//
// We do not implement top-p / top-k for this milestone -- the goal is to
// have a *deterministic*, reproducible generation when T=0 so that CPU
// and FPGA paths can be compared token-by-token.

#pragma once

#include <cstdint>
#include <vector>

namespace m2 {

class Sampler {
public:
    Sampler(int vocab_size, float temperature, uint64_t seed);

    // logits has length vocab_size. Mutated in place when T>0 (softmax).
    int sample(std::vector<float>& logits);

private:
    int      vocab_size_;
    float    temperature_;
    uint64_t rng_state_;

    uint32_t next_u32();
    float    next_unit_float();
    int      sample_argmax(const std::vector<float>& logits) const;
    int      sample_mult(std::vector<float>& logits);
};

} // namespace m2
