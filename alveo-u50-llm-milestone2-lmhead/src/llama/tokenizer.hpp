// tokenizer.hpp -- minimal SentencePiece-BPE tokenizer matching the format
// emitted by Karpathy's `tokenizer.bin`.
//
// Format:
//   int32  max_token_length
//   for i in [0, vocab_size):
//     float32 score
//     int32   len
//     char    bytes[len]
//
// Encode follows Karpathy's reference algorithm:
//   1. Optionally prepend BOS (id 1).
//   2. Optionally prepend " " (sentencepiece "dummy prefix" -- adds the
//      U+2581 lower-one-eighth-block "▁" semantics so leading words are
//      treated the same as mid-sentence ones).
//   3. Encode each input byte as a 1-byte token id (these always exist:
//      `<0x00>` through `<0xFF>` live near the top of the vocab).
//   4. Greedy-merge: repeatedly find the adjacent (a,b) pair with the
//      highest score whose concatenation `ab` exists in vocab, and merge.
//      Stop when no improvement.
//
// The result is bit-identical to Karpathy's encode() for the same vocab.

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace m2 {

class Tokenizer {
public:
    // Reads tokenizer.bin. vocab_size must match the model.
    Tokenizer(const std::string& path, int vocab_size);

    // Encode text. Optionally prepend BOS (1). EOS (2) is typically only
    // appended manually for fine-tuned chat tokenizers.
    std::vector<int> encode(const std::string& text,
                            bool prepend_bos = true,
                            bool append_eos  = false) const;

    // Decode a single token id back to a string, following Karpathy:
    //   * The very first generated token (prev==BOS) strips the leading
    //     space that gets added to all words.
    //   * Tokens of the form "<0xNN>" are converted back to raw bytes.
    std::string decode(int prev_token, int token) const;

    int vocab_size() const { return static_cast<int>(vocab_.size()); }
    const std::string& piece(int id) const { return vocab_.at(id); }

private:
    std::vector<std::string> vocab_;
    std::vector<float>       scores_;
    unsigned int             max_token_length_ = 0;
    std::unordered_map<std::string, int> piece_to_id_;
};

} // namespace m2
