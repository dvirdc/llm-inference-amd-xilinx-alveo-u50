#include "tokenizer.hpp"

#include <cstdio>
#include <cstdint>
#include <cstring>

#include "../common/errors.hpp"

namespace m2 {

Tokenizer::Tokenizer(const std::string& path, int vocab_size) {
    if (vocab_size <= 0) {
        M2_THROW(ShapeError, "tokenizer: bad vocab_size " << vocab_size);
    }
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        M2_THROW(IoError, "cannot open tokenizer: " << path);
    }

    uint32_t mlen = 0;
    if (std::fread(&mlen, sizeof(uint32_t), 1, f) != 1) {
        std::fclose(f);
        M2_THROW(FormatError, "tokenizer: short read on max_token_length");
    }
    max_token_length_ = mlen;

    vocab_.resize(vocab_size);
    scores_.resize(vocab_size);

    for (int i = 0; i < vocab_size; ++i) {
        if (std::fread(&scores_[i], sizeof(float), 1, f) != 1) {
            std::fclose(f);
            M2_THROW(FormatError, "tokenizer: short read on score " << i);
        }
        int32_t len = 0;
        if (std::fread(&len, sizeof(int32_t), 1, f) != 1) {
            std::fclose(f);
            M2_THROW(FormatError, "tokenizer: short read on len " << i);
        }
        if (len < 0 || static_cast<unsigned>(len) > mlen + 1) {
            std::fclose(f);
            M2_THROW(FormatError, "tokenizer: invalid piece length " << len << " at " << i);
        }
        vocab_[i].resize(len);
        if (len > 0 && std::fread(vocab_[i].data(), 1, len, f) != static_cast<size_t>(len)) {
            std::fclose(f);
            M2_THROW(FormatError, "tokenizer: short read on bytes " << i);
        }
        piece_to_id_[vocab_[i]] = i;
    }
    std::fclose(f);
}

std::vector<int> Tokenizer::encode(const std::string& text,
                                   bool prepend_bos,
                                   bool append_eos) const {
    // 1) initial tokens: BOS + optional dummy prefix + per-byte ids.
    //
    // SentencePiece does NOT have a 1-token-per-byte alphabet directly;
    // instead it has tokens whose textual form happens to be a single
    // UTF-8 character or even a single byte. Karpathy relies on the
    // tokenizer.bin including a 1-char piece for every input byte that
    // appears (true for English text against the Llama2 vocab).
    std::vector<int> tokens;
    tokens.reserve(text.size() + 2);

    if (prepend_bos) tokens.push_back(1); // BOS

    // Add a dummy leading space if there's any text. This is the
    // SentencePiece convention -- without it, "Hello" and " Hello"
    // would tokenize differently from each other in mid-sentence.
    if (!text.empty()) {
        auto it = piece_to_id_.find(" ");
        if (it != piece_to_id_.end()) tokens.push_back(it->second);
    }

    // Per-byte fallback: lookup each character as a 1-char piece.
    for (size_t i = 0; i < text.size(); ) {
        // For ASCII text every char is one byte -- look up directly.
        // For UTF-8 we'd need to group continuation bytes; the Llama
        // SentencePiece vocab generally has multi-byte pieces for
        // common UTF-8 runs, so this naive 1-byte scan still works for
        // ASCII prompts.
        std::string single(1, text[i]);
        auto it = piece_to_id_.find(single);
        if (it == piece_to_id_.end()) {
            // Byte fallback: SentencePiece uses "<0xHH>" for raw bytes
            // that don't appear standalone in vocab. Llama2 tokenizer
            // reserves token ids 3..258 for these (byte 0x00..0xFF).
            int byte_id = 3 + static_cast<unsigned char>(text[i]);
            tokens.push_back(byte_id);
        } else {
            tokens.push_back(it->second);
        }
        i += 1;
    }

    // 2) BPE merges: repeatedly pick the best-scoring adjacent merge.
    //
    // O(N^2) in the number of tokens because we re-scan after every
    // merge. Fine for milestone-2 prompts; Karpathy uses the same.
    while (true) {
        float best_score = -1e30f;
        int   best_id    = -1;
        int   best_idx   = -1;
        for (size_t i = 0; i + 1 < tokens.size(); ++i) {
            std::string merged = vocab_[tokens[i]] + vocab_[tokens[i + 1]];
            auto it = piece_to_id_.find(merged);
            if (it != piece_to_id_.end() && scores_[it->second] > best_score) {
                best_score = scores_[it->second];
                best_id    = it->second;
                best_idx   = static_cast<int>(i);
            }
        }
        if (best_idx == -1) break;
        tokens[best_idx] = best_id;
        tokens.erase(tokens.begin() + best_idx + 1);
    }

    if (append_eos) tokens.push_back(2); // EOS

    return tokens;
}

std::string Tokenizer::decode(int prev_token, int token) const {
    if (token < 0 || token >= static_cast<int>(vocab_.size())) return "";
    std::string piece = vocab_[token];

    // After BOS, the first piece often starts with a leading space that
    // would print as " word" instead of "word". Strip it -- matches the
    // behavior of the reference run.c.
    if (prev_token == 1 && !piece.empty() && piece[0] == ' ') {
        piece.erase(0, 1);
    }

    // Convert sentencepiece byte fallback "<0xHH>" back to the raw byte.
    if (piece.size() == 6 && piece[0] == '<' && piece[1] == '0' && piece[2] == 'x' &&
        piece[5] == '>') {
        auto hex2 = [](char a, char b) -> int {
            auto v = [](char c) {
                if (c >= '0' && c <= '9') return c - '0';
                if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
                if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
                return -1;
            };
            int hi = v(a), lo = v(b);
            if (hi < 0 || lo < 0) return -1;
            return (hi << 4) | lo;
        };
        int b = hex2(piece[3], piece[4]);
        if (b >= 0) return std::string(1, static_cast<char>(b));
    }
    return piece;
}

} // namespace m2
