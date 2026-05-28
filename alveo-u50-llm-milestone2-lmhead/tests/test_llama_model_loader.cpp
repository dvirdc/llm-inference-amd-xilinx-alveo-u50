// test_llama_model_loader.cpp -- exercises the Karpathy checkpoint
// loader's *error paths* without needing a real model file, and (if the
// user provides one via TEST_LLAMA_CHECKPOINT env var) does a smoke
// load and sanity-checks the parsed config.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

#include "../src/common/errors.hpp"
#include "../src/llama/llama_model.hpp"

namespace {
int g_pass = 0, g_fail = 0;
#define EXPECT_TRUE(x) do { if (x) {++g_pass;} else { ++g_fail; \
  std::printf("  FAIL [%s:%d] %s\n", __FILE__, __LINE__, #x); } } while (0)

void test_missing_file_throws() {
    std::printf("[test] missing file throws IoError\n");
    bool threw = false;
    try {
        m2::load_llama_model("/definitely/does/not/exist.bin");
    } catch (const m2::IoError&) {
        threw = true;
    } catch (...) { /* other exception types are not what we want */ }
    EXPECT_TRUE(threw);
}

void test_garbage_header_throws() {
    std::printf("[test] garbage header throws FormatError\n");
    // Write a 28-byte file of zeros (all dims = 0) and try to load it.
    const char* path = "/tmp/m2_garbage_header.bin";
    std::ofstream f(path, std::ios::binary);
    std::vector<int32_t> hdr(7, 0);
    f.write(reinterpret_cast<const char*>(hdr.data()), 7 * sizeof(int32_t));
    f.close();
    bool threw = false;
    try {
        m2::load_llama_model(path);
    } catch (const m2::FormatError&) { threw = true; }
    catch (...) {}
    EXPECT_TRUE(threw);
    std::remove(path);
}

// If TEST_LLAMA_CHECKPOINT is set in the environment, load it and print
// the config so the operator can eyeball it. Skipped otherwise.
void smoke_load_if_env() {
    const char* path = std::getenv("TEST_LLAMA_CHECKPOINT");
    if (!path || !*path) {
        std::printf("[test] smoke load skipped (set TEST_LLAMA_CHECKPOINT=<path> to enable)\n");
        return;
    }
    std::printf("[test] smoke load: %s\n", path);
    try {
        auto m = m2::load_llama_model(path);
        m2::print_llama_config(m.config);
        EXPECT_TRUE(!m.weights.token_embedding_table.empty());
        EXPECT_TRUE(!m.weights.wcls.empty());
        EXPECT_TRUE(static_cast<int>(m.weights.wcls.size())
                    == m.config.vocab_size * m.config.dim);
    } catch (const std::exception& e) {
        std::printf("  load threw: %s\n", e.what());
        ++g_fail;
    }
}

} // anonymous

int main() {
    test_missing_file_throws();
    test_garbage_header_throws();
    smoke_load_if_env();
    std::printf("\n[test_llama_model_loader] passed=%d failed=%d\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
