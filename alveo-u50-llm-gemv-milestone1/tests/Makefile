# ============================================================================
#  tests/Makefile  --  CPU-only unit tests for the gemv project.
# ----------------------------------------------------------------------------
#  These tests do NOT need Vitis or XRT. They exercise the CPU reference and
#  the deterministic test-data generator. They live outside the Vitis IDE
#  build flow on purpose so you can run them quickly from the terminal:
#
#      make -C tests            # build + run all tests
#      make -C tests clean
#
#  Headers come from gemv_common/src and implementations from gemv_host/src,
#  matching the IDE's include layout (see gemv_host/vitis-comp.json).
# ============================================================================

CXX      ?= g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra \
            -I../gemv_common/src \
            -I../gemv_host/src

COMMON_OBJS := ../gemv_host/src/gemv_cpu.cpp ../gemv_host/src/test_data.cpp

BUILD := build

.PHONY: all run clean
all: run

$(BUILD):
	mkdir -p $@

$(BUILD)/test_gemv_cpu: test_gemv_cpu.cpp $(COMMON_OBJS) | $(BUILD)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD)/test_data_generation: test_data_generation.cpp ../gemv_host/src/test_data.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $^ -o $@

run: $(BUILD)/test_gemv_cpu $(BUILD)/test_data_generation
	@echo "==> running test_gemv_cpu"
	@$(BUILD)/test_gemv_cpu
	@echo ""
	@echo "==> running test_data_generation"
	@$(BUILD)/test_data_generation

clean:
	rm -rf $(BUILD)
