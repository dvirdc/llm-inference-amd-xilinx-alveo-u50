# ============================================================================
#  Makefile -- alveo-u50-llm-milestone2-lmhead
# ----------------------------------------------------------------------------
#  Targets (most common first):
#    make test                      # CPU-only unit tests (no FPGA needed)
#    make host                      # build all four apps
#    make all TARGET=<t> DEVICE=<p> # host + xclbin
#    make run-cpu                   # convenience: run main_cpu against models/
#    make run-hybrid-{sw,hw,real}   # convenience wrappers
#    make test-fpga                 # FPGA-dependent end-to-end test
#    make clean
#
#  Required variables for FPGA builds:
#    DEVICE   -- Vitis platform (e.g. xilinx_u50_gen3x16_xdma_5_202210_1)
#    TARGET   -- sw_emu | hw_emu | hw
#
#  Optional variables:
#    DEBUG=1  -- disable host optimization, increase debug info (default off)
#    VERBOSE=1-- echo each compiler command
#    MILESTONE1_DIR -- where to copy gemv_int8.cpp / xrt_utils from
# ============================================================================

# ---- User-overridable variables ------------------------------------------
TARGET           ?= hw
DEVICE           ?= xilinx_u50_gen3x16_xdma_5_202210_1
KERNEL           ?= gemv_int8
MILESTONE1_DIR   ?= ../alveo-u50-llm-gemv-milestone1
VPP_KERNEL_FREQ  ?= 250

# ---- Layout --------------------------------------------------------------
BUILD       := build
SRC         := src
HOST_DIR    := $(SRC)/host
KERNEL_DIR  := $(SRC)/kernels
COMMON_DIR  := $(SRC)/common
LLAMA_DIR   := $(SRC)/llama
FPGA_DIR    := $(SRC)/fpga
APPS_DIR    := $(SRC)/apps
TESTS_DIR   := tests
CFG_DIR     := configs
LINK_CFG    := $(CFG_DIR)/u50_hbm.cfg

# Tagged outputs so swapping TARGET doesn't silently reuse stale binaries.
XO          := $(BUILD)/$(KERNEL).$(TARGET).xo
XCLBIN      := $(BUILD)/$(KERNEL).$(TARGET).xclbin
VPP_TEMP    := $(BUILD)/$(TARGET)/tmp
VPP_LOG     := $(BUILD)/$(TARGET)/log
VPP_REPORT  := $(BUILD)/$(TARGET)/report

# ---- Toolchain -----------------------------------------------------------
CXX           ?= g++
VPP           ?= v++
EMCONFIGUTIL  ?= emconfigutil

# ---- Common compile flags ------------------------------------------------
ifeq ($(DEBUG),1)
    OPT_FLAGS := -O0 -g3
else
    OPT_FLAGS := -O2 -g
endif
WARN_FLAGS := -Wall -Wextra -Wno-unused-parameter
CXXSTD     := -std=c++17

INCLUDES := -I$(SRC)
ifneq ($(XILINX_XRT),)
    XRT_INC := -I$(XILINX_XRT)/include
    XRT_LIB := -L$(XILINX_XRT)/lib -lxrt_coreutil -luuid
else
    XRT_INC :=
    XRT_LIB :=
endif

HOST_CXXFLAGS := $(CXXSTD) $(OPT_FLAGS) $(WARN_FLAGS) $(INCLUDES) $(XRT_INC)
HOST_LDFLAGS  := $(XRT_LIB) -pthread

# Q is `@` unless VERBOSE=1.
Q := $(if $(VERBOSE),,@)

# ---- Object lists (shared across all apps) ------------------------------
# "Pure" libs: don't pull XRT, safe for CPU-only tests + main_cpu.
PURE_SRCS := \
  $(LLAMA_DIR)/llama_model.cpp \
  $(LLAMA_DIR)/tokenizer.cpp \
  $(LLAMA_DIR)/sampler.cpp \
  $(LLAMA_DIR)/quantization.cpp \
  $(LLAMA_DIR)/compare.cpp \
  $(LLAMA_DIR)/llama_forward_cpu.cpp

# FPGA layer (pulls XRT).
FPGA_SRCS := \
  $(FPGA_DIR)/fpga_gemv_engine.cpp \
  $(HOST_DIR)/xrt_utils.cpp

PURE_OBJS := $(patsubst $(SRC)/%.cpp,$(BUILD)/obj/%.o,$(PURE_SRCS))
FPGA_OBJS := $(patsubst $(SRC)/%.cpp,$(BUILD)/obj/%.o,$(FPGA_SRCS))

APPS := main_cpu main_hybrid_lmhead inspect_model quantize_lm_head
APP_BINS := $(addprefix $(BUILD)/,$(APPS))

# ---- Phony targets -------------------------------------------------------
.PHONY: all host xclbin emconfig test test-fpga clean help \
        copy-milestone1 \
        run-cpu run-hybrid-sw-emu run-hybrid-hw-emu run-hybrid-hw \
        check-xrt check-device

help:
	@echo "Common targets:"
	@echo "  make copy-milestone1 MILESTONE1_DIR=<path>"
	@echo "  make test                              # CPU-only unit tests"
	@echo "  make host                              # build all four apps"
	@echo "  make xclbin TARGET=sw_emu|hw_emu|hw DEVICE=<platform>"
	@echo "  make all    TARGET=...    DEVICE=..."
	@echo "  make emconfig TARGET=sw_emu|hw_emu DEVICE=..."
	@echo "  make run-cpu"
	@echo "  make run-hybrid-sw-emu DEVICE=..."
	@echo "  make run-hybrid-hw-emu DEVICE=..."
	@echo "  make run-hybrid-hw     DEVICE=..."
	@echo "  make test-fpga TEST_XCLBIN=... TEST_LLAMA_CHECKPOINT=... TEST_LLAMA_TOKENIZER=..."
	@echo "  make clean"
	@echo ""
	@echo "Current: TARGET=$(TARGET)  DEVICE=$(DEVICE)  DEBUG=$(if $(DEBUG),1,0)"

all: host xclbin

# ---- Safety checks -------------------------------------------------------
check-xrt:
	@if [ -z "$(XILINX_XRT)" ]; then \
	    echo "ERROR: XILINX_XRT not set. source /opt/xilinx/xrt/setup.sh first."; \
	    exit 1; fi

check-device:
	@if [ -z "$(DEVICE)" ]; then \
	    echo "ERROR: DEVICE empty. Pass DEVICE=<platform>"; \
	    exit 1; fi

# ---- Copy reusable bits from Milestone 1 --------------------------------
copy-milestone1:
	@if [ -z "$(MILESTONE1_DIR)" ] || [ ! -d "$(MILESTONE1_DIR)" ]; then \
	    echo "ERROR: MILESTONE1_DIR=$(MILESTONE1_DIR) does not exist"; exit 1; fi
	cp $(MILESTONE1_DIR)/gemv_kernel/src/gemv_int8.cpp $(KERNEL_DIR)/gemv_int8.cpp
	cp $(MILESTONE1_DIR)/gemv_host/src/xrt_utils.hpp   $(HOST_DIR)/xrt_utils.hpp
	cp $(MILESTONE1_DIR)/gemv_host/src/xrt_utils.cpp   $(HOST_DIR)/xrt_utils.cpp
	cp $(MILESTONE1_DIR)/gemv_common/src/timer.hpp     $(HOST_DIR)/timer.hpp
	cp $(MILESTONE1_DIR)/gemv/hw_link/gemv-link.cfg    $(CFG_DIR)/u50_hbm.cfg
	@echo "Copied M1 sources into this tree."

# ---- Generic object rule -------------------------------------------------
$(BUILD)/obj/%.o: $(SRC)/%.cpp | $(BUILD)/obj
	@mkdir -p $(dir $@)
	$(Q)$(CXX) $(HOST_CXXFLAGS) -c $< -o $@

# ---- Per-app build (each links PURE_OBJS + FPGA_OBJS as needed) ---------
# All four apps share the same .o set. main_cpu and inspect_model don't
# touch FPGA code, but linking the FpgaGemvEngine objects is harmless --
# they'll be dead-stripped by the linker. Compile cost is small.
$(BUILD)/main_cpu:             $(APPS_DIR)/main_cpu.cpp             $(PURE_OBJS) | $(BUILD) check-xrt
	$(Q)$(CXX) $(HOST_CXXFLAGS) $< $(PURE_OBJS) $(HOST_LDFLAGS) -o $@

$(BUILD)/inspect_model:        $(APPS_DIR)/inspect_model.cpp        $(PURE_OBJS) | $(BUILD) check-xrt
	$(Q)$(CXX) $(HOST_CXXFLAGS) $< $(PURE_OBJS) $(HOST_LDFLAGS) -o $@

$(BUILD)/quantize_lm_head:     $(APPS_DIR)/quantize_lm_head.cpp     $(PURE_OBJS) | $(BUILD) check-xrt
	$(Q)$(CXX) $(HOST_CXXFLAGS) $< $(PURE_OBJS) $(HOST_LDFLAGS) -o $@

$(BUILD)/main_hybrid_lmhead:   $(APPS_DIR)/main_hybrid_lmhead.cpp   $(PURE_OBJS) $(FPGA_OBJS) | $(BUILD) check-xrt
	$(Q)$(CXX) $(HOST_CXXFLAGS) $< $(PURE_OBJS) $(FPGA_OBJS) $(HOST_LDFLAGS) -o $@

host: $(APP_BINS)

# ---- Kernel build --------------------------------------------------------
VPP_COMMON := -t $(TARGET) --platform $(DEVICE) \
              --save-temps --temp_dir $(VPP_TEMP) \
              --log_dir $(VPP_LOG) --report_dir $(VPP_REPORT)

xclbin: check-xrt check-device $(XCLBIN)

$(XO): $(KERNEL_DIR)/$(KERNEL).cpp | $(BUILD) $(VPP_TEMP)
	$(VPP) -c $(VPP_COMMON) -k $(KERNEL) \
	       --kernel_frequency $(VPP_KERNEL_FREQ) \
	       -I$(KERNEL_DIR) $< -o $@

$(XCLBIN): $(XO) $(LINK_CFG) | $(BUILD) $(VPP_TEMP)
	$(VPP) -l $(VPP_COMMON) --config $(LINK_CFG) \
	       --kernel_frequency $(VPP_KERNEL_FREQ) \
	       $(XO) -o $@

emconfig: check-device | $(BUILD)
	cd $(BUILD) && $(EMCONFIGUTIL) --platform $(DEVICE)

# ---- CPU-only tests ------------------------------------------------------
TEST_CXXFLAGS := $(CXXSTD) -O2 -Wall -Wextra -I$(SRC)
TEST_BINS := \
  $(BUILD)/test_quantization \
  $(BUILD)/test_gemv_cpu \
  $(BUILD)/test_llama_model_loader \
  $(BUILD)/test_topk_compare

test: $(TEST_BINS)
	@echo "==> test_quantization"      && $(BUILD)/test_quantization
	@echo "==> test_gemv_cpu"           && $(BUILD)/test_gemv_cpu
	@echo "==> test_llama_model_loader" && $(BUILD)/test_llama_model_loader
	@echo "==> test_topk_compare"       && $(BUILD)/test_topk_compare

$(BUILD)/test_quantization: $(TESTS_DIR)/test_quantization.cpp $(LLAMA_DIR)/quantization.cpp | $(BUILD)
	$(CXX) $(TEST_CXXFLAGS) $^ -o $@

$(BUILD)/test_gemv_cpu: $(TESTS_DIR)/test_gemv_cpu.cpp $(LLAMA_DIR)/quantization.cpp | $(BUILD)
	$(CXX) $(TEST_CXXFLAGS) $^ -o $@

$(BUILD)/test_llama_model_loader: $(TESTS_DIR)/test_llama_model_loader.cpp $(LLAMA_DIR)/llama_model.cpp | $(BUILD)
	$(CXX) $(TEST_CXXFLAGS) $^ -o $@

$(BUILD)/test_topk_compare: $(TESTS_DIR)/test_topk_compare.cpp $(LLAMA_DIR)/compare.cpp | $(BUILD)
	$(CXX) $(TEST_CXXFLAGS) $^ -o $@

# ---- FPGA-dependent test --------------------------------------------------
# Links the full set so it can actually touch the device.
$(BUILD)/test_fpga_lm_head: $(TESTS_DIR)/test_fpga_lm_head.cpp \
                            $(PURE_OBJS) $(FPGA_OBJS) | $(BUILD) check-xrt
	$(CXX) $(HOST_CXXFLAGS) $< $(PURE_OBJS) $(FPGA_OBJS) $(HOST_LDFLAGS) -o $@

test-fpga: $(BUILD)/test_fpga_lm_head
	@TEST_XCLBIN=$$TEST_XCLBIN \
	 TEST_LLAMA_CHECKPOINT=$$TEST_LLAMA_CHECKPOINT \
	 TEST_LLAMA_TOKENIZER=$$TEST_LLAMA_TOKENIZER \
	 $(BUILD)/test_fpga_lm_head

# ---- Convenience runners --------------------------------------------------
run-cpu: $(BUILD)/main_cpu
	$(BUILD)/main_cpu \
	  --checkpoint models/stories15M.bin \
	  --tokenizer  models/tokenizer.bin \
	  --prompt "Once upon a time" --steps 16 --temperature 0.0 --verbose

run-hybrid-sw-emu: $(BUILD)/main_hybrid_lmhead emconfig
	XCL_EMULATION_MODE=sw_emu $(BUILD)/main_hybrid_lmhead \
	  --xclbin $(BUILD)/$(KERNEL).sw_emu.xclbin \
	  --checkpoint models/stories15M.bin --tokenizer models/tokenizer.bin \
	  --prompt "Once upon a time" --steps 4 --temperature 0.0 \
	  --use-fpga-lm-head --compare-every-token --verbose

run-hybrid-hw-emu: $(BUILD)/main_hybrid_lmhead emconfig
	XCL_EMULATION_MODE=hw_emu $(BUILD)/main_hybrid_lmhead \
	  --xclbin $(BUILD)/$(KERNEL).hw_emu.xclbin \
	  --checkpoint models/stories15M.bin --tokenizer models/tokenizer.bin \
	  --prompt "Once upon a time" --steps 4 --temperature 0.0 \
	  --use-fpga-lm-head --compare-every-token --verbose

run-hybrid-hw: $(BUILD)/main_hybrid_lmhead
	$(BUILD)/main_hybrid_lmhead \
	  --xclbin $(BUILD)/$(KERNEL).hw.xclbin \
	  --checkpoint models/stories15M.bin --tokenizer models/tokenizer.bin \
	  --prompt "Once upon a time" --steps 16 --temperature 0.0 \
	  --use-fpga-lm-head --verbose

# ---- Dir creation --------------------------------------------------------
$(BUILD):
	mkdir -p $@
$(BUILD)/obj: | $(BUILD)
	mkdir -p $@
$(VPP_TEMP): | $(BUILD)
	mkdir -p $(VPP_TEMP) $(VPP_LOG) $(VPP_REPORT)

# ---- Clean ---------------------------------------------------------------
clean:
	rm -rf $(BUILD) .Xil _x v++_* xrt.log emconfig.json *.compile_summary *.link_summary *.run_summary
