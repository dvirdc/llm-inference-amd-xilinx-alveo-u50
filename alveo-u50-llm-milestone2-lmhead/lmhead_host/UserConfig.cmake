# Copyright (C) 2023 Advanced Micro Devices, Inc.  All rights reserved.
#
# This file lists the sources that make up the `lmhead_host` Vitis IDE
# HOST component. ALL paths are relative to this directory (lmhead_host/).
#
# The actual source tree lives one level up at ../src/. We point at it
# directly rather than duplicating files into lmhead_host/src/, so the
# Makefile-driven build (`make host`) and the Vitis IDE build see the
# same files. If you add a new .cpp under src/, list it here too.

cmake_minimum_required(VERSION 3.16)

###    USER SETTINGS  START    ###

set(USER_COMPILE_DEFINITIONS
)

set(USER_UNDEFINED_SYMBOLS
)

# Extra include dirs beyond the `../src` already added by CMakeLists.txt.
# Empty here -- all headers are reachable from `../src`.
set(USER_INCLUDE_DIRECTORIES
)

# IDE-driven build produces a single executable named after the component
# (`lmhead_host`). Its sources are:
#   * the app entry point (main_hybrid_lmhead)
#   * the FPGA wrapper + the M1 XRT helpers
#   * the entire llama library (model, tokenizer, sampler, quantization,
#     compare, forward)
# Order does not matter -- CMake handles dependencies.
set(USER_COMPILE_SOURCES
"../src/apps/main_hybrid_lmhead.cpp"
"../src/llama/llama_model.cpp"
"../src/llama/tokenizer.cpp"
"../src/llama/sampler.cpp"
"../src/llama/quantization.cpp"
"../src/llama/compare.cpp"
"../src/llama/llama_forward_cpu.cpp"
"../src/fpga/fpga_gemv_engine.cpp"
"../src/host/xrt_utils.cpp"
)

set(USER_CMAKE_CXX_STANDARD
)

set(USER_COMPILE_WARNINGS_ALL -Wall)
set(USER_COMPILE_WARNINGS_EXTRA -Wextra)
set(USER_COMPILE_WARNINGS_AS_ERRORS )
set(USER_COMPILE_WARNINGS_CHECK_SYNTAX_ONLY )
set(USER_COMPILE_WARNINGS_PEDANTIC )
set(USER_COMPILE_WARNINGS_PEDANTIC_AS_ERRORS )
set(USER_COMPILE_WARNINGS_INHIBIT_ALL )

# Default Vitis-IDE debug build: -O0 + -g3. The Makefile flow uses -O2/-g
# by default but accepts DEBUG=1 to match these settings. Keeping -O0 here
# means breakpoints inside inlined templates (std::vector, etc.) are
# reachable in the debugger.
set(USER_COMPILE_OPTIMIZATION_LEVEL -O0)
set(USER_COMPILE_OPTIMIZATION_OTHER_FLAGS )
set(USER_COMPILE_DEBUG_LEVEL -g3)
set(USER_COMPILE_DEBUG_OTHER_FLAGS )
set(USER_COMPILE_PROFILING_ENABLE )
set(USER_COMPILE_VERBOSE )
set(USER_COMPILE_ANSI )
set(USER_COMPILE_OTHER_FLAGS )

set(USER_LINK_NO_START_FILES )
set(USER_LINK_NO_DEFAULT_LIBS )
set(USER_LINK_NO_STDLIB )
set(USER_LINK_OMIT_ALL_SYMBOL_INFO )
set(USER_LINK_LIBRARIES
)
set(USER_LINK_DIRECTORIES
)
set(USER_LINK_OTHER_FLAGS
)

###   END OF USER SETTINGS SECTION ###
###   DO NOT EDIT BEYOND THIS LINE ###

set(USER_COMPILE_OPTIONS
    ${USER_COMPILE_WARNINGS_ALL}
    ${USER_COMPILE_WARNINGS_EXTRA}
    ${USER_COMPILE_WARNINGS_AS_ERRORS}
    ${USER_COMPILE_WARNINGS_CHECK_SYNTAX_ONLY}
    ${USER_COMPILE_WARNINGS_PEDANTIC}
    ${USER_COMPILE_WARNINGS_PEDANTIC_AS_ERRORS}
    ${USER_COMPILE_WARNINGS_INHIBIT_ALL}
    ${USER_COMPILE_OPTIMIZATION_LEVEL}
    ${USER_COMPILE_OPTIMIZATION_OTHER_FLAGS}
    ${USER_COMPILE_DEBUG_LEVEL}
    ${USER_COMPILE_DEBUG_OTHER_FLAGS}
    ${USER_COMPILE_VERBOSE}
    ${USER_COMPILE_ANSI}
    ${USER_COMPILE_OTHER_FLAGS}
)
foreach(entry ${USER_UNDEFINED_SYMBOLS})
    list(APPEND USER_COMPILE_OPTIONS " -U${entry}")
endforeach()

set(USER_LINK_OPTIONS
    ${USER_LINKER_NO_START_FILES}
    ${USER_LINKER_NO_DEFAULT_LIBS}
    ${USER_LINKER_NO_STDLIB}
    ${USER_LINKER_OMIT_ALL_SYMBOL_INFO}
    ${USER_LINK_OTHER_FLAGS}
)
