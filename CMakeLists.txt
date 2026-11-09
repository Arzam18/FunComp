cmake_minimum_required(VERSION 3.20)
project(chess_engine VERSION 3.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# An engine built without optimization is a different, much weaker program:
# default to Release unless a build type is chosen.
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Link-time optimization lets calls inline across the engine's files, and
# -march=native tunes for this machine's CPU (the binaries then only run on
# CPUs with the same instruction sets). Both are off by default: measured
# on 2026-09-26 with GCC 16, neither changed bench speed (all four
# combinations within 1.5%). Worth measuring again once the evaluation
# uses SIMD.
option(LTO "Link-time optimization" OFF)
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
if (LTO AND ipo_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()

# ARCH sets -march to a portable CPU level, so release builds don't use
# instructions a tester's CPU lacks: x86-64-v2 (the default; POPCNT and
# SSE4.2, every x86-64 CPU since about 2009) or x86-64-v3 (AVX2, BMI2:
# Haswell, Zen and later). Baseline x86-64 has no popcount instruction:
# GCC then calls a library routine for every bit count, 7% of a search's
# instructions (about 10% of its time). Empty leaves it to the compiler;
# NATIVE overrides it. On 64-bit ARM (Android's arm64-v8a, Linux aarch64)
# the default is armv8-a, whose NEON every such CPU has; armv8.2-a+dotprod
# adds the int8 dot product (SDOT), in phones since about 2018.
if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
    set(ARCH_DEFAULT "armv8-a")
else()
    set(ARCH_DEFAULT "x86-64-v2")
endif()
set(ARCH "${ARCH_DEFAULT}" CACHE STRING
    "CPU level for -march: x86-64-v2, x86-64-v3, armv8-a, armv8.2-a+dotprod, or empty")
if (ARCH AND NOT NATIVE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-march=${ARCH})
endif()

option(NATIVE "Optimize for the build machine's CPU" OFF)
if (NATIVE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-march=native)
endif()

# Profile-guided optimisation (GCC; 2026-10-08, for round 3's datagen):
# PGO=GENERATE builds an instrumented engine whose runs write profiles
# beside its object files; PGO=USE, configured in the same build directory
# afterwards, rebuilds with them. It changes how the code is compiled, not
# what it computes: the bench must come out the same.
set(PGO "" CACHE STRING "Profile-guided optimisation: GENERATE, USE, or empty")
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    if (PGO STREQUAL "GENERATE")
        add_compile_options(-fprofile-generate)
        add_link_options(-fprofile-generate)
    elseif (PGO STREQUAL "USE")
        add_compile_options(-fprofile-use -fprofile-correction -Wno-missing-profile)
    endif()
endif()

# -DTUNE=ON advertises the search parameters in params.h as UCI options,
# for automated tuning (SPSA). Normal builds leave them out of the GUI.
option(TUNE "Expose search parameters as UCI options" OFF)
# FROZEN_PARAMS compiles the search parameters as constants in the engine
# (params.h): the same search, 2.3-3.9% faster on one thread (results.md,
# "Frozen parameters"). For release and datagen builds; a TUNE build keeps
# them variables, and so do the tests, which build their own copy of the
# engine library. Only NnueScale stays settable in a frozen engine.
option(FROZEN_PARAMS "Compile the search parameters as constants (off with TUNE)" ON)
if (TUNE)
    set(FROZEN_PARAMS OFF)
endif()
# Continuation logging (cont_log.h, the genconlog command) and tree sampling
# (tree_sample.h, the treesample command): compiled in only with CONT_LOG,
# since even switched off their hooks cost the search (the log's about 2%
# nps).
option(CONT_LOG "Compile in continuation logging and tree sampling (genconlog, treesample)" OFF)

function(engine_warnings target)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Wconversion)
    endif()
    # Link the C++ runtime statically on MinGW. Otherwise the executable
    # loads whichever libstdc++-6.dll comes first on PATH (Git for Windows
    # ships an older, incompatible one), and it can't be copied elsewhere
    # to run under a match runner.
    get_target_property(type ${target} TYPE)
    if (MINGW AND type STREQUAL "EXECUTABLE")
        target_link_options(${target} PRIVATE -static)
        # Reserve 16 MB of stack (address space only) for every thread,
        # the search thread included, instead of MinGW's 2 MB: deep
        # searches recurse far, singular-extension probes on top.
        target_link_options(${target} PRIVATE -Wl,--stack,16777216)
    elseif (ANDROID AND type STREQUAL "EXECUTABLE")
        # Android: one static executable (the NDK's libc and libc++ linked
        # in), so a chess app can run it from its own folder on any
        # Android 7 or later (API 24), whatever the device's libraries.
        target_link_options(${target} PRIVATE -static)
    elseif (MSVC AND type STREQUAL "EXECUTABLE")
        # The same for MSVC (default 1 MB). POSIX builds size the search
        # thread's stack when they create it (Worker in uci.cpp).
        target_link_options(${target} PRIVATE /STACK:16777216)
    endif()
endfunction()

# Fathom (MIT licence, third_party/fathom/LICENSE): Syzygy tablebase
# probing, unmodified from github.com/jdart1/Fathom at commit c9c6fef
# (2025-12-24), built with its own settings. Its headers are system
# headers to the engine, so our warnings skip them. It locks with pthreads
# outside Windows.
find_package(Threads REQUIRED)
add_library(fathom STATIC third_party/fathom/tbprobe.c)
target_include_directories(fathom SYSTEM PUBLIC third_party/fathom)
target_compile_definitions(fathom PUBLIC TB_NO_HELPER_API)
target_link_libraries(fathom PUBLIC Threads::Threads)
set_target_properties(fathom PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)

# Everything but the UCI front end, shared by the engine and the tests. With
# FROZEN_PARAMS the engine links a second copy, engine_frozen, with the
# parameters as constants; the tests keep `engine`, whose parameters they set.
set(ENGINE_SOURCES
    src/position.cpp
    src/attacks.cpp
    src/movegen.cpp
    src/eval.cpp
    src/initiative.cpp
    src/nnue.cpp
    src/see.cpp
    src/pruning.cpp
    src/search.cpp
    src/tablebase.cpp
    src/threads.cpp
    src/guide_features.cpp
    src/labels.cpp
    src/bingo.cpp
    src/cont_log.cpp
    src/datagen.cpp
    src/featstats.cpp
    src/pair_features.cpp
    src/pawn_features.cpp
    src/square_features.cpp
    src/perft.cpp
    src/pgn.cpp
    src/reservoir.cpp
)
# The release network compiled into the binary (3.0 on; src/embedded_net.cpp): a .nnue file, the default
# nets/funcomp-3.1.nnue when the tree has it (the public release's); empty: none (EvalFile then loads a network,
# the hand-written evaluation otherwise).
set(EMBED_NET_DEFAULT "")
if (EXISTS ${CMAKE_SOURCE_DIR}/nets/funcomp-3.1.nnue)
    set(EMBED_NET_DEFAULT ${CMAKE_SOURCE_DIR}/nets/funcomp-3.1.nnue)
endif()
set(EMBED_NET "${EMBED_NET_DEFAULT}" CACHE FILEPATH "A network compiled into the binary (empty: none)")
list(APPEND ENGINE_SOURCES src/embedded_net.cpp)
if (EMBED_NET)
    if (NOT EXISTS ${EMBED_NET})
        message(FATAL_ERROR "EMBED_NET: no file ${EMBED_NET}")
    endif()
    set_source_files_properties(src/embedded_net.cpp PROPERTIES
        COMPILE_DEFINITIONS "FUNCOMP_EMBED_NET_PATH=\"${EMBED_NET}\""
        OBJECT_DEPENDS ${EMBED_NET})
    message(STATUS "Embedded network: ${EMBED_NET}")
endif()
# The source tree's git commit, for data files to record what made them
# (labels' .params): regenerated at every build, rewritten only on change.
add_custom_target(git_hash ALL
    COMMAND ${CMAKE_COMMAND} -DSRC=${CMAKE_SOURCE_DIR} -DOUT=${CMAKE_BINARY_DIR}/generated/git_hash.h
            -P ${CMAKE_SOURCE_DIR}/cmake/git_hash.cmake
    BYPRODUCTS ${CMAKE_BINARY_DIR}/generated/git_hash.h)
function(engine_library name)
    add_library(${name} STATIC ${ENGINE_SOURCES})
    target_include_directories(${name} PUBLIC src)
    target_link_libraries(${name} PUBLIC fathom)
    engine_warnings(${name})
    add_dependencies(${name} git_hash)
    target_include_directories(${name} PRIVATE ${CMAKE_BINARY_DIR}/generated)
    if (CONT_LOG)
        target_compile_definitions(${name} PUBLIC CONT_LOG)
    endif()
endfunction()
engine_library(engine)
if (FROZEN_PARAMS)
    engine_library(engine_frozen)
    target_compile_definitions(engine_frozen PUBLIC FUNCOMP_FROZEN_PARAMS)
    set(CHESS_ENGINE_LIB engine_frozen)
    message(STATUS "Frozen parameters: the engine compiles them as constants")
else()
    set(CHESS_ENGINE_LIB engine)
endif()

add_executable(chess
    src/main.cpp
    src/uci.cpp
    src/bench.cpp
    src/telemetry.cpp
    src/tree_sample.cpp
)
target_link_libraries(chess PRIVATE ${CHESS_ENGINE_LIB})
if (TUNE)
    target_compile_definitions(chess PRIVATE TUNE)
endif()
engine_warnings(chess)
# Windows: version information and an application manifest in the executable
# (cmake/win/funcomp.rc.in): binaries without them are more often taken for
# malware by Defender's heuristics.
if (WIN32)
    enable_language(RC)
    set(FUNCOMP_MANIFEST ${CMAKE_SOURCE_DIR}/cmake/win/funcomp.manifest)
    configure_file(cmake/win/funcomp.rc.in ${CMAKE_BINARY_DIR}/generated/funcomp.rc @ONLY)
    target_sources(chess PRIVATE ${CMAKE_BINARY_DIR}/generated/funcomp.rc)
endif()

enable_testing()

add_executable(perft_tests tests/perft_tests.cpp)
target_link_libraries(perft_tests PRIVATE engine)
engine_warnings(perft_tests)
add_test(NAME perft COMMAND perft_tests)

# The bench node count: any change to it is a change to the search, made
# on purpose (and stated in the commit) or not at all. Update it with such
# a change.
# 1119452 since the release merge (2026-10-19): the 5c values and tune-5c's
# features (round3-engine), equal to round 3's datagen engine c4a22c3.
# 1197522 since the 3.0 fine tune's values (2026-10-20).
# 1016380 since 3.1's focused tune's values (2026-10-29).
set(BENCH_NODES 1016380)
# With the release network embedded (nets/funcomp-3.1.nnue), bench searches with it: its own count. Another
# network embedded gives another count (set BENCH_NODES_EMBEDDED for it). 1031103 and 1094101: 3.1's network
# (t160-l-o, layered) at 3.1's focused tune's values, the SIMD count equal to the same network from a file
# (3.0's network at the 3.0 fine tune's values: 1187868 on both paths).
# A multi-layer network has a count for each of the two float paths (src/nnue.cpp, the layers after the
# accumulators): AVX2 with FMA, or 64-bit ARM's NEON, which computes what AVX2 does (x86-64-v3, v4, Android), and
# the plain C++ fallback (x86-64, x86-64-v2), which runs layer2 in float where the others run it in int8 and rounds
# without fused multiply-adds: BENCH_NODES_EMBEDDED_PLAIN. A flat network: the same count on both (the author,
# 2026-10-28: a count for each build, not the fallback made to round alike).
set(BENCH_NODES_EMBEDDED 1031103 CACHE STRING
    "The bench count with the embedded network (x86-64-v3, v4, Android; the others: BENCH_NODES_EMBEDDED_PLAIN)")
set(BENCH_NODES_EMBEDDED_PLAIN 1094101 CACHE STRING
    "The bench count with the embedded network, builds whose float layers are plain C++ (x86-64, x86-64-v2)")
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_FLAGS "")
if (NATIVE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_REQUIRED_FLAGS "-march=native")
elseif (ARCH AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_REQUIRED_FLAGS "-march=${ARCH}")
endif()
unset(FUNCOMP_SIMD_FLOAT CACHE)  # checked again at every configure: ARCH or NATIVE may have changed
check_cxx_source_compiles("
#if !((defined(__AVX2__) && defined(__FMA__)) || (defined(__ARM_NEON) && defined(__aarch64__)))
#error the plain float layers
#endif
int main() { return 0; }" FUNCOMP_SIMD_FLOAT)
unset(CMAKE_REQUIRED_FLAGS)
if (EMBED_NET AND FUNCOMP_SIMD_FLOAT)
    set(BENCH_NODES ${BENCH_NODES_EMBEDDED})
    message(STATUS "Float layers: SIMD, bench ${BENCH_NODES} (BENCH_NODES_EMBEDDED)")
elseif (EMBED_NET)
    set(BENCH_NODES ${BENCH_NODES_EMBEDDED_PLAIN})
    message(STATUS "Float layers: plain C++, bench ${BENCH_NODES} (BENCH_NODES_EMBEDDED_PLAIN)")
endif()
add_test(NAME bench COMMAND chess bench)
set_tests_properties(bench PROPERTIES PASS_REGULAR_EXPRESSION "${BENCH_NODES} nodes ")

add_executable(search_tests tests/search_tests.cpp src/bench.cpp)
target_link_libraries(search_tests PRIVATE engine)
engine_warnings(search_tests)
add_test(NAME search COMMAND search_tests)
# The same bench from the tests' copy of the engine, whose parameters are
# variables: with FROZEN_PARAMS, the frozen and variable forms must agree.
add_test(NAME bench_variable COMMAND search_tests bench)
set_tests_properties(bench_variable PROPERTIES PASS_REGULAR_EXPRESSION "${BENCH_NODES} nodes ")
# A directory with the 3-4-5-piece Syzygy tables turns on the tablebase
# tests that need them (about 1 GB, so not required).
set(SYZYGY_PATH "" CACHE PATH "Syzygy tables for the tests (optional)")
if (SYZYGY_PATH)
    set_tests_properties(search PROPERTIES ENVIRONMENT "FUNCOMP_SYZYGY=${SYZYGY_PATH}")
endif()
