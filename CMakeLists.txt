cmake_minimum_required(VERSION 3.24)
project(zorlo C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# executables next to the ggml/llama DLLs so they run without PATH changes
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# llama.cpp: only the libraries (ggml kernels, GGUF, tokenizer); its own tools are built separately
set(LLAMA_CURL           OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON   ON  CACHE BOOL "" FORCE)  # chat templates (Jinja), tool-call and reasoning parsers
if (GGML_CPU_ALL_VARIANTS)
    set(GGML_NATIVE      OFF CACHE BOOL "" FORCE)  # release: one ggml-cpu DLL per CPU type, picked at load time
else()
    set(GGML_NATIVE      ON  CACHE BOOL "" FORCE)
endif()
# apply our llama.cpp patches once (a no-op when already applied)
foreach(p ${CMAKE_SOURCE_DIR}/patches/ggml-tiled-vnni-same-rounding.patch ${CMAKE_SOURCE_DIR}/patches/ggml-cpu-float-avx2-order.patch)
    execute_process(COMMAND git apply --reverse --check ${p} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/third_party/llama.cpp
                    RESULT_VARIABLE applied OUTPUT_QUIET ERROR_QUIET)
    if (NOT applied EQUAL 0)
        execute_process(COMMAND git apply ${p} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/third_party/llama.cpp RESULT_VARIABLE r)
        if (NOT r EQUAL 0)
            message(FATAL_ERROR "cannot apply ${p}")
        endif()
    endif()
endforeach()

add_subdirectory(third_party/llama.cpp EXCLUDE_FROM_ALL)

# float math (f32/f16 matmul incl. the router, softmax, silu/exp, flash attention) in the AVX2 order on AVX-512
# builds too, so every x86 CPU variant gives the same output (patches/ggml-cpu-float-avx2-order.patch);
# quantized and VNNI kernels are unchanged
get_property(ggml_targets DIRECTORY third_party/llama.cpp/ggml/src PROPERTY BUILDSYSTEM_TARGETS)
foreach(t ${ggml_targets})
    if (t MATCHES "^ggml-cpu(-[a-z0-9_.]+)?$" AND NOT t MATCHES "-feats$")
        target_compile_definitions(${t} PRIVATE GGML_CPU_X86_FLOAT_AVX2)
    endif()
endforeach()

# ggml's AVX-512 VNNI kernels (tiled K-quant matmul, ~prefill): MSVC has no /arch switch for VNNI, so define
# it when the build machine has it (GGML_NATIVE targets this machine anyway). Same results as the AVX2
# kernel with patches/ggml-tiled-vnni-same-rounding.patch applied to the submodule.
if (MSVC AND GGML_NATIVE)
    file(WRITE ${CMAKE_BINARY_DIR}/vnni_check.c [[
#include <intrin.h>
int main(void) {
    int r[4];
    __cpuidex(r, 7, 0);
    return (r[2] & (1 << 11)) && (r[1] & (1 << 31)) && (r[1] & (1 << 17)) ? 0 : 1;  /* VNNI, VL, DQ */
}
]])
    try_run(ZORLO_VNNI_RUN ZORLO_VNNI_COMPILE ${CMAKE_BINARY_DIR}/vnni_check ${CMAKE_BINARY_DIR}/vnni_check.c)
    if (ZORLO_VNNI_COMPILE AND ZORLO_VNNI_RUN EQUAL 0)
        message(STATUS "zorlo: CPU has AVX-512 VNNI, enabling ggml's VNNI kernels")
        target_compile_definitions(ggml-cpu PRIVATE __AVX512VNNI__)
    endif()
endif()

add_library(zorlo-engine STATIC
    src/engine/pack.cpp
    src/engine/expert_cache.cpp
    src/engine/moe_model.cpp)
target_include_directories(zorlo-engine PUBLIC src third_party/llama.cpp/vendor)
# ggml only: the CPU backend is reached through the registry (moe_model.cpp), never linked by symbol
target_link_libraries(zorlo-engine PUBLIC ggml)

if (NOT TARGET cpp-httplib)
    add_subdirectory(third_party/llama.cpp/vendor/cpp-httplib)
endif()

add_executable(zorlo src/cli/main.cpp src/cli/pull.cpp src/cli/start.cpp src/cli/agent.cpp src/pack/pack.cpp)
# the agent app (ui/) is served from next to zorlo.exe; keep the build folder's copy current
add_custom_command(TARGET zorlo POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/ui $<TARGET_FILE_DIR:zorlo>/ui)  # `zorlo pack` is zorlo-pack built in
target_compile_definitions(zorlo PRIVATE ZORLO_PACK_IN_CLI)
target_include_directories(zorlo PRIVATE third_party/llama.cpp/vendor/cpp-httplib)
target_link_libraries(zorlo PRIVATE zorlo-engine llama llama-common cpp-httplib)
if (WIN32)
    target_link_libraries(zorlo PRIVATE winhttp bcrypt)  # zorlo pull: HTTPS download and SHA-256, both in Windows
endif()

# Zorlo Desktop: the app window + tray icon around zorlo.exe. Needs the WebView2 SDK: run tools/get-webview2.ps1 once.
set(ZORLO_WEBVIEW2 ${CMAKE_SOURCE_DIR}/third_party/webview2/build/native)
if (WIN32 AND MSVC AND EXISTS ${ZORLO_WEBVIEW2}/include/WebView2.h)
    add_executable(zorlo-desktop WIN32 src/desktop/desktop.cpp src/desktop/desktop.rc)
    set_target_properties(zorlo-desktop PROPERTIES OUTPUT_NAME "Zorlo Desktop")
    target_include_directories(zorlo-desktop PRIVATE ${ZORLO_WEBVIEW2}/include third_party/llama.cpp/vendor)
    target_compile_definitions(zorlo-desktop PRIVATE UNICODE _UNICODE)
    target_link_options(zorlo-desktop PRIVATE /MANIFEST:NO)  # desktop.rc carries the manifest
    target_link_libraries(zorlo-desktop PRIVATE ${ZORLO_WEBVIEW2}/x64/WebView2LoaderStatic.lib winhttp comctl32 shlwapi version advapi32 shell32 ole32)
    add_dependencies(zorlo-desktop zorlo)
else()
    message(STATUS "Zorlo Desktop skipped: run tools/get-webview2.ps1 to build it")
endif()

add_executable(zorlo-pack src/pack/pack.cpp)
target_link_libraries(zorlo-pack PRIVATE ggml)

add_executable(zorlo-trace tools/trace/trace.cpp)
target_link_libraries(zorlo-trace PRIVATE llama ggml)

add_executable(zorlo-golden tools/golden/golden.cpp)
target_link_libraries(zorlo-golden PRIVATE llama ggml)
