# strata-metal: the macOS / Apple Silicon engine, Strata's line protocol over llama.cpp's Metal backend
# (docs/MACOS.md).  Entered from the top-level CMakeLists.txt with -DSTRATA_ENABLE_METAL=ON; nothing else there
# is configured.  llama.cpp comes at a pinned commit (STRATA_LLAMA_COMMIT), or from STRATA_LLAMA_DIR, a checkout of
# exactly that commit (offline builds).
set(STRATA_LLAMA_COMMIT "71ad0590f4808b6202f9213d166913858c73b1bc")   # 2026-10-08: few-row MMA mat-mul (#30065), +8% decode with MTP
set(STRATA_LLAMA_DIR "" CACHE PATH "a llama.cpp checkout at STRATA_LLAMA_COMMIT with metal/patches/core applied (default: fetched)")

set(GGML_METAL ON CACHE BOOL "" FORCE)
set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)   # the shaders inside the binary: no Xcode needed to run it
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)     # common/speculative: the MTP drafts (--mtp)
set(LLAMA_BUILD_MTMD ON CACHE BOOL "" FORCE)       # strata-vision: the image encoder
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)

if(STRATA_LLAMA_DIR)
  add_subdirectory(${STRATA_LLAMA_DIR} llama.cpp EXCLUDE_FROM_ALL)
else()
  # Strata's own Metal kernels (docs/MACOS.md, "For developers"), as patches on the pinned commit; each one is checked
  # by llama.cpp's test-backend-ops and A/B-tested on the model.  Off by default: on the 2026-10-08 pin they were not
  # measurably faster (docs/MACOS.md).  A new name, so a build folder made with the old ON default does not keep it.
  option(STRATA_METAL_KERNEL_PATCHES "apply metal/patches/*.patch to the fetched llama.cpp" OFF)
  # metal/patches/core/*.patch always: what strata-metal's own options need from llama.cpp (--ablate)
  file(GLOB _patches "${CMAKE_CURRENT_SOURCE_DIR}/patches/core/*.patch")
  list(SORT _patches)
  if(STRATA_METAL_KERNEL_PATCHES)
    file(GLOB _kernel "${CMAKE_CURRENT_SOURCE_DIR}/patches/*.patch")
    list(SORT _kernel)
    list(APPEND _patches ${_kernel})
  endif()
  set(_patch "")
  set(_sig "")
  # patch(1), not git apply: the fetched tree lies inside this repository's build folder, and git apply inside
  # another repository resolves the paths against that repository and silently applies nothing
  find_program(STRATA_PATCH patch REQUIRED)
  foreach(_p ${_patches})
    if(_patch)
      list(APPEND _patch COMMAND)
    else()
      set(_patch PATCH_COMMAND)
    endif()
    list(APPEND _patch "${STRATA_PATCH}" -p1 -N -i "${_p}")
    file(MD5 "${_p}" _h)
    string(APPEND _sig "${_h}")
  endforeach()
  include(FetchContent)
  # a source folder per set of patches (named by their contents): FetchContent reuses a populated folder, so a changed
  # patch, or the kernel patches turned off, would otherwise keep building the old tree
  string(MD5 _sig "${_sig}")
  string(SUBSTRING "${_sig}" 0 10 _sig)
  set(_src "${FETCHCONTENT_BASE_DIR}/llama_cpp-src-${_sig}")
  FetchContent_Declare(llama_cpp URL https://github.com/ggml-org/llama.cpp/archive/${STRATA_LLAMA_COMMIT}.tar.gz
                       SOURCE_DIR "${_src}" ${_patch})
  FetchContent_MakeAvailable(llama_cpp)
endif()

add_executable(strata-metal strata_metal.cpp)
target_compile_features(strata-metal PRIVATE cxx_std_17)
target_compile_definitions(strata-metal PRIVATE STRATA_VERSION="${PROJECT_VERSION}"
                           STRATA_LLAMA_COMMIT="${STRATA_LLAMA_COMMIT}")
target_link_libraries(strata-metal PRIVATE llama llama-common)

# the image engine (Nucleus-Image on ggml/Metal, docs/MACOS.md): its own executable, as strata-vision is.  Hardened as
# bible's C++ skill lists for Apple clang 17 on arm64 (flags the compiler and linker accept there)
add_executable(strata-image ${CMAKE_CURRENT_SOURCE_DIR}/image/strata_image.cpp ${CMAKE_CURRENT_SOURCE_DIR}/image/nucleus_dit.cpp
               ${CMAKE_CURRENT_SOURCE_DIR}/image/text_encoder.cpp ${CMAKE_CURRENT_SOURCE_DIR}/image/hunyuan_image.cpp
               ${CMAKE_CURRENT_SOURCE_DIR}/image/gguf_weights.cpp ${CMAKE_CURRENT_SOURCE_DIR}/image/nucleus_vae.cpp)
target_compile_features(strata-image PRIVATE cxx_std_17)
target_compile_options(strata-image PRIVATE -Wall -Wextra -fstack-protector-strong -ftrivial-auto-var-init=zero
                       -mbranch-protection=standard)
target_link_libraries(strata-image PRIVATE llama ggml z)
# ggml-backend-impl.h (gguf_weights.cpp: a Metal mapping that failed is a buffer with no context, not a null buffer)
get_target_property(STRATA_GGML_SRC ggml SOURCE_DIR)
target_include_directories(strata-image PRIVATE ${STRATA_GGML_SRC})

# the image encoder (tools/vision): the same source as on a PC, on llama.cpp's mtmd with Metal (--gpu)
add_executable(strata-vision ${CMAKE_CURRENT_SOURCE_DIR}/../tools/vision/strata_vision.cpp)
target_compile_features(strata-vision PRIVATE cxx_std_17)
target_link_libraries(strata-vision PRIVATE mtmd llama)
