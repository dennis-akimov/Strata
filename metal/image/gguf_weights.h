// metal/image/gguf_weights.h - a GGUF file's tensors on the GPU, memory-mapped: the weights are the file's pages
// (clean, file-backed: macOS can drop them under pressure and read them again), not a copy, as llama.cpp's mmap load.
// Only the 1-D BF16 tensors (biases, norm weights) are copied, to float32 for the elementwise ops.
#pragma once

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata_image {

// RAII owners for the C handles (bible C++: no raw owning pointers, cg R.11 / tob-raw-owning-t)
struct GgmlFree { void operator()(ggml_context* c) const { ggml_free(c); } };
struct GgufFree { void operator()(gguf_context* c) const { gguf_free(c); } };
struct BackendFree { void operator()(ggml_backend_t b) const { ggml_backend_free(b); } };
struct BufferFree { void operator()(ggml_backend_buffer_t b) const { ggml_backend_buffer_free(b); } };
struct GallocrFree { void operator()(ggml_gallocr_t g) const { ggml_gallocr_free(g); } };
using GgmlCtx = std::unique_ptr<ggml_context, GgmlFree>;
using GgufCtx = std::unique_ptr<gguf_context, GgufFree>;
using Backend = std::unique_ptr<std::remove_pointer_t<ggml_backend_t>, BackendFree>;
using Buffer = std::unique_ptr<std::remove_pointer_t<ggml_backend_buffer_t>, BufferFree>;
using Gallocr = std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, GallocrFree>;

class Weights {
public:
    Weights() = default;
    Weights(const Weights&) = delete;
    Weights& operator=(const Weights&) = delete;
    ~Weights();

    // Maps `path` (a GGUF of architecture `arch`) for `backend`'s device; false with `err` set otherwise.
    bool load(const std::string& path, const std::string& arch, ggml_backend_t backend, std::string& err);

    ggml_tensor* get(const std::string& name) const;          // a weight, or abort with its name (a damaged file)
    bool has(const std::string& name) const { return by_name_.count(name) != 0; }
    uint32_t u32(const std::string& key, uint32_t def) const;
    float f32(const std::string& key, float def) const;
    std::vector<float> floats(const std::string& key) const;    // an array of numbers (empty when absent)
    std::string str(const std::string& key) const;

private:
    std::string arch_;
    GgufCtx g_;
    GgmlCtx meta_, ctx_, owned_ctx_;
    Buffer mapped_, owned_;
    void* map_ = nullptr;
    size_t map_len_ = 0;
    std::unordered_map<std::string, ggml_tensor*> by_name_;
};

}  // namespace strata_image
