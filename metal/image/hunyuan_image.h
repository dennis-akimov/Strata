// metal/image/hunyuan_image.h - HunyuanImage-3.0-Instruct-Distil (an 80B mixture of experts, 13B active a token) on
// ggml/Metal: the transformer, the image head and the sampler's step.  The weights come from tools/image/hunyuan_gguf.py,
// the tokens from tools/image/hunyuan_tokens.py; the arithmetic is Tencent's modeling_hunyuan_image_3.py, op for op.
//
// One picture is a prompt prefix (text, causal) and, each step, 3 conditioning tokens (timestep, guidance, timestep_r)
// and the h x w image tokens.  The prefix never sees the image (causal), so its keys and values are computed once;
// each step runs only the 3 + h*w tokens, which attend to the prefix and causally to each other, the image tokens to
// the whole image.  The model is CFG-distilled: one pass a step.
#pragma once

#include "gguf_weights.h"

#include <string>
#include <vector>

namespace strata_image {

class HunyuanImage {
public:
    bool load(const std::string& path, ggml_backend_t backend, std::string& err);

    // The prompt's token ids before <timestep>: their keys and values in every layer, kept for step().
    // The check: `hidden` [P, 4096] in place of the ids' embeddings, and `only` one layer.
    bool prefix(const std::vector<int32_t>& ids, std::string& err, const std::vector<float>* hidden = nullptr,
                int only = -1);

    // The flow velocity [32, h, w] (CHW) of the latent `z` [32, h, w] at timestep `t` (sigma * 1000), with the
    // next step's timestep `r` and the guidance (1000 * scale, as bfloat16).  `layers`, when not null, gets the
    // input tokens and every layer's output, [3 + h*w, 4096] each, for the check against the reference.
    // The check alone: `hidden` [3 + h*w, 4096] in place of the input tokens, and `only` = l: layer l's output;
    // `only` = -2: the image head (UNetUp) on the image tokens of `hidden`, the velocity.
    std::vector<float> step(const std::vector<float>& z, int h, int w, float t, float r, float guidance, std::string& err,
                            std::vector<std::vector<float>>* layers = nullptr, const std::vector<float>* hidden = nullptr,
                            int only = -1);

    int latent_channels() const { return latent_; }
    size_t compute_bytes() const { return compute_bytes_; }

private:
    ggml_tensor* w(const std::string& name) const { return wts_.get(name); }
    ggml_tensor* layer(ggml_context* ctx, ggml_tensor* x, int l, ggml_tensor* pos, ggml_tensor* ff, ggml_tensor* mask,
                       ggml_tensor** k_out, ggml_tensor** v_out) const;

    Weights wts_;
    int hidden_ = 4096, layers_ = 32, heads_ = 32, kv_heads_ = 8, head_dim_ = 128, experts_ = 64, topk_ = 8;
    int latent_ = 32;
    float eps_ = 1e-5f, theta_ = 10000.0f;
    std::vector<float> rope_ff_;                               // [head_dim / 2]: see load()
    // the prefix's keys and values per layer, [head_dim, kv_heads, P] float32, in their own GPU buffer
    GgmlCtx cache_ctx_;
    Buffer cache_buf_;
    std::vector<ggml_tensor*> cache_k_, cache_v_;
    int prefix_len_ = 0;
    size_t compute_bytes_ = 0;
    ggml_backend_t backend_ = nullptr;                         // not owned
};

// The 3D VAE's decoder (AutoencoderKLConv3D) for one latent frame: 4 frames out, the last one kept, as the pipeline.
class HunyuanVAE {
public:
    bool load(const std::string& path, ggml_backend_t backend, std::string& err);   // the same GGUF (vae.* tensors)

    // `z` [32, h, w] CHW, already divided by the scaling factor, to the picture [16h, 16w, 3] in [-1, 1] (not clipped).
    std::vector<float> decode(const std::vector<float>& z, int h, int w, std::string& err) const;

    float scaling_factor() const { return scaling_; }

private:
    Weights wts_;
    float scaling_ = 0.562679178327931f;
    ggml_backend_t backend_ = nullptr;                         // not owned
};

}  // namespace strata_image
