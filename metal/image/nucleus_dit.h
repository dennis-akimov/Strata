// metal/image/nucleus_dit.h - Nucleus-Image's DiT (17B mixture of experts, expert-choice routing) on ggml/Metal,
// the image half of strata-image.  The weights come from tools/image/nucleus_gguf.py; the arithmetic is the port's
// (treadon/mlx-nucleus-image nucleus_image/dit.py), op for op, and tools/image/check_dit.py holds it to that port's
// float32 output block by block.
#pragma once

#include "gguf_weights.h"

#include <string>
#include <vector>

namespace strata_image {

struct HParams {
    int hidden = 2048, heads = 16, kv_heads = 4, head_dim = 128, layers = 32, experts = 64, moe_dim = 1344;
    int text_dim = 4096, in_channels = 64, out_channels = 16, patch = 2;
    float route_scale = 2.5f, rope_theta = 10000.0f;
    std::vector<float> capacity;      // per block; 0: a dense FFN
    std::vector<int> axes = {16, 56, 56};
};

// The conditioning of one forward pass computed on the host (or taken from a reference dump, to test the blocks alone)
struct Conditioning {
    std::vector<float> t_emb;         // [hidden]: the sinusoidal timestep embedding (before the MLP)
    std::vector<float> img_cos, img_sin, txt_cos, txt_sin;   // [tokens, head_dim / 2]
};

// One prompt's hidden states [T, text_dim] (not owned)
struct Text {
    const std::vector<float>* h;
    int T;
};

class NucleusDiT {
public:
    // Maps the GGUF for the GPU (gguf_weights.h); false with `err` set when the file is not a nucleus-dit or is damaged.
    bool load(const std::string& path, ggml_backend_t backend, std::string& err);

    // The DiT on `x` [S, in_channels] (row-major, tokens of a grid_h x grid_w latent grid) and the prompt's hidden states
    // `txt` [T, text_dim] at time `t` (sigma, 0..1): the velocity [S, out_channels].  `blocks`, when not null, gets every
    // transformer block's output ([S, hidden] each), for the check against the port.
    std::vector<float> forward(const std::vector<float>& x, int grid_h, int grid_w, const std::vector<float>& txt, int T,
                               float t, const Conditioning* given, std::vector<std::vector<float>>* blocks,
                               std::string& err);

    // The DiT for several prompts on the same `x` (the CFG's prompt and negative prompt): their velocities, in one
    // graph whose routed experts run batched (each expert's weights read once for all of them).
    std::vector<std::vector<float>> forward_branches(const std::vector<float>& x, int grid_h, int grid_w,
                                                     const std::vector<Text>& texts, float t, std::string& err);

    // One transformer block alone on a given hidden state `h` [S, hidden] (the check: the port's previous block in,
    // this block out - so an error is this block's, not one carried in from the blocks before it).
    std::vector<float> block(int layer, const std::vector<float>& h, int grid_h, int grid_w, const std::vector<float>& txt,
                             int T, float t, const Conditioning* given, std::string& err);

    // The host-side conditioning the forward pass uses (exposed for the check)
    Conditioning conditioning(int grid_h, int grid_w, int T, float t) const;

    const HParams& hp() const { return hp_; }
    size_t compute_bytes() const { return compute_bytes_; }     // the last forward's activations (GPU memory)

private:
    std::vector<std::vector<float>> run(const std::vector<float>& x, int grid_h, int grid_w, const std::vector<Text>& texts,
                                        float t, const Conditioning* given, std::vector<std::vector<float>>* blocks,
                                        int only, std::string& err);
    ggml_tensor* w(const std::string& name) const { return wts_.get(name); }
    ggml_tensor* linear(ggml_context* ctx, ggml_tensor* x, const std::string& name, bool bias) const;

    HParams hp_;
    Weights wts_;
    size_t compute_bytes_ = 0;
    ggml_backend_t backend_ = nullptr;                         // not owned
};

}  // namespace strata_image
