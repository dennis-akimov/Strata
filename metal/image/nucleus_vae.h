// metal/image/nucleus_vae.h - Nucleus-Image's VAE decoder (Qwen-Image's, one frame: every Conv3d is a Conv2d) on
// ggml/Metal: the port's nucleus_image/vae.py op for op, weights from tools/image/nucleus_gguf.py --vae.
#pragma once

#include "gguf_weights.h"

#include <string>
#include <vector>

namespace strata_image {

class NucleusVAE {
public:
    bool load(const std::string& path, ggml_backend_t backend, std::string& err);

    // `z` [h, w, 16] row-major (the latent, already z * std + mean) to the picture [8h, 8w, 3] in [-1, 1] (not clipped).
    std::vector<float> decode(const std::vector<float>& z, int h, int w, std::string& err) const;

    const std::vector<float>& mean() const { return mean_; }
    const std::vector<float>& std_dev() const { return std_; }

private:
    Weights wts_;
    ggml_backend_t backend_ = nullptr;                         // not owned
    std::vector<float> mean_, std_;
};

}  // namespace strata_image
