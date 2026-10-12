// metal/image/nucleus_vae.cpp - see nucleus_vae.h.  Feature maps are ggml's [W, H, C] (a CHW array); the norms run
// over the channels, so they see the map permuted to [C, W, H] (and the attention keeps it so: [C, tokens]).
#include "nucleus_vae.h"

#include <algorithm>
#include <cmath>

namespace strata_image {
namespace {

constexpr int kGraphSize = 4096;
constexpr float kEps = 1e-6f;               // the port's RMSNorm2D
constexpr int64_t kIm2colBytes = int64_t{4} << 30;
constexpr int64_t kAttnChunkBytes = int64_t{512} << 20;

}  // namespace

bool NucleusVAE::load(const std::string& path, ggml_backend_t backend, std::string& err) {
    backend_ = backend;
    if (!wts_.load(path, "nucleus-vae", backend, err)) return false;
    mean_ = wts_.floats("latents_mean");
    std_ = wts_.floats("latents_std");
    if (mean_.size() != 16 || std_.size() != 16) { err = path + ": no latents_mean / latents_std"; return false; }
    return true;
}

std::vector<float> NucleusVAE::decode(const std::vector<float>& z, int h, int w, std::string& err) const {
    if (h < 1 || w < 1 || z.size() != static_cast<size_t>(h) * static_cast<size_t>(w) * 16) {
        err = "decode: the latent's size does not match its shape";
        return {};
    }
    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * kGraphSize + ggml_graph_overhead_custom(kGraphSize, false);
    ip.no_alloc = true;
    GgmlCtx owner(ggml_init(ip));
    ggml_context* ctx = owner.get();
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, kGraphSize, false);
    const auto W = [&](const std::string& n) { return wts_.get(n); };

    auto conv = [&](ggml_tensor* x, const std::string& n) {
        ggml_tensor* k = W(n + ".weight");                                     // [kw, kh, in, out]
        const int p = static_cast<int>(k->ne[0] / 2);
        // im2col + matmul (1.6 s and 4.8 GB of scratch at 1024x1024; conv_2d_direct: 12 s and 1.5 GB) while its
        // buffer (kw * kh * in values a pixel) stays under kIm2colBytes; direct past it (2048x2048's last blocks)
        const int64_t cols = k->ne[0] * k->ne[1] * k->ne[2] * x->ne[0] * x->ne[1] * 4;
        ggml_tensor* y = cols <= kIm2colBytes ? ggml_conv_2d(ctx, k, x, 1, 1, p, p, 1, 1)
                                              : ggml_conv_2d_direct(ctx, k, x, 1, 1, p, p, 1, 1);
        return ggml_add(ctx, y, ggml_reshape_3d(ctx, W(n + ".bias"), 1, 1, k->ne[3]));
    };
    auto norm_cwh = [&](ggml_tensor* x, const std::string& gamma, bool silu) {  // [W, H, C] -> normed [C, W, H]
        ggml_tensor* t = ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3));
        t = ggml_mul(ctx, ggml_rms_norm(ctx, t, kEps), W(gamma));
        return silu ? ggml_silu(ctx, t) : t;
    };
    auto norm = [&](ggml_tensor* x, const std::string& gamma, bool silu) {
        return ggml_cont(ctx, ggml_permute(ctx, norm_cwh(x, gamma, silu), 2, 0, 1, 3));
    };
    auto resnet = [&](ggml_tensor* x, const std::string& p) {
        ggml_tensor* r = conv(norm(x, p + "norm1.gamma", true), p + "conv1");
        r = conv(norm(r, p + "norm2.gamma", true), p + "conv2");
        if (wts_.has(p + "conv_shortcut.weight")) x = conv(x, p + "conv_shortcut");
        return ggml_add(ctx, x, r);
    };
    auto attention = [&](ggml_tensor* x, const std::string& p) {             // one head over every position
        const int64_t C = x->ne[2], N = x->ne[0] * x->ne[1];
        ggml_tensor* t = ggml_reshape_2d(ctx, norm_cwh(x, p + "norm.gamma", false), C, N);
        ggml_tensor* wq = W(p + "to_qkv.weight");
        ggml_tensor* qkv = ggml_add(ctx, ggml_mul_mat(ctx, ggml_reshape_2d(ctx, wq, C, 3 * C), t), W(p + "to_qkv.bias"));
        auto part = [&](int i) {
            return ggml_cont(ctx, ggml_view_2d(ctx, qkv, C, N, qkv->nb[1], static_cast<size_t>(i * C) * sizeof(float)));
        };
        // the queries in chunks of at most kAttnChunkBytes of scores (17 GB in one piece at 2048x2048)
        ggml_tensor* q = part(0);
        ggml_tensor* k = part(1);
        ggml_tensor* vt = ggml_cont(ctx, ggml_transpose(ctx, part(2)));                // [N, C]
        const int64_t cs = std::clamp<int64_t>(kAttnChunkBytes / (N * 4), 256, N);
        ggml_tensor* o = nullptr;
        for (int64_t c0 = 0; c0 < N; c0 += cs) {
            const int64_t n = std::min<int64_t>(cs, N - c0);
            ggml_tensor* qc = n == N ? q : ggml_view_2d(ctx, q, C, n, q->nb[1], static_cast<size_t>(c0) * q->nb[1]);
            ggml_tensor* kq = ggml_mul_mat(ctx, k, qc);                                // [N keys, n queries]
            kq = ggml_soft_max_ext(ctx, kq, nullptr, 1.0f / std::sqrt(static_cast<float>(C)), 0.0f);
            ggml_tensor* oc = ggml_mul_mat(ctx, vt, kq);                               // [C, n]
            o = o ? ggml_concat(ctx, o, oc, 1) : oc;
        }
        ggml_tensor* wp = W(p + "proj.weight");
        o = ggml_add(ctx, ggml_mul_mat(ctx, ggml_reshape_2d(ctx, wp, C, C), o), W(p + "proj.bias"));
        o = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, o, C, x->ne[0], x->ne[1]), 2, 0, 1, 3));
        return ggml_add(ctx, x, o);
    };

    ggml_tensor* zin = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, w, h, 16);
    ggml_set_input(zin);
    ggml_tensor* x = conv(conv(zin, "post_quant_conv"), "decoder.conv_in");
    x = resnet(x, "decoder.mid_block.resnets.0.");
    x = attention(x, "decoder.mid_block.attentions.0.");
    x = resnet(x, "decoder.mid_block.resnets.1.");
    for (int b = 0; wts_.has("decoder.up_blocks." + std::to_string(b) + ".resnets.0.conv1.weight"); ++b) {
        const std::string p = "decoder.up_blocks." + std::to_string(b) + ".";
        for (int r = 0; wts_.has(p + "resnets." + std::to_string(r) + ".conv1.weight"); ++r)
            x = resnet(x, p + "resnets." + std::to_string(r) + ".");
        if (wts_.has(p + "upsamplers.0.resample.1.weight"))
            x = conv(ggml_upscale(ctx, x, 2, GGML_SCALE_MODE_NEAREST), p + "upsamplers.0.resample.1");
    }
    x = conv(norm(x, "decoder.norm_out.gamma", true), "decoder.conv_out");      // [8w, 8h, 3]
    ggml_set_output(x);
    ggml_build_forward_expand(gf, x);

    Gallocr galloc(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_)));
    if (!ggml_gallocr_alloc_graph(galloc.get(), gf)) { err = "not enough GPU memory for the VAE decoder"; return {}; }
    std::vector<float> chw(z.size());                                            // [h, w, 16] -> [16, h, w]
    for (int y = 0; y < h; ++y)
        for (int xx = 0; xx < w; ++xx)
            for (int c = 0; c < 16; ++c)
                chw[(static_cast<size_t>(c) * h + y) * w + xx] = z[(static_cast<size_t>(y) * w + xx) * 16 + c];
    ggml_backend_tensor_set(zin, chw.data(), 0, ggml_nbytes(zin));
    if (ggml_backend_graph_compute(backend_, gf) != GGML_STATUS_SUCCESS) { err = "the VAE graph failed on the GPU"; return {}; }
    const int64_t ow = x->ne[0], oh = x->ne[1];
    std::vector<float> out_chw(static_cast<size_t>(ggml_nelements(x)));
    ggml_backend_tensor_get(x, out_chw.data(), 0, ggml_nbytes(x));
    std::vector<float> img(out_chw.size());                                      // [3, H, W] -> [H, W, 3]
    for (int64_t y = 0; y < oh; ++y)
        for (int64_t xx = 0; xx < ow; ++xx)
            for (int64_t c = 0; c < 3; ++c)
                img[static_cast<size_t>((y * ow + xx) * 3 + c)] = out_chw[static_cast<size_t>((c * oh + y) * ow + xx)];
    return img;
}

}  // namespace strata_image
