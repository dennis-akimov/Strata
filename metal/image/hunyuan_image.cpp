// metal/image/hunyuan_image.cpp - see hunyuan_image.h.  Shapes are ggml's, innermost first: tokens are [features,
// tokens], feature maps [W, H, C] (a CHW array), so a weight stored [out, in] multiplies as ggml_mul_mat(W, x).
#include "hunyuan_image.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace strata_image {
namespace {

constexpr int kGraphSize = 32768;
constexpr int64_t kKeyBlock = 64;            // flash attention's tensor-API kernel: keys a block (nucleus_dit.cpp)
constexpr float kGroupNormEps = 1e-5f;       // nn.GroupNorm's default (the image head's normalization())
constexpr float kVaeEps = 1e-6f;
constexpr int kFreqDim = 256;                // TimestepEmbedder's frequency_embedding_size
constexpr int64_t kIm2colBytes = int64_t{4} << 30;
constexpr int64_t kAttnChunkBytes = int64_t{512} << 20;

GgmlCtx new_graph_ctx() {
    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * kGraphSize + ggml_graph_overhead_custom(kGraphSize, false);
    ip.no_alloc = true;
    return GgmlCtx(ggml_init(ip));
}

// timestep_embedding(t, 256): [cos(t f_i), sin(t f_i)], f_i = 10000^(-i / 128), float32 as the reference
void sinusoid(float t, float* out) {
    const int half = kFreqDim / 2;
    for (int i = 0; i < half; ++i) {
        const float f = std::exp(-9.210340371976184f * static_cast<float>(i) / static_cast<float>(half));
        out[i] = std::cos(t * f);
        out[half + i] = std::sin(t * f);
    }
}

// a convolution with "same" padding: im2col + matmul while its buffer stays under kIm2colBytes, direct past it
ggml_tensor* conv2d(ggml_context* ctx, ggml_tensor* k, ggml_tensor* x) {
    const int p = static_cast<int>(k->ne[0] / 2);
    const int64_t cols = k->ne[0] * k->ne[1] * k->ne[2] * x->ne[0] * x->ne[1] * static_cast<int64_t>(ggml_type_size(k->type));
    return cols <= kIm2colBytes ? ggml_conv_2d(ctx, k, x, 1, 1, p, p, 1, 1) : ggml_conv_2d_direct(ctx, k, x, 1, 1, p, p, 1, 1);
}

ggml_tensor* channel(ggml_context* ctx, ggml_tensor* v) { return ggml_reshape_3d(ctx, v, 1, 1, v->ne[0]); }

}  // namespace

bool HunyuanImage::load(const std::string& path, ggml_backend_t backend, std::string& err) {
    backend_ = backend;
    if (!wts_.load(path, "hunyuan-image3", backend, err)) return false;
    hidden_ = static_cast<int>(wts_.u32("hidden_size", 4096));
    layers_ = static_cast<int>(wts_.u32("num_hidden_layers", 32));
    heads_ = static_cast<int>(wts_.u32("num_attention_heads", 32));
    kv_heads_ = static_cast<int>(wts_.u32("num_key_value_heads", 8));
    head_dim_ = static_cast<int>(wts_.u32("head_dim", 128));
    experts_ = static_cast<int>(wts_.u32("num_experts", 64));
    topk_ = static_cast<int>(wts_.u32("moe_topk", 8));
    latent_ = static_cast<int>(wts_.u32("latent_channels", 32));
    eps_ = wts_.f32("rms_norm_eps", 1e-5f);
    theta_ = wts_.f32("rope_theta", 10000.0f);
    // only this model's shapes: the graph and the RoPE sections assume them (cg ES.46: checked before any use)
    if (wts_.str("hunyuan-image3.rope_layout") != "sections_yx" || hidden_ != 4096 || layers_ < 1 || layers_ > 64 ||
        heads_ != 32 || kv_heads_ != 8 || head_dim_ != 128 || experts_ < 1 || experts_ > 256 || topk_ < 1 ||
        topk_ > experts_ || latent_ != 32) {
        err = path + ": not a HunyuanImage-3.0 GGUF of tools/image/hunyuan_gguf.py";
        return false;
    }
    // the reference rotates pair i (dims i, i + 64) at frequency theta^(-2i/128), by y for even i and x for odd i; the
    // converter put the even pairs first (sections: 32 on y, 32 on x).  ggml_rope_multi's frequency of pair p is
    // theta^(-2p/128): the factor that makes it the reference's
    rope_ff_.clear();
    const int half = head_dim_ / 2;
    for (int p = 0; p < half; ++p) {
        const int i = p < half / 2 ? 2 * p : 2 * (p - half / 2) + 1;
        rope_ff_.push_back(std::pow(theta_, -2.0f * static_cast<float>(p) / static_cast<float>(head_dim_)) /
                           std::pow(theta_, -2.0f * static_cast<float>(i) / static_cast<float>(head_dim_)));
    }
    return true;
}

// One decoder layer on x [D, n]: pre-norm attention (QK RMS norm after RoPE) over the cached prefix and these tokens,
// then the mixture of experts (softmax router, top 8 renormalized, plus the shared MLP).  k_out / v_out, when not null,
// get this layer's keys and values [hd, kv_heads, n] (the prefix pass keeps them).
ggml_tensor* HunyuanImage::layer(ggml_context* ctx, ggml_tensor* x, int l, ggml_tensor* pos, ggml_tensor* ff,
                                 ggml_tensor* mask, ggml_tensor** k_out, ggml_tensor** v_out) const {
    const std::string p = "model.layers." + std::to_string(l) + ".";
    const int64_t n = x->ne[1], D = hidden_, hd = head_dim_;
    int sections[GGML_MROPE_SECTIONS] = {head_dim_ / 4, head_dim_ / 4, 0, 0};
    auto rms = [&](ggml_tensor* v, const std::string& wn) { return ggml_mul(ctx, ggml_rms_norm(ctx, v, eps_), w(wn)); };
    auto rope = [&](ggml_tensor* v) {
        return ggml_rope_multi(ctx, v, pos, ff, head_dim_, sections, GGML_ROPE_TYPE_MROPE, 0, theta_, 1.0f, 0.0f, 1.0f,
                               0.0f, 0.0f);
    };

    ggml_tensor* hn = rms(x, p + "input_layernorm.weight");
    ggml_tensor* q = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, w(p + "self_attn.q_proj.weight"), hn), hd, heads_, n);
    ggml_tensor* k = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, w(p + "self_attn.k_proj.weight"), hn), hd, kv_heads_, n);
    ggml_tensor* v = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, w(p + "self_attn.v_proj.weight"), hn), hd, kv_heads_, n);
    q = rms(rope(q), p + "self_attn.query_layernorm.weight");
    k = rms(rope(k), p + "self_attn.key_layernorm.weight");
    if (k_out) { *k_out = k; *v_out = v; }
    ggml_tensor* K = k;
    ggml_tensor* V = v;
    if (prefix_len_ > 0 && !k_out) {
        K = ggml_concat(ctx, cache_k_[static_cast<size_t>(l)], k, 2);
        V = ggml_concat(ctx, cache_v_[static_cast<size_t>(l)], v, 2);
    }
    const int64_t pad = mask->ne[0] - K->ne[2];
    auto kv = [&](ggml_tensor* t) {                                          // [hd, kvh, L] -> [hd, Lp, kvh] F16
        return ggml_cast(ctx, ggml_pad(ctx, ggml_cont(ctx, ggml_permute(ctx, t, 0, 2, 1, 3)), 0, static_cast<int>(pad), 0, 0),
                         GGML_TYPE_F16);
    };
    ggml_tensor* o = ggml_flash_attn_ext(ctx, ggml_permute(ctx, q, 0, 2, 1, 3), kv(K), kv(V), mask,
                                         1.0f / std::sqrt(static_cast<float>(hd)), 0.0f, 0.0f);
    ggml_prec_set_acc(o, GGML_PREC_F32);
    o = ggml_mul_mat(ctx, w(p + "self_attn.o_proj.weight"), ggml_reshape_2d(ctx, o, D, n));
    x = ggml_add(ctx, x, o);

    hn = rms(x, p + "post_attention_layernorm.weight");
    // the shared MLP and each expert: x1 * silu(x2), x1 the first half of gate_and_up_proj (ggml_swiglu_swapped)
    ggml_tensor* shared = ggml_mul_mat(ctx, w(p + "mlp.shared_mlp.down_proj.weight"),
                                       ggml_swiglu_swapped(ctx, ggml_mul_mat(ctx, w(p + "mlp.shared_mlp.gate_and_up_proj.weight"), hn)));
    ggml_tensor* probs = ggml_soft_max(ctx, ggml_mul_mat(ctx, w(p + "mlp.gate.wg.weight"), hn));          // [E, n]
    ggml_tensor* sel = ggml_argsort_top_k(ctx, probs, topk_);                                              // [k, n]
    ggml_tensor* wk = ggml_reshape_2d(ctx, ggml_get_rows(ctx, ggml_reshape_3d(ctx, probs, 1, experts_, n), sel), topk_, n);
    wk = ggml_div(ctx, wk, ggml_clamp(ctx, ggml_sum_rows(ctx, wk), 1e-8f, std::numeric_limits<float>::infinity()));
    ggml_tensor* gu = ggml_mul_mat_id(ctx, w(p + "mlp.experts.gate_and_up_proj"), ggml_reshape_3d(ctx, hn, D, 1, n), sel);
    ggml_tensor* ye = ggml_mul_mat_id(ctx, w(p + "mlp.experts.down_proj"), ggml_swiglu_swapped(ctx, gu), sel);  // [D, k, n]
    ye = ggml_mul(ctx, ye, ggml_reshape_3d(ctx, wk, 1, topk_, n));
    ggml_tensor* moe = nullptr;
    for (int j = 0; j < topk_; ++j) {
        ggml_tensor* yj = ggml_view_2d(ctx, ye, D, n, ye->nb[2], static_cast<size_t>(j) * ye->nb[1]);
        moe = moe ? ggml_add(ctx, moe, yj) : yj;
    }
    return ggml_add(ctx, x, ggml_add(ctx, shared, moe));
}

bool HunyuanImage::prefix(const std::vector<int32_t>& ids, std::string& err, const std::vector<float>* hidden, int only) {
    const size_t rows = hidden ? hidden->size() / static_cast<size_t>(hidden_) : ids.size();
    if (rows < 1 || rows > 16384 || (hidden && hidden->size() % static_cast<size_t>(hidden_))) {
        err = "prefix: from 1 to 16384 tokens (whole rows)";
        return false;
    }
    const int P = static_cast<int>(rows);
    const int64_t vocab = w("model.wte.weight")->ne[1];
    if (!hidden && std::any_of(ids.begin(), ids.end(), [&](int32_t id) { return id < 0 || id >= vocab; })) {
        err = "prefix: a token id outside the vocabulary";               // get_rows would read past the table
        return false;
    }
    // the cache: [hd, kv_heads, P] float32 a layer, keys and values
    ggml_init_params cp{};
    cp.mem_size = ggml_tensor_overhead() * static_cast<size_t>(2 * layers_ + 1);
    cp.no_alloc = true;
    prefix_len_ = 0;
    cache_k_.clear();
    cache_v_.clear();
    cache_buf_.reset();
    cache_ctx_.reset(ggml_init(cp));
    for (int l = 0; l < layers_; ++l) {
        cache_k_.push_back(ggml_new_tensor_3d(cache_ctx_.get(), GGML_TYPE_F32, head_dim_, kv_heads_, P));
        cache_v_.push_back(ggml_new_tensor_3d(cache_ctx_.get(), GGML_TYPE_F32, head_dim_, kv_heads_, P));
    }
    cache_buf_.reset(ggml_backend_alloc_ctx_tensors(cache_ctx_.get(), backend_));
    if (!cache_buf_) { err = "not enough GPU memory for the prompt's keys and values"; return false; }

    GgmlCtx owner = new_graph_ctx();
    ggml_context* ctx = owner.get();
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, kGraphSize, false);
    const int64_t Lp = (P + kKeyBlock - 1) / kKeyBlock * kKeyBlock;
    ggml_tensor* tok = hidden ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_, P) : ggml_new_tensor_1d(ctx, GGML_TYPE_I32, P);
    ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * P);
    ggml_tensor* ff = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim_ / 2);
    ggml_tensor* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, Lp, P);
    for (ggml_tensor* t : {tok, pos, ff, mask}) ggml_set_input(t);
    ggml_tensor* x = hidden ? tok : ggml_get_rows(ctx, w("model.wte.weight"), tok);                     // [D, P]
    for (int l = 0; l < layers_; ++l) {
        if (only >= 0 && l != only) continue;
        ggml_tensor *k = nullptr, *v = nullptr;
        x = layer(ctx, x, l, pos, ff, mask, &k, &v);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, k, cache_k_[static_cast<size_t>(l)]));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, v, cache_v_[static_cast<size_t>(l)]));
    }
    Gallocr galloc(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_)));
    if (!ggml_gallocr_alloc_graph(galloc.get(), gf)) { err = "not enough GPU memory for the prompt pass"; return false; }
    // a layer alone copies its keys and values before attention: the mask then has no buffer, nothing to set
    auto set = [](ggml_tensor* tsr, const void* data) { if (tsr->buffer) ggml_backend_tensor_set(tsr, data, 0, ggml_nbytes(tsr)); };
    set(tok, hidden ? static_cast<const void*>(hidden->data()) : ids.data());
    std::vector<int32_t> pv(static_cast<size_t>(4 * P), 0);
    for (int i = 0; i < P; ++i) pv[static_cast<size_t>(i)] = pv[static_cast<size_t>(P + i)] = i;   // text: y = x = index
    set(pos, pv.data());
    set(ff, rope_ff_.data());
    std::vector<ggml_fp16_t> m(static_cast<size_t>(Lp * P));
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity());
    for (int i = 0; i < P; ++i)
        for (int64_t j = 0; j < Lp; ++j) m[static_cast<size_t>(i * Lp + j)] = j <= i ? zero : ninf;            // causal
    set(mask, m.data());
    if (ggml_backend_graph_compute(backend_, gf) != GGML_STATUS_SUCCESS) { err = "the prompt pass failed on the GPU"; return false; }
    prefix_len_ = P;
    return true;
}

std::vector<float> HunyuanImage::step(const std::vector<float>& z, int h, int w_, float t, float r, float guidance,
                                      std::string& err, std::vector<std::vector<float>>* layers,
                                      const std::vector<float>* hidden, int only) {
    if (h < 1 || w_ < 1 || h > 256 || w_ > 256) { err = "step: the latent grid must be from 1 to 256 a side"; return {}; }
    const int S = h * w_, n = 3 + S, D = hidden_, P = prefix_len_;
    if (P < 1) { err = "step: no prompt (prefix)"; return {}; }
    if (static_cast<int64_t>(z.size()) != static_cast<int64_t>(latent_) * S ||
        (hidden && static_cast<int64_t>(hidden->size()) != static_cast<int64_t>(n) * D)) {
        err = "step: the latent's size";
        return {};
    }
    GgmlCtx owner = new_graph_ctx();
    ggml_context* ctx = owner.get();
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, kGraphSize, false);
    const int64_t Lp = (P + n + kKeyBlock - 1) / kKeyBlock * kKeyBlock;
    ggml_tensor* zin = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, w_, h, latent_);
    ggml_tensor* hin = hidden ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, n) : nullptr;
    ggml_tensor* freq = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kFreqDim, 3);       // t, guidance, r
    ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * n);
    ggml_tensor* ff = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim_ / 2);
    ggml_tensor* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, Lp, n);
    for (ggml_tensor* x : {zin, freq, pos, ff, mask}) ggml_set_input(x);
    if (hin) ggml_set_input(hin);

    auto col = [&](int i) { return ggml_view_2d(ctx, freq, kFreqDim, 1, freq->nb[1], static_cast<size_t>(i) * freq->nb[1]); };
    auto linear = [&](ggml_tensor* x, const std::string& name) {
        return ggml_add(ctx, ggml_mul_mat(ctx, w(name + ".weight"), x), w(name + ".bias"));
    };
    auto embed = [&](const std::string& name, int i) {                              // TimestepEmbedder: [D, 1]
        return linear(ggml_gelu_erf(ctx, linear(col(i), name + ".mlp.0")), name + ".mlp.2");
    };
    auto conv = [&](ggml_tensor* x, const std::string& name) {
        return ggml_add(ctx, conv2d(ctx, w(name + ".weight"), x), channel(ctx, w(name + ".bias")));
    };
    auto gnorm = [&](ggml_tensor* x, const std::string& name) {                    // GroupNorm(32) on [W, H, C]
        return ggml_add(ctx, ggml_mul(ctx, ggml_group_norm(ctx, x, 32, kGroupNormEps), channel(ctx, w(name + ".weight"))),
                        channel(ctx, w(name + ".bias")));
    };
    auto resblock = [&](ggml_tensor* x, const std::string& p, ggml_tensor* emb) {
        ggml_tensor* hh = conv(ggml_silu(ctx, gnorm(x, p + "in_layers.0")), p + "in_layers.2");
        const int64_t C = hh->ne[2];
        ggml_tensor* eo = linear(ggml_silu(ctx, emb), p + "emb_layers.1");                 // [2C, 1]: scale, shift
        ggml_tensor* scale = ggml_view_1d(ctx, eo, C, 0);
        ggml_tensor* shift = ggml_view_1d(ctx, eo, C, static_cast<size_t>(C) * sizeof(float));
        hh = ggml_add(ctx, ggml_mul(ctx, gnorm(hh, p + "out_layers.0"), channel(ctx, ggml_scale_bias(ctx, ggml_cont(ctx, scale), 1.0f, 1.0f))),
                      channel(ctx, ggml_cont(ctx, shift)));
        hh = conv(ggml_silu(ctx, hh), p + "out_layers.3");
        return ggml_add(ctx, wts_.has(p + "skip_connection.weight") ? conv(x, p + "skip_connection") : x, hh);
    };

    // the input tokens: [timestep, guidance, timestep_r, image (UNetDown: conv, ResBlock on time_embed(t))]
    ggml_tensor* x = hin;
    if (!x) {
        ggml_tensor* img = resblock(conv(zin, "patch_embed.model.0"), "patch_embed.model.1.", embed("time_embed", 0));
        img = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, img, 1, 2, 0, 3)), D, S);            // [D, S], row-major
        x = ggml_concat(ctx, ggml_concat(ctx, embed("timestep_emb", 0), embed("guidance_emb", 1), 1),
                        embed("timestep_r_emb", 2), 1);
        x = ggml_concat(ctx, x, img, 1);                                                                   // [D, n]
    }
    std::vector<ggml_tensor*> outs;
    if (layers) { ggml_set_output(x); outs.push_back(x); }
    for (int l = 0; l < layers_ && only != -2; ++l) {
        if (only >= 0 && l != only) continue;
        x = layer(ctx, x, l, pos, ff, mask, nullptr, nullptr);
        if (layers) { ggml_set_output(x); outs.push_back(x); }
    }
    // the image tokens' hidden states (no final norm: the reference's ln_f is the text head's) -> UNetUp
    ggml_tensor* y = x;
    if (only < 0) {
        y = ggml_view_2d(ctx, x, D, S, x->nb[1], 3 * x->nb[1]);
        y = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, ggml_cont(ctx, y), D, w_, h), 2, 0, 1, 3)); // [w, h, D]
        y = resblock(y, "final_layer.model.0.", embed("time_embed_2", 0));
        y = conv(ggml_silu(ctx, gnorm(y, "final_layer.model.1.0")), "final_layer.model.1.2");             // [w, h, 32]
    }
    ggml_set_output(y);
    ggml_build_forward_expand(gf, y);

    Gallocr galloc(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_)));
    if (!ggml_gallocr_alloc_graph(galloc.get(), gf)) { err = "not enough GPU memory for a step"; return {}; }
    compute_bytes_ = ggml_gallocr_get_buffer_size(galloc.get(), 0);
    // an input the graph does not use (the latent of a layer alone) has no buffer: nothing to set
    auto set = [](ggml_tensor* tsr, const void* data) { if (tsr && tsr->buffer) ggml_backend_tensor_set(tsr, data, 0, ggml_nbytes(tsr)); };
    set(zin, z.data());
    if (hidden) set(hin, hidden->data());
    std::vector<float> fr(static_cast<size_t>(3 * kFreqDim));
    sinusoid(t, fr.data());
    sinusoid(guidance, fr.data() + kFreqDim);
    sinusoid(r, fr.data() + 2 * kFreqDim);
    set(freq, fr.data());
    // positions: the 3 conditioning tokens at P, P+1, P+2 (y = x = index); image token (i, j) at
    // L + (S - h)/2 + i, L + (S - w)/2 + j, L = P + 3 (build_2d_rope)
    std::vector<int32_t> pv(static_cast<size_t>(4 * n), 0);
    const int L = P + 3;
    for (int i = 0; i < 3; ++i) pv[static_cast<size_t>(i)] = pv[static_cast<size_t>(n + i)] = P + i;
    for (int i = 0; i < h; ++i)
        for (int j = 0; j < w_; ++j) {
            const size_t k = static_cast<size_t>(3 + i * w_ + j);
            pv[k] = L + (S - h) / 2 + i;
            pv[static_cast<size_t>(n) + k] = L + (S - w_) / 2 + j;
        }
    set(pos, pv.data());
    set(ff, rope_ff_.data());
    // the mask: the conditioning tokens causal, the image tokens see the prefix, the conditioning and the whole image
    std::vector<ggml_fp16_t> m(static_cast<size_t>(Lp * n));
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity());
    for (int i = 0; i < n; ++i) {
        const int64_t last = i < 3 ? P + i : P + n - 1;
        for (int64_t j = 0; j < Lp; ++j) m[static_cast<size_t>(i * Lp + j)] = j <= last ? zero : ninf;
    }
    set(mask, m.data());
    if (ggml_backend_graph_compute(backend_, gf) != GGML_STATUS_SUCCESS) { err = "a step failed on the GPU"; return {}; }
    if (layers) {
        layers->clear();
        for (ggml_tensor* o : outs) {
            std::vector<float> v(static_cast<size_t>(ggml_nelements(o)));
            ggml_backend_tensor_get(o, v.data(), 0, ggml_nbytes(o));
            layers->push_back(std::move(v));
        }
    }
    std::vector<float> out(static_cast<size_t>(ggml_nelements(y)));
    ggml_backend_tensor_get(y, out.data(), 0, ggml_nbytes(y));
    return out;
}

// ---------------------------------------------------------------------------------------------------------- the VAE

bool HunyuanVAE::load(const std::string& path, ggml_backend_t backend, std::string& err) {
    backend_ = backend;
    if (!wts_.load(path, "hunyuan-image3", backend, err)) return false;
    scaling_ = wts_.f32("vae_scaling_factor", scaling_);
    return true;
}

// Feature maps are lists of frames, each [W, H, C].  A Conv3d (3x3x3, zero padding) is three 2D convolutions, one per
// temporal tap (.t0 .t1 .t2: frames f-1, f, f+1); a GroupNorm normalizes over every frame (the frames stacked on H).
std::vector<float> HunyuanVAE::decode(const std::vector<float>& z, int h, int w, std::string& err) const {
    const std::vector<float> chans = wts_.floats("vae_block_out_channels");
    const int zc = static_cast<int>(wts_.u32("latent_channels", 32));
    if (h < 1 || w < 1 || z.size() != static_cast<size_t>(h) * static_cast<size_t>(w) * static_cast<size_t>(zc) ||
        chans.size() != 5) {
        err = "decode: the latent's size does not match its shape";
        return {};
    }
    GgmlCtx owner = new_graph_ctx();
    ggml_context* ctx = owner.get();
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, kGraphSize, false);
    const auto W = [&](const std::string& n) { return wts_.get(n); };
    using Frames = std::vector<ggml_tensor*>;

    // F16 kernels, as the pipeline decodes (fp16 autocast): half the im2col scratch, so even 1024x1024's last level
    // takes im2col + matmul rather than the direct convolution
    auto f16 = [&](ggml_tensor* k) { return ggml_cast(ctx, k, GGML_TYPE_F16); };
    auto conv3 = [&](const Frames& x, const std::string& n) {
        Frames out;
        const int T = static_cast<int>(x.size());
        ggml_tensor* bias = channel(ctx, W(n + ".bias"));
        if (!wts_.has(n + ".weight.t1") || T == 1) {           // 1x1x1, or one frame: only the middle tap sees data
            ggml_tensor* k = f16(W(wts_.has(n + ".weight.t1") ? n + ".weight.t1" : n + ".weight"));
            for (ggml_tensor* f : x) out.push_back(ggml_add(ctx, conv2d(ctx, k, f), bias));
            return out;
        }
        // the three taps as one kernel [3, 3, in, 3 * out]: each input frame's im2col is made once.  Input frame s
        // gives tap k's part to output frame s + 1 - k: out[f] = Y[f][tap 1] + Y[f-1][tap 0] + Y[f+1][tap 2]
        ggml_tensor* k = f16(ggml_concat(ctx, ggml_concat(ctx, W(n + ".weight.t0"), W(n + ".weight.t1"), 3),
                                         W(n + ".weight.t2"), 3));
        const int64_t co = W(n + ".bias")->ne[0];
        Frames y;
        for (ggml_tensor* f : x) y.push_back(conv2d(ctx, k, f));
        auto tap = [&](int f, int t) {
            ggml_tensor* v = y[static_cast<size_t>(f)];
            return ggml_view_3d(ctx, v, v->ne[0], v->ne[1], co, v->nb[1], v->nb[2], static_cast<size_t>(t * co) * v->nb[2]);
        };
        for (int f = 0; f < T; ++f) {
            ggml_tensor* o = ggml_add(ctx, tap(f, 1), bias);
            if (f > 0) o = ggml_add(ctx, o, tap(f - 1, 0));
            if (f + 1 < T) o = ggml_add(ctx, o, tap(f + 1, 2));
            out.push_back(o);
        }
        return out;
    };
    auto gnorm = [&](const Frames& x, const std::string& n, bool silu) {
        const int T = static_cast<int>(x.size());
        ggml_tensor* s = x[0];
        for (int f = 1; f < T; ++f) s = ggml_concat(ctx, s, x[static_cast<size_t>(f)], 1);              // [W, H*T, C]
        s = ggml_add(ctx, ggml_mul(ctx, ggml_group_norm(ctx, s, 32, kVaeEps), channel(ctx, W(n + ".weight"))),
                     channel(ctx, W(n + ".bias")));
        if (silu) s = ggml_silu(ctx, s);
        if (T == 1) return Frames{s};
        Frames out;
        const int64_t fh = x[0]->ne[1];
        for (int f = 0; f < T; ++f)
            out.push_back(ggml_cont(ctx, ggml_view_3d(ctx, s, s->ne[0], fh, s->ne[2], s->nb[1], s->nb[2],
                                                      static_cast<size_t>(f * fh) * s->nb[1])));
        return out;
    };
    auto add = [&](const Frames& a, const Frames& b) {
        Frames o;
        for (size_t f = 0; f < a.size(); ++f) o.push_back(ggml_add(ctx, a[f], b[f]));
        return o;
    };
    auto resnet = [&](const Frames& x, const std::string& p) {
        Frames hh = conv3(gnorm(x, p + "norm1", true), p + "conv1");
        hh = conv3(gnorm(hh, p + "norm2", true), p + "conv2");
        return add(wts_.has(p + "nin_shortcut.weight") ? conv3(x, p + "nin_shortcut") : x, hh);
    };
    // channel j of the result is channel j / rep of x
    auto repeat_channels = [&](ggml_tensor* x, int64_t rep) {
        const int64_t C = x->ne[2], P = x->ne[0] * x->ne[1];
        ggml_tensor* t = ggml_repeat(ctx, ggml_reshape_3d(ctx, x, P, 1, C), ggml_new_tensor_3d(ctx, GGML_TYPE_F32, P, rep, C));
        return ggml_reshape_3d(ctx, t, x->ne[0], x->ne[1], C * rep);
    };
    // "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)" for one frame: the frames r1 = 0, 1 (or 0) of [W, H, r1*4*C]
    auto unshuffle = [&](ggml_tensor* x, int r1n) {
        const int64_t Wd = x->ne[0], Hd = x->ne[1], C = x->ne[2] / (4 * r1n);
        Frames out;
        for (int r1 = 0; r1 < r1n; ++r1) {
            ggml_tensor* rows = nullptr;
            for (int r2 = 0; r2 < 2; ++r2) {
                // channels (r1, r2, r3 = 0..1, c): [W, H, C, 2] -> [2 (r3), W, H, C] -> [2W, H, C]
                const size_t off = static_cast<size_t>((r1 * 4 + r2 * 2) * C) * x->nb[2];
                ggml_tensor* v = ggml_view_4d(ctx, x, Wd, Hd, C, 2, x->nb[1], x->nb[2], static_cast<size_t>(C) * x->nb[2], off);
                v = ggml_reshape_4d(ctx, ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 3, 0)), 2 * Wd, 1, Hd, C);
                rows = rows ? ggml_concat(ctx, rows, v, 1) : v;                                        // [2W, 2 (r2), H, C]
            }
            out.push_back(ggml_reshape_3d(ctx, rows, 2 * Wd, 2 * Hd, C));
        }
        return out;
    };
    auto upsample = [&](const Frames& x, const std::string& p, int64_t cout, bool temporal) {
        const int r1n = temporal ? 2 : 1;
        const Frames hh = conv3(x, p + "conv");
        Frames out;
        for (size_t f = 0; f < x.size(); ++f) {
            const int64_t rep = r1n * 4 * cout / x[f]->ne[2];
            const Frames a = unshuffle(hh[f], r1n), b = unshuffle(repeat_channels(x[f], rep), r1n);
            for (int r = 0; r < r1n; ++r) out.push_back(ggml_add(ctx, a[static_cast<size_t>(r)], b[static_cast<size_t>(r)]));
        }
        return out;
    };
    auto attention = [&](const Frames& x, const std::string& p) {     // one head over every frame's positions (T = 1)
        const int64_t C = x[0]->ne[2], N = x[0]->ne[0] * x[0]->ne[1];
        ggml_tensor* t = gnorm(x, p + "norm", false)[0];
        auto proj = [&](ggml_tensor* v, const std::string& n) {                               // 1x1 conv on [C, N]
            return ggml_add(ctx, ggml_mul_mat(ctx, ggml_reshape_2d(ctx, W(n + ".weight"), C, C), v), W(n + ".bias"));
        };
        ggml_tensor* tc = ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, t, N, C)));           // [C, N]
        ggml_tensor* q = proj(tc, p + "q");
        ggml_tensor* k = proj(tc, p + "k");
        ggml_tensor* vt = ggml_cont(ctx, ggml_transpose(ctx, proj(tc, p + "v")));                       // [N, C]
        const int64_t cs = std::clamp<int64_t>(kAttnChunkBytes / (N * 4), std::min<int64_t>(256, N), N);
        ggml_tensor* o = nullptr;
        for (int64_t c0 = 0; c0 < N; c0 += cs) {
            const int64_t nq = std::min<int64_t>(cs, N - c0);
            ggml_tensor* qc = nq == N ? q : ggml_view_2d(ctx, q, C, nq, q->nb[1], static_cast<size_t>(c0) * q->nb[1]);
            ggml_tensor* kq = ggml_soft_max_ext(ctx, ggml_mul_mat(ctx, k, qc), nullptr, 1.0f / std::sqrt(static_cast<float>(C)), 0.0f);
            ggml_tensor* oc = ggml_mul_mat(ctx, vt, kq);                                                 // [C, nq]
            o = o ? ggml_concat(ctx, o, oc, 1) : oc;
        }
        o = proj(o, p + "proj_out");
        o = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_transpose(ctx, o)), x[0]->ne[0], x[0]->ne[1], C);
        return Frames{ggml_add(ctx, x[0], o)};
    };

    ggml_tensor* zin = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, w, h, zc);
    ggml_set_input(zin);
    const std::string d = "vae.decoder.";
    const int64_t c0 = static_cast<int64_t>(chans[4]);                    // block_out_channels reversed: 1024 first
    Frames x = add(conv3({zin}, d + "conv_in"), {repeat_channels(zin, c0 / zc)});
    x = resnet(x, d + "mid.block_1.");
    x = attention(x, d + "mid.attn_1.");
    x = resnet(x, d + "mid.block_2.");
    const int temporal_levels = static_cast<int>(std::lround(std::log2(wts_.u32("vae_ffactor_temporal", 4))));
    for (int lvl = 0; lvl < 5; ++lvl) {
        const std::string p = d + "up." + std::to_string(lvl) + ".";
        for (int b = 0; wts_.has(p + "block." + std::to_string(b) + ".conv1.bias"); ++b)
            x = resnet(x, p + "block." + std::to_string(b) + ".");
        if (wts_.has(p + "upsample.conv.bias"))
            x = upsample(x, p + "upsample.", static_cast<int64_t>(chans[static_cast<size_t>(3 - lvl)]), lvl < temporal_levels);
    }
    x = conv3(gnorm(x, d + "norm_out", true), d + "conv_out");
    ggml_tensor* out = x.back();                                          // the last frame, as the pipeline keeps it
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);

    Gallocr galloc(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_)));
    if (!ggml_gallocr_alloc_graph(galloc.get(), gf)) { err = "not enough GPU memory for the VAE decoder"; return {}; }
    ggml_backend_tensor_set(zin, z.data(), 0, ggml_nbytes(zin));
    if (ggml_backend_graph_compute(backend_, gf) != GGML_STATUS_SUCCESS) { err = "the VAE graph failed on the GPU"; return {}; }
    const int64_t ow = out->ne[0], oh = out->ne[1];
    std::vector<float> chw(static_cast<size_t>(ggml_nelements(out)));
    ggml_backend_tensor_get(out, chw.data(), 0, ggml_nbytes(out));
    std::vector<float> img(chw.size());                                   // [3, H, W] -> [H, W, 3]
    for (int64_t y = 0; y < oh; ++y)
        for (int64_t xx = 0; xx < ow; ++xx)
            for (int64_t c = 0; c < 3; ++c)
                img[static_cast<size_t>((y * ow + xx) * 3 + c)] = chw[static_cast<size_t>((c * oh + y) * ow + xx)];
    return img;
}

}  // namespace strata_image
