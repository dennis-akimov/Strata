// metal/image/nucleus_dit.cpp - see nucleus_dit.h.  Shapes are ggml's, innermost first: activations are [features,
// tokens], so a weight stored [out, in] (a Linear's) multiplies as ggml_mul_mat(W, x) -> [out, tokens].
#include "nucleus_dit.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace strata_image {
namespace {

constexpr int kGraphSize = 65536;
constexpr float kNormEps = 1e-5f;           // MLX's nn.LayerNorm / nn.RMSNorm default, as the port builds them
constexpr float kOutNormEps = 1e-6f;        // the port's norm_out
constexpr int64_t kAttnChunkBytes = int64_t{512} << 20;

}  // namespace

bool NucleusDiT::load(const std::string& path, ggml_backend_t backend, std::string& err) {
    backend_ = backend;
    if (!wts_.load(path, "nucleus-dit", backend, err)) return false;
    hp_.hidden = static_cast<int>(wts_.u32("hidden", 2048));
    hp_.heads = static_cast<int>(wts_.u32("heads", 16));
    hp_.kv_heads = static_cast<int>(wts_.u32("kv_heads", 4));
    hp_.head_dim = static_cast<int>(wts_.u32("head_dim", 128));
    hp_.layers = static_cast<int>(wts_.u32("layers", 32));
    hp_.experts = static_cast<int>(wts_.u32("experts", 64));
    hp_.moe_dim = static_cast<int>(wts_.u32("moe_dim", 1344));
    hp_.text_dim = static_cast<int>(wts_.u32("text_dim", 4096));
    hp_.in_channels = static_cast<int>(wts_.u32("in_channels", 64));
    hp_.out_channels = static_cast<int>(wts_.u32("out_channels", 16));
    hp_.patch = static_cast<int>(wts_.u32("patch", 2));
    hp_.route_scale = wts_.f32("route_scale", 2.5f);
    hp_.rope_theta = wts_.f32("rope_theta", 10000.0f);
    hp_.capacity = wts_.floats("capacity_factors");
    hp_.axes.clear();
    for (const float a : wts_.floats("axes_dims")) hp_.axes.push_back(static_cast<int>(a));
    if (static_cast<int>(hp_.capacity.size()) != hp_.layers || hp_.axes.size() != 3 ||
        hp_.axes[0] + hp_.axes[1] + hp_.axes[2] != hp_.head_dim || hp_.heads % hp_.kv_heads != 0) {
        err = path + ": inconsistent nucleus-dit settings";
        return false;
    }
    return true;
}

ggml_tensor* NucleusDiT::linear(ggml_context* ctx, ggml_tensor* x, const std::string& name, bool bias) const {
    ggml_tensor* y = ggml_mul_mat(ctx, w(name + ".weight"), x);
    return bias ? ggml_add(ctx, y, w(name + ".bias")) : y;
}

Conditioning NucleusDiT::conditioning(int grid_h, int grid_w, int T, float t) const {
    Conditioning c;
    // timestep_embedding(t * 1000, hidden): [cos(args), sin(args)], float32 as the port computes it
    const int half = hp_.hidden / 2;
    const float tt = t * 1000.0f;
    c.t_emb.resize(static_cast<size_t>(hp_.hidden));
    for (int i = 0; i < half; ++i) {
        const float f = std::exp((-9.210340371976184f * static_cast<float>(i)) / static_cast<float>(half));
        const float arg = tt * f;
        c.t_emb[static_cast<size_t>(i)] = std::cos(arg);
        c.t_emb[static_cast<size_t>(half + i)] = std::sin(arg);
    }
    // RoPE (NucleusMoEEmbedRope): per axis freqs = 1 / theta^(2k / dim); the image's height and width positions are
    // centred ([-ceil(n/2), floor(n/2)) - scale_rope), its frame position 0; the text's positions start at
    // max(grid_h / 2, grid_w / 2), the same on every axis
    std::vector<std::vector<float>> fr(3);
    for (int a = 0; a < 3; ++a)
        for (int k = 0; k < hp_.axes[static_cast<size_t>(a)]; k += 2)
            fr[static_cast<size_t>(a)].push_back(1.0f / std::pow(hp_.rope_theta, static_cast<float>(k) /
                                                                  static_cast<float>(hp_.axes[static_cast<size_t>(a)])));
    const int hd2 = hp_.head_dim / 2;
    const int S = grid_h * grid_w;
    c.img_cos.assign(static_cast<size_t>(S * hd2), 0.0f);
    c.img_sin.assign(static_cast<size_t>(S * hd2), 0.0f);
    const int n_neg_h = grid_h - grid_h / 2, n_neg_w = grid_w - grid_w / 2;
    for (int y = 0; y < grid_h; ++y)
        for (int x = 0; x < grid_w; ++x) {
            const float pos[3] = {0.0f, static_cast<float>(y - n_neg_h), static_cast<float>(x - n_neg_w)};
            float* co = &c.img_cos[static_cast<size_t>((y * grid_w + x) * hd2)];
            float* si = &c.img_sin[static_cast<size_t>((y * grid_w + x) * hd2)];
            int o = 0;
            for (int a = 0; a < 3; ++a)
                for (const float f : fr[static_cast<size_t>(a)]) {
                    const float ang = pos[a] * f;
                    co[o] = std::cos(ang);
                    si[o] = std::sin(ang);
                    ++o;
                }
        }
    const int start = std::max(grid_h / 2, grid_w / 2);
    c.txt_cos.assign(static_cast<size_t>(T * hd2), 0.0f);
    c.txt_sin.assign(static_cast<size_t>(T * hd2), 0.0f);
    for (int i = 0; i < T; ++i) {
        const float p = static_cast<float>(start + i);
        int o = 0;
        for (int a = 0; a < 3; ++a)
            for (const float f : fr[static_cast<size_t>(a)]) {
                c.txt_cos[static_cast<size_t>(i * hd2 + o)] = std::cos(p * f);
                c.txt_sin[static_cast<size_t>(i * hd2 + o)] = std::sin(p * f);
                ++o;
            }
    }
    return c;
}

std::vector<float> NucleusDiT::forward(const std::vector<float>& x, int grid_h, int grid_w, const std::vector<float>& txt,
                                       int T, float t, const Conditioning* given, std::vector<std::vector<float>>* blocks,
                                       std::string& err) {
    auto y = run(x, grid_h, grid_w, {{&txt, T}}, t, given, blocks, -1, err);
    return y.empty() ? std::vector<float>{} : std::move(y[0]);
}

std::vector<std::vector<float>> NucleusDiT::forward_branches(const std::vector<float>& x, int grid_h, int grid_w,
                                                             const std::vector<Text>& texts, float t, std::string& err) {
    return run(x, grid_h, grid_w, texts, t, nullptr, nullptr, -1, err);
}

std::vector<float> NucleusDiT::block(int layer, const std::vector<float>& h, int grid_h, int grid_w,
                                     const std::vector<float>& txt, int T, float t, const Conditioning* given,
                                     std::string& err) {
    if (layer < 0 || layer >= hp_.layers) { err = "block: no such layer"; return {}; }
    auto y = run(h, grid_h, grid_w, {{&txt, T}}, t, given, nullptr, layer, err);
    return y.empty() ? std::vector<float>{} : std::move(y[0]);
}

// only < 0: the whole DiT on the latent tokens `x`; only = l: block l alone on the hidden state `x` [S, hidden].
// Every branch (a prompt: the CFG's two) sees the same `x`; they run side by side in one graph, and their routed experts
// in one batched mul_mat, so each expert's weights are read once a step for both.
std::vector<std::vector<float>> NucleusDiT::run(const std::vector<float>& x, int grid_h, int grid_w,
                                                const std::vector<Text>& texts, float t, const Conditioning* given,
                                                std::vector<std::vector<float>>* blocks, int only, std::string& err) {
    const int S = grid_h * grid_w, D = hp_.hidden, E = hp_.experts, hd = hp_.head_dim;
    const int nh = hp_.heads, nkv = hp_.kv_heads, hd2 = hd / 2;
    const int B = static_cast<int>(texts.size());
    const int x_dim = only < 0 ? hp_.in_channels : hp_.hidden;
    bool ok = B >= 1 && S >= 1 && static_cast<int64_t>(x.size()) == static_cast<int64_t>(S) * x_dim && (!given || B == 1);
    for (const Text& tx : texts)
        ok = ok && tx.T >= 1 && static_cast<int64_t>(tx.h->size()) == static_cast<int64_t>(tx.T) * hp_.text_dim;
    if (!ok) {
        err = "forward: input sizes do not match the grid and the text length";
        return {};
    }
    std::vector<Conditioning> conds;
    for (const Text& tx : texts) conds.push_back(given ? *given : conditioning(grid_h, grid_w, tx.T, t));

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * kGraphSize + ggml_graph_overhead_custom(kGraphSize, false);
    ip.no_alloc = true;
    GgmlCtx owner(ggml_init(ip));
    ggml_context* ctx = owner.get();
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, kGraphSize, false);

    auto input = [&](ggml_tensor* tsr) { ggml_set_input(tsr); return tsr; };
    ggml_tensor* x_in = input(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, x_dim, S));
    ggml_tensor* temb = input(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, 1));
    ggml_tensor* icos = input(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd2, S));
    ggml_tensor* isin = input(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd2, S));
    ggml_tensor* eye = input(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, S, S));   // its rows: the one-hot of a token
    std::vector<ggml_tensor*> txt_in(B), tcos(B), tsin(B);
    for (int b = 0; b < B; ++b) {
        txt_in[b] = input(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp_.text_dim, texts[b].T));
        tcos[b] = input(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd2, texts[b].T));
        tsin[b] = input(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd2, texts[b].T));
    }

    auto rms = [&](ggml_tensor* v, const std::string& wn) { return ggml_mul(ctx, ggml_rms_norm(ctx, v, kNormEps), w(wn)); };
    // apply_rotary_emb_complex: (x[2i], x[2i+1]) as a complex number times (cos + i sin)
    auto rope = [&](ggml_tensor* v, int heads, int n, ggml_tensor* co, ggml_tensor* si) {
        ggml_tensor* v4 = ggml_reshape_4d(ctx, v, 2, hd2, heads, n);
        auto plane = [&](int k) {
            ggml_tensor* p = ggml_view_4d(ctx, v4, 1, hd2, heads, n, v4->nb[1], v4->nb[2], v4->nb[3],
                                          static_cast<size_t>(k) * v4->nb[0]);
            return ggml_reshape_3d(ctx, ggml_cont(ctx, p), hd2, heads, n);
        };
        ggml_tensor* re = plane(0);
        ggml_tensor* im = plane(1);
        ggml_tensor* c3 = ggml_reshape_3d(ctx, co, hd2, 1, n);
        ggml_tensor* s3 = ggml_reshape_3d(ctx, si, hd2, 1, n);
        ggml_tensor* o_re = ggml_sub(ctx, ggml_mul(ctx, re, c3), ggml_mul(ctx, im, s3));
        ggml_tensor* o_im = ggml_add(ctx, ggml_mul(ctx, re, s3), ggml_mul(ctx, im, c3));
        ggml_tensor* both = ggml_concat(ctx, ggml_reshape_4d(ctx, o_re, 1, hd2, heads, n),
                                        ggml_reshape_4d(ctx, o_im, 1, hd2, heads, n), 0);
        return ggml_reshape_3d(ctx, both, hd, heads, n);
    };
    ggml_tensor* h0 = only < 0 ? linear(ctx, x_in, "img_in", true) : x_in;                 // [D, S]
    std::vector<ggml_tensor*> h(B, h0), txtn(B);
    ggml_tensor* c = linear(ctx, ggml_silu(ctx, linear(ctx, temb, "time_text_embed.timestep_embedder.linear_1", true)),
                            "time_text_embed.timestep_embedder.linear_2", true);
    c = rms(c, "time_text_embed.norm.weight");                                             // [D, 1]
    for (int b = 0; b < B; ++b) txtn[b] = rms(txt_in[b], "txt_norm.weight");                // [text_dim, T]
    ggml_tensor* c_act = ggml_silu(ctx, c);
    ggml_tensor* c_rep = nullptr;                                                          // c for every token

    std::vector<ggml_tensor*> outs;
    for (int l = only < 0 ? 0 : only; l < (only < 0 ? hp_.layers : only + 1); ++l) {
        const std::string p = "transformer_blocks." + std::to_string(l) + ".";
        ggml_tensor* mod = linear(ctx, c_act, p + "img_mod.1", true);                       // [4D, 1]
        auto part = [&](int k) { return ggml_cont(ctx, ggml_view_2d(ctx, mod, D, 1, mod->nb[1],
                                                                     static_cast<size_t>(k) * D * sizeof(float))); };
        ggml_tensor* scale1 = ggml_scale_bias(ctx, part(0), 1.0f, 1.0f);
        ggml_tensor* gate1 = ggml_tanh(ctx, ggml_clamp(ctx, part(1), -2.0f, 2.0f));
        ggml_tensor* scale2 = ggml_scale_bias(ctx, part(2), 1.0f, 1.0f);
        ggml_tensor* gate2 = ggml_tanh(ctx, ggml_clamp(ctx, part(3), -2.0f, 2.0f));
        const float cf = hp_.capacity[static_cast<size_t>(l)];
        const int C = std::min(S, std::max(1, static_cast<int>(std::ceil(cf * static_cast<float>(S) / static_cast<float>(E)))));
        const int N = C * E;

        std::vector<ggml_tensor*> m2(B), xe(B), idx(B), norm(B), onehot_t(B);
        for (int b = 0; b < B; ++b) {
            const int T = texts[b].T;
            // attention: image queries over image + text keys/values, 16 query heads on 4 key/value heads
            ggml_tensor* m1 = ggml_mul(ctx, ggml_norm(ctx, h[b], kNormEps), scale1);
            ggml_tensor* q = rms(ggml_reshape_3d(ctx, linear(ctx, m1, p + "attn.to_q", false), hd, nh, S), p + "attn.norm_q.weight");
            ggml_tensor* k = rms(ggml_reshape_3d(ctx, linear(ctx, m1, p + "attn.to_k", false), hd, nkv, S), p + "attn.norm_k.weight");
            ggml_tensor* v = ggml_reshape_3d(ctx, linear(ctx, m1, p + "attn.to_v", false), hd, nkv, S);
            ggml_tensor* tp = linear(ctx, txtn[b], p + "encoder_proj", true);                  // [D, T]
            ggml_tensor* tk = rms(ggml_reshape_3d(ctx, linear(ctx, tp, p + "attn.add_k_proj", false), hd, nkv, T),
                                  p + "attn.norm_added_k.weight");
            ggml_tensor* tv = ggml_reshape_3d(ctx, linear(ctx, tp, p + "attn.add_v_proj", false), hd, nkv, T);
            q = rope(q, nh, S, icos, isin);
            k = rope(k, nkv, S, icos, isin);
            tk = rope(tk, nkv, T, tcos[b], tsin[b]);
            ggml_tensor* K = ggml_concat(ctx, k, tk, 2);                                      // [hd, nkv, S+T]
            ggml_tensor* V = ggml_concat(ctx, v, tv, 2);
            // attention in F32 (flash attention, F16 or F32 K/V, moved the 3-prompt LPIPS from 0.079 to 0.104 / 0.10
            // for no measurable speed: the routing amplifies small differences)
            ggml_tensor* qp = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));               // [hd, S, nh]
            ggml_tensor* kp = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));               // [hd, L, nkv]
            ggml_tensor* vt = ggml_cont(ctx, ggml_permute(ctx, V, 1, 2, 0, 3));               // [L, hd, nkv]
            // the queries in chunks of at most kAttnChunkBytes of scores (17 GB in one piece at 2048x2048); each row
            // of scores is its query's own, so the chunks change nothing
            const int64_t L = S + T;
            const int64_t cs = std::clamp<int64_t>(kAttnChunkBytes / (L * nh * 4), 256, S);
            ggml_tensor* o = nullptr;
            for (int64_t c0 = 0; c0 < S; c0 += cs) {
                const int64_t n = std::min<int64_t>(cs, S - c0);
                ggml_tensor* qc = n == S ? qp : ggml_cont(ctx, ggml_view_3d(ctx, qp, hd, n, nh, qp->nb[1], qp->nb[2],
                                                                            static_cast<size_t>(c0) * qp->nb[1]));
                ggml_tensor* kq = ggml_mul_mat(ctx, kp, qc);                                  // [L, n, nh]
                kq = ggml_soft_max_ext(ctx, kq, nullptr, 1.0f / std::sqrt(static_cast<float>(hd)), 0.0f);
                ggml_tensor* oc = ggml_mul_mat(ctx, vt, kq);                                  // [hd, n, nh]
                o = o ? ggml_concat(ctx, o, oc, 1) : oc;
            }
            o = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3)), D, S);
            o = linear(ctx, o, p + "attn.to_out.0", false);
            h[b] = ggml_add(ctx, h[b], ggml_mul(ctx, o, gate1));

            ggml_tensor* n2 = ggml_norm(ctx, h[b], kNormEps);
            m2[b] = ggml_mul(ctx, n2, scale2);
            if (cf <= 0.0f) continue;
            // expert-choice: each expert scores every token (router on [timestep, unmodulated token]) and takes its
            // top C; a token taken by several experts gets weights that sum to route_scale
            if (!c_rep) c_rep = ggml_repeat(ctx, c, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, S));
            ggml_tensor* rin = ggml_concat(ctx, c_rep, n2, 0);
            ggml_tensor* probs = ggml_soft_max(ctx, linear(ctx, rin, p + "img_mlp.gate", false));   // [E, S]
            ggml_tensor* aff = ggml_cont(ctx, ggml_transpose(ctx, probs));                         // [S, E]
            ggml_tensor* order = ggml_argsort(ctx, aff, GGML_SORT_ORDER_DESC);                      // [S, E]
            idx[b] = ggml_reshape_1d(ctx, ggml_cont(ctx, ggml_view_2d(ctx, order, C, E, order->nb[1], 0)), N);
            ggml_tensor* onehot = ggml_get_rows(ctx, eye, idx[b]);                                  // [S, N]
            ggml_tensor* score = ggml_reshape_2d(ctx, ggml_sum_rows(ctx, ggml_mul(ctx,
                ggml_reshape_3d(ctx, onehot, S, C, E), ggml_reshape_3d(ctx, aff, S, 1, E))), 1, N);  // [1, N]
            onehot_t[b] = ggml_cont(ctx, ggml_transpose(ctx, onehot));                              // [N, S]
            ggml_tensor* sums = ggml_mul_mat(ctx, onehot_t[b], ggml_reshape_2d(ctx, score, N, 1));   // [S, 1]
            ggml_tensor* sums_at = ggml_get_rows(ctx, ggml_reshape_2d(ctx, sums, 1, S), idx[b]);     // [1, N]
            norm[b] = ggml_scale(ctx, ggml_div(ctx, score, ggml_scale_bias(ctx, sums_at, 1.0f, 1e-12f)), hp_.route_scale);
            xe[b] = ggml_reshape_3d(ctx, ggml_get_rows(ctx, m2[b], idx[b]), D, C, E);              // [D, C, E]
        }

        // the routed experts of every branch at once: [D, B*C, E]
        std::vector<ggml_tensor*> ye(B);
        if (cf > 0.0f) {
            ggml_tensor* xa = xe[0];
            for (int b = 1; b < B; ++b) xa = ggml_concat(ctx, xa, xe[b], 1);
            ggml_tensor* gu = ggml_mul_mat(ctx, w(p + "img_mlp.experts.gate_up_proj"), xa);         // [2H, B*C, E]
            ggml_tensor* ya = ggml_mul_mat(ctx, w(p + "img_mlp.experts.down_proj"), ggml_swiglu(ctx, gu));   // silu(gate) * up
            for (int b = 0; b < B; ++b)
                ye[b] = B == 1 ? ya : ggml_cont(ctx, ggml_view_3d(ctx, ya, D, C, E, ya->nb[1], ya->nb[2],
                                                                  static_cast<size_t>(b) * C * ya->nb[1]));
        }
        for (int b = 0; b < B; ++b) {
            ggml_tensor* ff = nullptr;
            // the dense FFN and the shared expert: hidden * silu(gate), hidden the first half (ggml_swiglu_swapped)
            if (cf <= 0.0f) {
                ff = linear(ctx, ggml_swiglu_swapped(ctx, linear(ctx, m2[b], p + "img_mlp.net.0.proj", false)),
                            p + "img_mlp.net.2", false);
            } else {
                ggml_tensor* yw = ggml_mul(ctx, ggml_reshape_2d(ctx, ye[b], D, N), norm[b]);         // [D, N] weighted
                ggml_tensor* routed = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, yw)), onehot_t[b]);   // [D, S]
                ggml_tensor* shared = linear(ctx, ggml_swiglu_swapped(ctx, linear(ctx, m2[b],
                    p + "img_mlp.shared_expert.net.0.proj", false)), p + "img_mlp.shared_expert.net.2", false);
                ff = ggml_add(ctx, shared, routed);
            }
            h[b] = ggml_add(ctx, h[b], ggml_mul(ctx, ff, gate2));
            if (blocks && b == 0) {
                ggml_set_output(h[b]);
                outs.push_back(h[b]);
            }
            ggml_build_forward_expand(gf, h[b]);
        }
    }
    std::vector<ggml_tensor*> out(h);
    if (only < 0) {
        ggml_tensor* mod = linear(ctx, c_act, "norm_out.linear", true);                        // [2D, 1]: scale, shift
        ggml_tensor* scale = ggml_scale_bias(ctx, ggml_cont(ctx, ggml_view_2d(ctx, mod, D, 1, mod->nb[1], 0)), 1.0f, 1.0f);
        ggml_tensor* shift = ggml_cont(ctx, ggml_view_2d(ctx, mod, D, 1, mod->nb[1], static_cast<size_t>(D) * sizeof(float)));
        for (int b = 0; b < B; ++b)
            out[b] = linear(ctx, ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, h[b], kOutNormEps), scale), shift),
                            "proj_out", false);                                                // [in_channels, S]
    }
    for (ggml_tensor* o : out) {
        ggml_set_output(o);
        ggml_build_forward_expand(gf, o);
    }

    Gallocr galloc(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_)));
    if (!ggml_gallocr_alloc_graph(galloc.get(), gf)) { err = "not enough GPU memory for the DiT's activations"; return {}; }
    compute_bytes_ = ggml_gallocr_get_buffer_size(galloc.get(), 0);
    // an input the graph does not use (the identity in a dense block alone) has no buffer: nothing to set
    auto set = [](ggml_tensor* tsr, const void* data) { if (tsr->buffer) ggml_backend_tensor_set(tsr, data, 0, ggml_nbytes(tsr)); };
    set(x_in, x.data());
    set(temb, conds[0].t_emb.data());
    set(icos, conds[0].img_cos.data());
    set(isin, conds[0].img_sin.data());
    for (int b = 0; b < B; ++b) {
        set(txt_in[b], texts[b].h->data());
        set(tcos[b], conds[b].txt_cos.data());
        set(tsin[b], conds[b].txt_sin.data());
    }
    if (eye->buffer) {
        std::vector<float> id(static_cast<size_t>(S) * static_cast<size_t>(S), 0.0f);
        for (int i = 0; i < S; ++i) id[static_cast<size_t>(i) * static_cast<size_t>(S) + static_cast<size_t>(i)] = 1.0f;
        set(eye, id.data());
    }
    if (ggml_backend_graph_compute(backend_, gf) != GGML_STATUS_SUCCESS) { err = "the DiT graph failed on the GPU"; return {}; }
    if (blocks) {
        blocks->clear();
        for (ggml_tensor* bt : outs) {
            std::vector<float> v(static_cast<size_t>(ggml_nelements(bt)));
            ggml_backend_tensor_get(bt, v.data(), 0, ggml_nbytes(bt));
            blocks->push_back(std::move(v));
        }
    }
    std::vector<std::vector<float>> ys;
    for (ggml_tensor* o : out) {
        std::vector<float> y(static_cast<size_t>(ggml_nelements(o)));
        ggml_backend_tensor_get(o, y.data(), 0, ggml_nbytes(o));
        ys.push_back(std::move(y));
    }
    return ys;
}

}  // namespace strata_image
