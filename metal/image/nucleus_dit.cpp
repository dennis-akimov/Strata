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
constexpr int64_t kKeyBlock = 64;            // OP_FLASH_ATTN_EXT_TENSOR_NCPSG: the tensor-API kernel's keys a block

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
    if (wts_.str("nucleus-dit.rope_layout") != "neox") {
        err = path + " is from an older tools/image/nucleus_gguf.py (no rope_layout): convert the DiT again";
        return false;
    }
    // ggml_rope_multi's frequency of pair i is theta^(-2i / head_dim); the port's restarts on every axis
    // (theta^(-2k / axis_dim), k counting within the axis): the factor that turns the one into the other
    rope_ff_.clear();
    for (int a = 0, i = 0; a < 3; ++a)
        for (int k = 0; k < hp_.axes[static_cast<size_t>(a)]; k += 2, ++i)
            rope_ff_.push_back(std::pow(hp_.rope_theta, -2.0f * static_cast<float>(i) / static_cast<float>(hp_.head_dim)) /
                               std::pow(hp_.rope_theta, -static_cast<float>(k) / static_cast<float>(hp_.axes[static_cast<size_t>(a)])));
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
    // RoPE positions (NucleusMoEEmbedRope), [frame, height, width, unused] each a run of tokens, as ggml_rope_multi
    // reads them: the image's height and width centred ([-ceil(n/2), floor(n/2))), its frame 0; the text's start at
    // max(grid_h / 2, grid_w / 2), the same on every axis
    const int S = grid_h * grid_w;
    c.img_pos.assign(static_cast<size_t>(4 * S), 0);
    const int n_neg_h = grid_h - grid_h / 2, n_neg_w = grid_w - grid_w / 2;
    for (int y = 0; y < grid_h; ++y)
        for (int x = 0; x < grid_w; ++x) {
            c.img_pos[static_cast<size_t>(S + y * grid_w + x)] = y - n_neg_h;
            c.img_pos[static_cast<size_t>(2 * S + y * grid_w + x)] = x - n_neg_w;
        }
    const int start = std::max(grid_h / 2, grid_w / 2);
    c.txt_pos.assign(static_cast<size_t>(4 * T), 0);
    for (int i = 0; i < T; ++i)
        for (int a = 0; a < 3; ++a) c.txt_pos[static_cast<size_t>(a * T + i)] = start + i;
    return c;
}

std::vector<float> NucleusDiT::forward(const std::vector<float>& x, int grid_h, int grid_w, const std::vector<float>& txt,
                                       int T, float t, std::vector<std::vector<float>>* blocks, std::string& err) {
    auto y = run(x, grid_h, grid_w, {{&txt, T}}, t, blocks, -1, err);
    return y.empty() ? std::vector<float>{} : std::move(y[0]);
}

std::vector<std::vector<float>> NucleusDiT::forward_branches(const std::vector<float>& x, int grid_h, int grid_w,
                                                             const std::vector<Text>& texts, float t, std::string& err) {
    return run(x, grid_h, grid_w, texts, t, nullptr, -1, err);
}

std::vector<float> NucleusDiT::block(int layer, const std::vector<float>& h, int grid_h, int grid_w,
                                     const std::vector<float>& txt, int T, float t, std::string& err) {
    if (layer < 0 || layer >= hp_.layers) { err = "block: no such layer"; return {}; }
    auto y = run(h, grid_h, grid_w, {{&txt, T}}, t, nullptr, layer, err);
    return y.empty() ? std::vector<float>{} : std::move(y[0]);
}

// only < 0: the whole DiT on the latent tokens `x`; only = l: block l alone on the hidden state `x` [S, hidden].
// Every branch (a prompt: the CFG's two) sees the same `x`; they run side by side in one graph, and their routed experts
// in one batched mul_mat, so each expert's weights are read once a step for both.
std::vector<std::vector<float>> NucleusDiT::run(const std::vector<float>& x, int grid_h, int grid_w,
                                                const std::vector<Text>& texts, float t,
                                                std::vector<std::vector<float>>* blocks, int only, std::string& err) {
    const int S = grid_h * grid_w, D = hp_.hidden, E = hp_.experts, hd = hp_.head_dim;
    const int nh = hp_.heads, nkv = hp_.kv_heads, hd2 = hd / 2;
    int sections[GGML_MROPE_SECTIONS] = {hp_.axes[0] / 2, hp_.axes[1] / 2, hp_.axes[2] / 2, 0};
    const int B = static_cast<int>(texts.size());
    const int x_dim = only < 0 ? hp_.in_channels : hp_.hidden;
    bool ok = B >= 1 && S >= 1 && static_cast<int64_t>(x.size()) == static_cast<int64_t>(S) * x_dim;
    for (const Text& tx : texts)
        ok = ok && tx.T >= 1 && static_cast<int64_t>(tx.h->size()) == static_cast<int64_t>(tx.T) * hp_.text_dim;
    if (!ok) {
        err = "forward: input sizes do not match the grid and the text length";
        return {};
    }
    std::vector<Conditioning> conds;
    for (const Text& tx : texts) conds.push_back(conditioning(grid_h, grid_w, tx.T, t));

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * kGraphSize + ggml_graph_overhead_custom(kGraphSize, false);
    ip.no_alloc = true;
    GgmlCtx owner(ggml_init(ip));
    ggml_context* ctx = owner.get();
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, kGraphSize, false);

    auto input = [&](ggml_tensor* tsr) { ggml_set_input(tsr); return tsr; };
    ggml_tensor* x_in = input(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, x_dim, S));
    ggml_tensor* temb = input(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, 1));
    ggml_tensor* ipos = input(ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * S));
    ggml_tensor* ff = input(ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd2));
    ggml_tensor* eye = input(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, S, S));   // its rows: the one-hot of a token
    // flash attention's keys padded to a multiple of kKeyBlock (ggml's tensor-API kernel needs it): a row of 0 for the
    // S + T keys and -inf for the padding, repeated for every query on the GPU
    std::vector<ggml_tensor*> txt_in(B), tpos(B), krow(B), kmask(B);
    for (int b = 0; b < B; ++b) {
        const int64_t Lp = (S + texts[b].T + kKeyBlock - 1) / kKeyBlock * kKeyBlock;
        txt_in[b] = input(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp_.text_dim, texts[b].T));
        tpos[b] = input(ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * texts[b].T));
        krow[b] = input(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, Lp, 1));
        kmask[b] = ggml_cast(ctx, ggml_repeat(ctx, krow[b], ggml_new_tensor_2d(ctx, GGML_TYPE_F32, Lp, S)), GGML_TYPE_F16);
    }

    auto rms = [&](ggml_tensor* v, const std::string& wn) { return ggml_mul(ctx, ggml_rms_norm(ctx, v, kNormEps), w(wn)); };
    // RoPE: the pairs (i, i + head_dim/2) - the converter stored the dims so (rope_layout "neox") - with the frame,
    // height and width positions per section of pairs and the port's per-axis frequencies (rope_ff_)
    auto rope = [&](ggml_tensor* v, ggml_tensor* pos) {
        return ggml_rope_multi(ctx, v, pos, ff, hd, sections, GGML_ROPE_TYPE_MROPE, 0, hp_.rope_theta, 1.0f, 0.0f, 1.0f,
                               0.0f, 0.0f);
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
            q = rope(q, ipos);
            k = rope(k, ipos);
            tk = rope(tk, tpos[b]);
            ggml_tensor* K = ggml_concat(ctx, k, tk, 2);                                      // [hd, nkv, S+T]
            ggml_tensor* V = ggml_concat(ctx, v, tv, 2);
            // flash attention on ggml's tensor-API kernel: K and V as F16, padded to kKeyBlock keys (masked), the
            // softmax and the sums in F32; no [L, S, heads] score matrix (1 GB at 1024x1024)
            const int64_t pad = kmask[b]->ne[0] - (S + T);
            auto kv = [&](ggml_tensor* x4) {                                                  // [hd, nkv, L] -> [hd, Lp, nkv]
                return ggml_cast(ctx, ggml_pad(ctx, ggml_cont(ctx, ggml_permute(ctx, x4, 0, 2, 1, 3)), 0,
                                               static_cast<int>(pad), 0, 0), GGML_TYPE_F16);
            };
            ggml_tensor* o = ggml_flash_attn_ext(ctx, ggml_permute(ctx, q, 0, 2, 1, 3), kv(K), kv(V), kmask[b],
                                    1.0f / std::sqrt(static_cast<float>(hd)), 0.0f, 0.0f);
            ggml_prec_set_acc(o, GGML_PREC_F32);
            o = ggml_reshape_2d(ctx, o, D, S);                                                // [hd, nh, S]: tokens' rows
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
            // each slot's affinity, gathered per expert: aff as [1, S, E] rows, the slots as [C, E] indices
            ggml_tensor* score = ggml_reshape_2d(ctx, ggml_get_rows(ctx, ggml_reshape_3d(ctx, aff, 1, S, E),
                                                                    ggml_reshape_2d(ctx, idx[b], C, E)), 1, N);  // [1, N]
            onehot_t[b] = ggml_cont(ctx, ggml_transpose(ctx, onehot));                              // [N, S]
            ggml_tensor* sums = ggml_mul_mat(ctx, onehot_t[b], ggml_reshape_2d(ctx, score, N, 1));   // [S, 1]
            ggml_tensor* sums_at = ggml_get_rows(ctx, ggml_reshape_2d(ctx, sums, 1, S), idx[b]);     // [1, N]
            norm[b] = ggml_scale(ctx, ggml_div(ctx, score, ggml_scale_bias(ctx, sums_at, 1.0f, 1e-12f)), hp_.route_scale);
            xe[b] = ggml_reshape_3d(ctx, ggml_get_rows(ctx, m2[b], idx[b]), D, C, E);              // [D, C, E]
        }

        // the routed experts of every branch at once, [D, C, E, B]: the weights broadcast over the branches, and each
        // branch's results are one contiguous slab
        std::vector<ggml_tensor*> ye(B);
        if (cf > 0.0f) {
            ggml_tensor* xa = ggml_reshape_4d(ctx, xe[0], D, C, E, 1);
            for (int b = 1; b < B; ++b) xa = ggml_concat(ctx, xa, ggml_reshape_4d(ctx, xe[b], D, C, E, 1), 3);
            ggml_tensor* gu = ggml_mul_mat(ctx, w(p + "img_mlp.experts.gate_up_proj"), xa);         // [2H, C, E, B]
            ggml_tensor* ya = ggml_mul_mat(ctx, w(p + "img_mlp.experts.down_proj"), ggml_swiglu(ctx, gu));   // silu(gate) * up
            for (int b = 0; b < B; ++b)
                ye[b] = ggml_view_3d(ctx, ya, D, C, E, ya->nb[1], ya->nb[2], static_cast<size_t>(b) * ya->nb[3]);
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
    set(ipos, conds[0].img_pos.data());
    set(ff, rope_ff_.data());
    for (int b = 0; b < B; ++b) {
        std::vector<float> row(static_cast<size_t>(krow[b]->ne[0]), -std::numeric_limits<float>::infinity());
        std::fill_n(row.begin(), S + texts[b].T, 0.0f);
        set(krow[b], row.data());
        set(txt_in[b], texts[b].h->data());
        set(tpos[b], conds[b].txt_pos.data());
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
