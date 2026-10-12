"""The parity check of strata-image's HunyuanImage-3.0 against Tencent's own modules (float32, CPU), part by part: the
80B model does not fit whole, so each part runs alone on the reference's input.

    python tools/image/hunyuan_ref.py dump HY3_CODE GGUF REF_DIR OUT_DIR
    build-metal/metal/strata-image --hunyuan GGUF --check-hunyuan OUT_DIR
    python tools/image/hunyuan_ref.py compare OUT_DIR

HY3_CODE: the model repo's Python files (modeling_hunyuan_image_3.py, ...).  REF_DIR: hunyuan_gguf.py --ref-subset's
.npy files (layers 0 and 31 without their experts, the embeddings, the image head, the VAE decoder; BF16 as uint16).
The experts of layers 0 and 31 come from the GGUF, dequantized, so the check sees the code and not the 4-bit rounding.  Needs torch, transformers, diffusers, einops (not Strata's .venv).
"""
import json
import pathlib
import sys
from types import SimpleNamespace

import numpy as np

HERE = pathlib.Path(__file__).resolve()
PROMPT, SIZE, SEED = "draw sweet cat", (1024, 1024), 7


def compare(out):
    out = pathlib.Path(out)
    rows = [("x0", "x0_cpp", "the input tokens (embeddings + image head)"), ("out0", "out0_cpp", "layer 0 alone"),
            ("out31", "out31_cpp", "layer 31 alone"), ("head", "head_cpp", "the image head alone (velocity)"),
            ("out0", "out0_chain_cpp", "layer 0 in the whole step"), ("img", "img_cpp", "the VAE alone")]
    P = int((out / "case.txt").read_text().split()[0])
    bad = 0
    for ref, cpp, what in rows:
        if not (out / f"{cpp}.npy").exists():
            print(f"{what}: no {cpp}.npy")
            bad += 1
            continue
        a, b = np.load(out / f"{ref}.npy"), np.load(out / f"{cpp}.npy")
        if ref in ("x0", "out0", "out31"):
            a = a[P:]                                                  # the step's tokens (the C++ returns those)
        a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
        rel = np.linalg.norm(a - b) / np.linalg.norm(a)
        cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
        print(f"{what:45s} rel {rel:.2e}  cos {cos:.6f}  max_abs {np.abs(a - b).max():.3e}  nonfinite {np.isnan(b).sum()}")
        bad += not (rel < 2e-2 and np.isfinite(b).all())
    return bad


def dump(code, gguf_path, ref_dir, out):
    import torch
    sys.path.insert(0, str(pathlib.Path(code).parent))
    pkg = pathlib.Path(code).name
    import contextlib
    torch.cuda.set_device = lambda *a, **k: None                       # the MoE calls these on every forward (CUDA)
    torch.cuda.nvtx.range = lambda *a, **k: contextlib.nullcontext()
    cfgmod = __import__(f"{pkg}.configuration_hunyuan_image_3", fromlist=["x"])
    model = __import__(f"{pkg}.modeling_hunyuan_image_3", fromlist=["x"])
    procmod = __import__(f"{pkg}.image_processor", fromlist=["x"])
    vaemod = __import__(f"{pkg}.autoencoder_kl_3d", fromlist=["x"])
    sys.path.insert(0, str(HERE.parent))
    sys.path.insert(0, str(HERE.parents[1]))
    from hunyuan_tokens import build
    from strata_tokenizer import Tokenizer
    from hunyuan_gguf import gguf                                      # Strata's llama.cpp gguf-py

    torch.manual_seed(0)
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ref = pathlib.Path(ref_dir)
    cfg = cfgmod.HunyuanImage3Config.from_json_file(str(pathlib.Path(code) / "config.json"))
    cfg._attn_implementation = "sdpa"

    def weight(name):
        a = np.load(ref / f"{name}.npy")
        if a.dtype == np.uint16:
            a = (a.astype(np.uint32) << 16).view(np.float32)
        return torch.from_numpy(np.ascontiguousarray(a))

    def load(module, prefix, extra=None):
        sd = {}
        for k in module.state_dict():
            sd[k] = extra[k] if extra and k in extra else weight(prefix + k)
        module.load_state_dict(sd, strict=True)
        return module.float().eval()

    # the tokens and the 2D RoPE as the model builds them (its own functions)
    tok = Tokenizer.from_hf_json(str(pathlib.Path(code) / "tokenizer.json"), "hunyuan")
    b = build(tok, PROMPT, *SIZE)
    ids, L, (h, w) = b["ids"], b["image_start"], b["grid"]
    S, P, N = h * w, L - 3, L + h * w
    cos, sin, allpos = model.build_batch_2d_rope(N, 128, image_infos=[[(slice(L, L + S), (h, w))]], return_all_pos=True)
    assert allpos[0][:, 0].tolist() == b["pos"], "hunyuan_tokens.py's positions differ from build_2d_rope's"
    mask = torch.tril(torch.ones(N, N, dtype=torch.bool))
    mask[L:, L:] = True                                                # the image sees the whole image
    mask = mask[None, None]

    # the step's inputs: the noise, the first step's timesteps, the guidance as bfloat16
    gen = torch.Generator().manual_seed(SEED)
    z = torch.randn(1, 32, h, w, generator=gen)
    s1 = 3 * 0.875 / (1 + 2 * 0.875)
    t, r = torch.tensor([1000.0]), torch.tensor([np.float32(s1) * np.float32(1000.0)])
    g = torch.tensor([2500.0], dtype=torch.bfloat16).float()

    emb = {n: load(model.TimestepEmbedder(hidden_size=4096), n + ".")
           for n in ("time_embed", "time_embed_2", "timestep_emb", "guidance_emb", "timestep_r_emb")}
    down = load(model.UNetDown(patch_size=1, emb_channels=4096, in_channels=32, hidden_channels=1024, out_channels=4096),
                "patch_embed.")
    up = load(model.UNetUp(patch_size=1, emb_channels=4096, in_channels=4096, hidden_channels=1024, out_channels=32,
                           out_norm=True), "final_layer.")
    wte = weight("model.wte.weight")
    with torch.no_grad():
        x = wte[torch.tensor(ids)][None].clone()                       # [1, N, D]
        img, _, _ = down(z, emb["time_embed"](t))
        x[0, L:L + S] = img[0]
        x[0, P] = emb["timestep_emb"](t)[0]
        x[0, P + 1] = emb["guidance_emb"](g)[0]
        x[0, P + 2] = emb["timestep_r_emb"](r)[0]
    np.save(out / "x0.npy", x[0].numpy())

    reader = gguf.GGUFReader(gguf_path)
    tensors = {tt.name: tt for tt in reader.tensors}

    def layer(l, inp):
        p = f"model.layers.{l}."
        m = model.HunyuanImage3DecoderLayer(cfg, l)
        extra = {}
        for proj in ("gate_and_up_proj", "down_proj"):
            tt = tensors[f"{p}mlp.experts.{proj}"]
            full = gguf.quants.dequantize(tt.data, tt.tensor_type)                # [64, out, in]
            for e in range(full.shape[0]):
                extra[f"mlp.experts.{e}.{proj}.weight"] = torch.from_numpy(np.ascontiguousarray(full[e]))
        m = load(m, p, extra)
        m.mlp.moe_impl = "eager"
        with torch.no_grad():
            return m(inp, attention_mask=mask, custom_pos_emb=(cos, sin))[0]

    y0 = layer(0, x)
    np.save(out / "out0.npy", y0[0].numpy())
    y31 = layer(31, y0)                                                # layer 31 alone, on a realistic input
    np.save(out / "out31.npy", y31[0].numpy())
    with torch.no_grad():
        v = up(y31[:, L:L + S], emb["time_embed_2"](t), h, w)          # UNetUp on the image tokens (no ln_f)
    np.save(out / "head.npy", v[0].numpy())
    np.save(out / "z.npy", z[0].numpy())

    empty = torch.empty                                                # its __init__ makes torch.empty(0, device="cuda")
    torch.empty = lambda *a, device=None, **k: empty(*a, **k)
    vae = vaemod.AutoencoderKLConv3D.from_config(cfg.vae)
    torch.empty = empty
    dec = {k: weight("vae.decoder." + k) for k in vae.decoder.state_dict()}
    vae.decoder.load_state_dict(dec, strict=True)
    vae = vae.float().eval()
    zv = torch.randn(1, 32, 1, 16, 16, generator=gen)
    with torch.no_grad():
        im = vae.decode(zv, return_dict=False)[0]                      # [1, 3, 1, 256, 256], the last frame
    np.save(out / "zv.npy", zv[0, :, 0].numpy())
    np.save(out / "img.npy", im[0, :, 0].permute(1, 2, 0).numpy())
    (out / "case.txt").write_text(f"{P} {h} {w} {float(t[0])} {float(r[0])} {float(g[0])}\n" +
                                  " ".join(map(str, ids[:P])) + "\n")
    print(json.dumps({"P": P, "h": h, "w": w, "t": float(t[0]), "r": float(r[0]), "g": float(g[0])}))


if __name__ == "__main__":
    if sys.argv[1] == "dump":
        dump(*sys.argv[2:6])
    elif sys.argv[1] == "compare":
        sys.exit(1 if compare(sys.argv[2]) else 0)
    else:
        sys.exit(__doc__)
