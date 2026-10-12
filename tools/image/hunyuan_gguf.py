"""HunyuanImage-3.0-Instruct-Distil (80B MoE, 169 GB of BF16 safetensors) to one GGUF for strata-image, streamed:

    python tools/image/hunyuan_gguf.py --out hunyuan-image3-q4_0.gguf [--experts q4_0|q8_0|bf16] [--work DIR]
        [--ref-subset REF_DIR]

The plan comes from the 32 shards' headers (HTTP range requests); then each shard is downloaded (aria2c, 16
connections), its tensors converted, and the shard deleted, so the disk holds one shard (~5.4 GB) and the output at a
time.  Only what text-to-image needs is kept: the 32 transformer layers, the token embeddings, the final norm, the image
head (patch_embed, final_layer, the timestep / guidance / timestep_r embedders) and the VAE decoder.  lm_head (text
generation), the SigLIP ViT and the VAE encoder (image input) are dropped.

Layout changes, each one exact (the C++ reads them back):
- attn qkv_proj [8 kv heads x (4 q + k + v) x 128, 4096] -> attn_q [4096, 4096], attn_k / attn_v [1024, 4096].
- RoPE: the model rotates (i, i + 64) with pair i at position y for even i and x for odd i; ggml_rope_multi (MROPE)
  wants each position's pairs contiguous.  The q / k rows and the q / k norm weights of every head are permuted so
  pair p < 32 is the old pair 2p (y) and pair p >= 32 the old pair 2(p - 32) + 1 (x) - a dot product of q and k is the
  same when both are permuted alike, and the RMS norm over a head does not see the order.  Key `rope_layout`.
- the 64 experts of a layer are stacked: ffn_gate_up_exps [64, 6144, 4096], ffn_down_exps [64, 4096, 3072]
  (gate_and_up_proj's first half multiplies, its second half goes through SiLU, as in the model).
- the VAE's Conv3d kernels [O, I, 3, 3, 3] become three 2D kernels `.t0/.t1/.t2` [O, I, 3, 3] (temporal taps).
"""
import argparse
import json
import os
import struct
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
LLAMA = sorted(HERE.parents[2].glob("build-metal/_deps/llama_cpp-src-*/gguf-py"), key=lambda p: p.stat().st_mtime)
if LLAMA:
    sys.path.insert(0, str(LLAMA[-1]))
import gguf  # noqa: E402

REPO = "tencent/HunyuanImage-3.0-Instruct-Distil"
BASE = f"https://huggingface.co/{REPO}/resolve/c8ffd07206f1b843697606968196e8f59f8ff38c/"   # pinned
ARCH = "hunyuan-image3"
HEAD_DIM = 128
DTYPES = {"BF16": (np.uint16, 2), "F32": (np.float32, 4), "F16": (np.float16, 2)}


def fetch(url, rng=None):
    req = urllib.request.Request(url, headers={"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except OSError:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def shard_header(name):
    """{tensor: (dtype, shape, start, end)} with the offsets in the file (the data starts after the header)."""
    n = struct.unpack("<Q", fetch(BASE + name, (0, 7)))[0]
    h = json.loads(fetch(BASE + name, (8, 8 + n - 1)))
    h.pop("__metadata__", None)
    return {k: (v["dtype"], v["shape"], 8 + n + v["data_offsets"][0], 8 + n + v["data_offsets"][1]) for k, v in h.items()}


def keep(name):
    return not name.startswith(("lm_head", "vision_model", "vision_aligner", "vae.encoder", "vae.quant_conv"))


def rope_perm():
    """New row index -> old row index within one head (see the module doc)."""
    pairs = [2 * p for p in range(32)] + [2 * p + 1 for p in range(32)]
    return np.array(pairs + [i + 64 for i in pairs])


def plan_tensors(headers, cfg):
    """[(out name, ggml kind, out shape, sources, transform)] in file order, and each one's last shard."""
    L, E = cfg["num_hidden_layers"], cfg["num_experts"]
    out = {}
    def add(name, kind, shape, sources, fn):
        out[name] = (kind, shape, sources, fn)
    where = {t: s for s, h in headers.items() for t in h}
    for t, s in where.items():
        if not keep(t) or ".mlp.experts." in t or ".self_attn.qkv_proj." in t:
            continue
        dt, shape = headers[s][t][0], headers[s][t][1]
        if t.startswith("vae.") and len(shape) == 5:
            if shape[2] == 1:
                add(t, "f32", [shape[0], shape[1], shape[3], shape[4]], [t], ("squeeze_t",))
            else:
                for k in range(shape[2]):
                    add(f"{t}.t{k}", "f32", [shape[0], shape[1], shape[3], shape[4]], [t], ("tap", k))
            continue
        name = t
        fn = ("as_is",)
        if t.endswith(("self_attn.query_layernorm.weight", "self_attn.key_layernorm.weight")):
            fn = ("perm_vec",)
        kind = "f32" if (dt == "F32" or len(shape) in (1, 4) or t.startswith("vae.")) else "bf16"
        add(name, kind, shape, [t], fn)
    nq, nkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    for l in range(L):
        p = f"model.layers.{l}."
        q = p + "self_attn.qkv_proj.weight"
        add(p + "self_attn.q_proj.weight", "bf16", [nq * HEAD_DIM, cfg["hidden_size"]], [q], ("qkv", "q"))
        add(p + "self_attn.k_proj.weight", "bf16", [nkv * HEAD_DIM, cfg["hidden_size"]], [q], ("qkv", "k"))
        add(p + "self_attn.v_proj.weight", "bf16", [nkv * HEAD_DIM, cfg["hidden_size"]], [q], ("qkv", "v"))
        for proj, rows, cols in (("gate_and_up_proj", 2 * cfg["moe_intermediate_size"][l], cfg["hidden_size"]),
                                 ("down_proj", cfg["hidden_size"], cfg["moe_intermediate_size"][l])):
            srcs = [f"{p}mlp.experts.{e}.{proj}.weight" for e in range(E)]
            add(f"{p}mlp.experts.{proj}", "exp", [E, rows, cols], srcs, ("stack",))
    last = {n: max(int(where[s].split("-")[1]) for s in v[2]) for n, v in out.items()}
    order = sorted(out, key=lambda n: (last[n], n))
    return [(n, *out[n]) for n in order], last, where


def quant_type(kind, experts):
    if kind == "f32":
        return gguf.GGMLQuantizationType.F32
    if kind == "bf16" or experts == "bf16":
        return gguf.GGMLQuantizationType.BF16
    return {"q4_0": gguf.GGMLQuantizationType.Q4_0, "q8_0": gguf.GGMLQuantizationType.Q8_0}[experts]


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def encode(arr_f32_or_bf16, qt):
    """The bytes of one tensor in type qt, from a float32 array or BF16 bits (uint16)."""
    a = arr_f32_or_bf16
    if qt == gguf.GGMLQuantizationType.BF16:
        return a if a.dtype == np.uint16 else (a.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
    f = bf16_to_f32(a) if a.dtype == np.uint16 else a.astype(np.float32)
    if qt == gguf.GGMLQuantizationType.F32:
        return f
    return gguf.quants.quantize(f, qt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--experts", choices=["q4_0", "q8_0", "bf16"], default="q4_0")
    ap.add_argument("--work", default=None, help="where one shard at a time is downloaded (default: next to --out)")
    ap.add_argument("--ref-subset", default=None,
                    help="a directory: also keep these tensors as they are (BF16), for the parity checks: layers 0 and "
                         "31, the embeddings, the image head and the VAE decoder (~11 GB)")
    ap.add_argument("--keep-shards", action="store_true")
    a = ap.parse_args()
    work = Path(a.work or Path(a.out).parent / "hunyuan-shards")
    work.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(fetch(BASE + "config.json"))
    index = json.loads(fetch(BASE + "model.safetensors.index.json"))["weight_map"]
    shards = sorted(set(index.values()))
    print(f"{len(shards)} shards; reading their headers ...", flush=True)
    headers = {s: shard_header(s) for s in shards}
    plan, last, where = plan_tensors(headers, cfg)
    print(f"{len(plan)} tensors out of {len(index)}", flush=True)

    w = gguf.GGUFWriter(a.out + ".tmp", ARCH)
    k = lambda n: f"{ARCH}.{n}"   # noqa: E731
    for key in ("hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "num_experts",
                "vocab_size"):
        w.add_uint32(k(key), int(cfg[key]))
    w.add_uint32(k("moe_topk"), int(cfg["moe_topk"][0]))
    w.add_uint32(k("moe_intermediate_size"), int(cfg["moe_intermediate_size"][0]))
    w.add_uint32(k("head_dim"), HEAD_DIM)
    w.add_float32(k("rms_norm_eps"), float(cfg["rms_norm_eps"]))
    w.add_float32(k("rope_theta"), float(cfg["rope_theta"]))
    w.add_uint32(k("latent_channels"), int(cfg["vae"]["latent_channels"]))
    w.add_uint32(k("vae_downsample"), int(cfg["vae_downsample_factor"][0]))
    w.add_float32(k("vae_scaling_factor"), float(cfg["vae"]["scaling_factor"]))
    w.add_array(k("vae_block_out_channels"), [int(x) for x in cfg["vae"]["block_out_channels"]])
    w.add_uint32(k("vae_layers_per_block"), int(cfg["vae"]["layers_per_block"]))
    w.add_uint32(k("vae_ffactor_temporal"), int(cfg["vae"]["ffactor_temporal"]))
    w.add_uint32(k("patch_embed_hidden_dim"), 1024)
    w.add_string(k("rope_layout"), "sections_yx")
    w.add_string(k("experts_type"), a.experts)
    w.add_string(k("source"), REPO)
    for name, kind, shape, srcs, fn in plan:
        qt = quant_type(kind, a.experts)
        if qt in (gguf.GGMLQuantizationType.Q4_0, gguf.GGMLQuantizationType.Q8_0):
            bs, ts = gguf.GGML_QUANT_SIZES[qt]
            if shape[-1] % bs:
                raise SystemExit(f"{name}: {shape[-1]} is not a multiple of {bs}")
            bshape = shape[:-1] + [shape[-1] // bs * ts]
            w.add_tensor_info(name, bshape, np.dtype(np.uint8), int(np.prod(bshape)), raw_dtype=qt)
        elif qt == gguf.GGMLQuantizationType.BF16:
            w.add_tensor_info(name, shape, np.dtype(np.uint16), int(np.prod(shape)) * 2, raw_dtype=qt)
        else:
            w.add_tensor_info(name, shape, np.dtype(np.float32), int(np.prod(shape)) * 4, raw_dtype=qt)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()

    ref_names = set()
    if a.ref_subset:
        for t in index:
            if keep(t) and (t.startswith(("model.layers.0.", "model.layers.31.", "patch_embed",
                                          "final_layer", "time_embed", "timestep", "guidance_emb", "vae.decoder",
                                          "model.ln_f", "model.wte"))):
                ref_names.add(t)
    ref = {}
    if a.ref_subset:
        Path(a.ref_subset).mkdir(parents=True, exist_ok=True)
    pending = {}       # stacked experts: name -> [64 encoded arrays or None]
    qkv_cache = {}
    perm = rope_perm()
    nq, nkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    groups = nq // nkv
    done = 0
    t0 = time.time()
    for si, shard in enumerate(shards, 1):
        path = work / shard
        size = max(v[3] for v in headers[shard].values())
        if not path.exists() or path.stat().st_size != size or (work / (shard + ".aria2")).exists():
            subprocess.run(["aria2c", "-q", "-x16", "-s16", "-k8M", "--file-allocation=none", "-d", str(work), "-o",
                            shard, BASE + shard], check=True)
        mm = np.memmap(path, dtype=np.uint8, mode="r")
        if mm.size != size:
            raise SystemExit(f"{shard}: {mm.size} bytes, the header says {size}")

        def read(t):
            dt, shape, s, e = headers[shard][t]
            npt, _ = DTYPES[dt]
            return np.frombuffer(mm[s:e], dtype=npt).reshape(shape)

        for t in headers[shard]:
            if t in ref_names:
                ref[t] = np.array(read(t))
            if ".mlp.experts." in t:
                l_part, e_part = t.split(".mlp.experts.")
                e, proj = e_part.split(".")[0], e_part.split(".")[1]
                name = f"{l_part}.mlp.experts.{proj}"
                qt = quant_type("exp", a.experts)
                pending.setdefault(name, [None] * cfg["num_experts"])[int(e)] = encode(read(t), qt)
            if t.endswith(".self_attn.qkv_proj.weight"):
                qkv_cache[t] = np.array(read(t))
        # write every planned tensor whose last shard is this one, in plan order
        for name, kind, shape, srcs, fn in plan:
            if last[name] != si:
                continue
            qt = quant_type(kind, a.experts)
            if fn[0] == "stack":
                parts = pending.pop(name)
                data = np.stack(parts)
            elif fn[0] == "qkv":
                W = qkv_cache[srcs[0]].reshape(nkv, groups + 2, HEAD_DIM, -1)
                if fn[1] == "q":
                    x = W[:, :groups].reshape(nq, HEAD_DIM, -1)[:, perm].reshape(nq * HEAD_DIM, -1)
                elif fn[1] == "k":
                    x = W[:, groups].reshape(nkv, HEAD_DIM, -1)[:, perm].reshape(nkv * HEAD_DIM, -1)
                else:
                    x = W[:, groups + 1].reshape(nkv * HEAD_DIM, -1)
                data = encode(np.ascontiguousarray(x), qt)
                if fn[1] == "v":
                    qkv_cache.pop(srcs[0])
            else:
                x = read(srcs[0])
                if fn[0] == "perm_vec":
                    x = x[perm]
                elif fn[0] == "tap":
                    x = x[:, :, fn[1]]
                elif fn[0] == "squeeze_t":
                    x = x[:, :, 0]
                data = encode(np.ascontiguousarray(x), qt)
            w.write_tensor_data(np.ascontiguousarray(data))
            done += 1
        for t, v in ref.items():                       # BF16 stays its bits (uint16)
            np.save(Path(a.ref_subset) / f"{t}.npy", v)
        ref.clear()
        del mm
        if not a.keep_shards:
            path.unlink()
        print(f"shard {si}/{len(shards)}: {done}/{len(plan)} tensors, {(time.time() - t0) / 60:.1f} min", flush=True)
    w.close()
    os.replace(a.out + ".tmp", a.out)
    print(f"{a.out}: {len(plan)} tensors in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
