"""Nucleus-Image's DiT (17B MoE) to one GGUF file for strata-image (metal/image/strata_image.cpp).

    python tools/image/nucleus_gguf.py --src <mlx-nucleus-image>/dit --out nucleus-dit-bf16.gguf [--experts q4_0]
    python tools/image/nucleus_gguf.py --vae --src <mlx-nucleus-image>/vae --out nucleus-vae.gguf

Reads the BF16 safetensors of treadon/mlx-nucleus-image (dit/ and its config.json) and writes them one tensor at a time
(the 34 GB model is never in memory at once).  Tensors keep their names; every one is BF16 as published, except the
routed experts with --experts q8_0 / q4_0 (32-value blocks: the down projection is 1344 wide, not a multiple of 256).

ggml reads shapes innermost first, so a Linear [out, in] is a mul_mat weight as it is; the experts are stored
[E, out, in] (gate_up_proj [64, 2048, 2688] and down_proj [64, 1344, 2048] are [E, in, out] in the port: x @ W[e]), so
ggml sees [in, out, E] and multiplies all 64 experts in one batched mul_mat.
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
LLAMA = sorted(HERE.parents[2].glob("build-metal/_deps/llama_cpp-src-*/gguf-py"), key=lambda p: p.stat().st_mtime)
if LLAMA:
    sys.path.insert(0, str(LLAMA[-1]))
import gguf  # noqa: E402  (llama.cpp's gguf-py, from the engine's build)

ARCH = "nucleus-dit"
# RoPE rotates the pairs (2i, 2i+1) of a head; ggml's rope_multi rotates (i, i + head_dim/2).  The query and key
# projections' outputs (and their norms' weights) are stored with each head's dims de-interleaved - evens, then odds -
# so ggml's rotation is the port's; a query-key dot product does not change when both are permuted alike.
ROPE_PERMUTED = ("attn.to_q.weight", "attn.to_k.weight", "attn.add_k_proj.weight",
                 "attn.norm_q.weight", "attn.norm_k.weight", "attn.norm_added_k.weight")


def deinterleave(arr, head_dim):
    """Rows (a Linear's outputs) or the vector of a [head_dim] norm weight, per head: [0, 2, 4, ..., 1, 3, 5, ...]."""
    order = np.concatenate([np.arange(0, head_dim, 2), np.arange(1, head_dim, 2)])
    rows = arr.reshape(-1, head_dim, *arr.shape[1:])
    return np.ascontiguousarray(rows[:, order].reshape(arr.shape))


SAFE_DTYPES = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32}   # BF16: the bits, as uint16


def load_safetensors(src: Path):
    """{name: array}: every *.safetensors in `src`, memory-mapped (read when used; the 34 GB is never in memory at once).
    A safetensors file is an 8-byte little-endian header length, a JSON header, then the tensors' bytes."""
    out = {}
    for f in sorted(glob.glob(str(src / "*.safetensors"))):
        with open(f, "rb") as fh:
            n = int.from_bytes(fh.read(8), "little")
            header = json.loads(fh.read(n))
        for name, t in header.items():
            if name == "__metadata__":
                continue
            lo, hi = t["data_offsets"]
            out[name] = np.memmap(f, dtype=SAFE_DTYPES[t["dtype"]], mode="r", offset=8 + n + lo,
                                  shape=tuple(t["shape"]))
            assert out[name].nbytes == hi - lo, name
    return out


def to_numpy(a):
    return np.asarray(a)


def write_vae(src: Path, out: str):
    """The VAE decoder (26M weights) as float32: ggml's Metal convolution takes F32/F16 kernels.  The port's convs are
    NHWC [out, kh, kw, in]; ggml's conv_2d kernel is [kw, kh, in, out] innermost first, i.e. [out, in, kh, kw]."""
    cfg = json.loads((src / "config.json").read_text())
    tensors = load_safetensors(src)
    w = gguf.GGUFWriter(out + ".tmp", "nucleus-vae")
    w.add_array("nucleus-vae.latents_mean", [float(x) for x in cfg["latents_mean"]])
    w.add_array("nucleus-vae.latents_std", [float(x) for x in cfg["latents_std"]])
    n = 0
    for name in sorted(tensors):
        if ".time_conv." in name:                  # the video model's temporal upsampling: unused for one frame
            continue
        arr = to_numpy(tensors[name])
        if arr.dtype == np.uint16:
            arr = (arr.astype(np.uint32) << 16).view(np.float32)
        if arr.ndim == 4:
            arr = arr.transpose(0, 3, 1, 2)
        w.add_tensor(name, np.ascontiguousarray(arr, dtype=np.float32))
        n += 1
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    os.replace(out + ".tmp", out)
    print(f"{out}: {n} tensors")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="the port's dit/ folder (safetensors + config.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--experts", choices=["bf16", "q8_0", "q4_0"], default="bf16")
    ap.add_argument("--vae", action="store_true", help="--src is the port's vae/ folder")
    a = ap.parse_args()
    src = Path(a.src)
    if a.vae:
        return write_vae(src, a.out)
    cfg = json.loads((src / "config.json").read_text())
    tensors = load_safetensors(src)
    qtype = {"q8_0": gguf.GGMLQuantizationType.Q8_0, "q4_0": gguf.GGMLQuantizationType.Q4_0}.get(a.experts)

    w = gguf.GGUFWriter(a.out + ".tmp", ARCH)
    k = lambda name: f"{ARCH}.{name}"          # noqa: E731
    heads, head_dim = cfg["num_attention_heads"], cfg["attention_head_dim"]
    w.add_uint32(k("hidden"), heads * head_dim)
    w.add_uint32(k("heads"), heads)
    w.add_uint32(k("kv_heads"), cfg["num_key_value_heads"])
    w.add_uint32(k("head_dim"), head_dim)
    w.add_uint32(k("layers"), cfg["num_layers"])
    w.add_uint32(k("experts"), cfg["num_experts"])
    w.add_uint32(k("moe_dim"), cfg["moe_intermediate_dim"])
    w.add_uint32(k("text_dim"), cfg["joint_attention_dim"])
    w.add_uint32(k("in_channels"), cfg["in_channels"])
    w.add_uint32(k("out_channels"), cfg["out_channels"])
    w.add_uint32(k("patch"), cfg["patch_size"])
    w.add_float32(k("route_scale"), float(cfg.get("route_scale", 2.5)))
    w.add_array(k("capacity_factors"), [float(x) for x in cfg["capacity_factors"]])
    w.add_array(k("axes_dims"), [int(x) for x in cfg["axes_dims_rope"]])
    w.add_float32(k("rope_theta"), 10000.0)
    w.add_string(k("experts_type"), a.experts)
    w.add_string(k("rope_layout"), "neox")          # see ROPE_PERMUTED; strata-image refuses a file without it

    plan = []                                   # (name, numpy shape, kind)
    for name in sorted(tensors):
        t = tensors[name]
        shape = list(t.shape)
        expert = name.endswith(("experts.gate_up_proj", "experts.down_proj"))
        if expert:
            shape = [shape[0], shape[2], shape[1]]          # [E, out, in]
        if expert and qtype is not None:
            bs, ts = gguf.GGML_QUANT_SIZES[qtype]
            if shape[-1] % bs:
                raise SystemExit(f"{name}: {shape[-1]} is not a multiple of {bs}")
            byte_shape = shape[:-1] + [shape[-1] // bs * ts]
            w.add_tensor_info(name, byte_shape, np.dtype(np.uint8), int(np.prod(byte_shape)), raw_dtype=qtype)
            plan.append((name, shape, "q"))
        else:
            w.add_tensor_info(name, shape, np.dtype(np.uint16), int(np.prod(shape)) * 2,
                              raw_dtype=gguf.GGMLQuantizationType.BF16)
            plan.append((name, shape, "bf16"))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    total = 0
    for i, (name, shape, kind) in enumerate(plan):
        arr = to_numpy(tensors[name])
        if name.endswith(("experts.gate_up_proj", "experts.down_proj")):
            arr = np.ascontiguousarray(arr.transpose(0, 2, 1))
        if name.endswith(ROPE_PERMUTED):
            arr = deinterleave(arr, head_dim)
        if kind == "q":
            f32 = (arr.astype(np.uint32) << 16).view(np.float32)       # bf16 bits -> float32, exactly
            arr = gguf.quants.quantize(f32, qtype)
        w.write_tensor_data(arr)
        total += arr.nbytes
        if i % 50 == 0:
            print(f"  {i + 1}/{len(plan)} {name} {total / 1e9:.1f} GB", flush=True)
    w.close()
    os.replace(a.out + ".tmp", a.out)
    print(f"{a.out}: {len(plan)} tensors, {total / 1e9:.2f} GB, experts {a.experts}")


if __name__ == "__main__":
    main()
