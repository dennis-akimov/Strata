"""A LoRA safetensors file with integer `alpha` scalars (lightx2v's Wan2.2-Lightning: I64) rewritten with float32 ones,
which stable-diffusion.cpp loads ("type i32 unsupported for integer quantization").  Every other tensor is copied byte
for byte; an alpha that float32 cannot hold exactly is refused.  The input is checked first (header, dtypes, shapes,
offsets that tile the data exactly), and the output is written to a temporary file that replaces OUT only when complete.

    python tools/video/lora_alpha_f32.py IN.safetensors OUT.safetensors
"""
import json
import math
import os
import struct
import sys

# safetensors dtypes: bytes an element, and the struct format of an integer one
SIZES = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E5M2": 1, "F8_E4M3": 1, "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
         "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8}
INTS = {"I8": "<b", "U8": "<B", "I16": "<h", "U16": "<H", "I32": "<i", "U32": "<I", "I64": "<q", "U64": "<Q"}
MAX_HEADER = 100 << 20


def unique_keys(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise ValueError(f"the header names {k!r} twice")
        out[k] = v
    return out


def read_header(f, size):
    """The header's tensors, sorted by offset, after every check; the data's start."""
    raw = f.read(8)
    if len(raw) != 8:
        raise ValueError("not a safetensors file: shorter than 8 bytes")
    n = struct.unpack("<Q", raw)[0]
    if not 2 <= n <= MAX_HEADER or 8 + n > size:
        raise ValueError(f"a header of {n} bytes does not fit")
    hdr = json.loads(f.read(n), object_pairs_hook=unique_keys)
    if not isinstance(hdr, dict):
        raise ValueError("the header is not a JSON object")
    meta = hdr.pop("__metadata__", None)
    base, end = 8 + n, 0
    items = []
    for name, t in hdr.items():
        if not isinstance(t, dict) or t.get("dtype") not in SIZES or not isinstance(t.get("shape"), list) or \
                not all(isinstance(d, int) and d >= 0 for d in t["shape"]) or \
                not (isinstance(t.get("data_offsets"), list) and len(t["data_offsets"]) == 2 and
                     all(isinstance(o, int) for o in t["data_offsets"])):
            raise ValueError(f"{name}: not a valid tensor entry")
        a, b = t["data_offsets"]
        if b - a != math.prod(t["shape"]) * SIZES[t["dtype"]]:
            raise ValueError(f"{name}: {b - a} bytes for shape {t['shape']} of {t['dtype']}")
        items.append((a, b, name, t))
    items.sort()
    for a, b, name, _ in items:                   # the tensors tile the data from 0, no gap, no overlap
        if a != end:
            raise ValueError(f"{name}: starts at {a}, not at {end}")
        end = b
    if base + end != size:
        raise ValueError(f"the data is {size - base} bytes, the tensors {end}")
    return items, meta, base


def main(src, dst):
    size = os.path.getsize(src)
    with open(src, "rb") as f:
        items, meta, base = read_header(f, size)
        out, blobs, off = {}, [], 0
        for a, b, name, t in items:
            data = f.read(b - a)
            if len(data) != b - a:
                raise ValueError(f"{name}: the file ends early")
            if t["dtype"] in INTS and name.endswith(".alpha") and t["shape"] in ([], [1]):
                (v,) = struct.unpack(INTS[t["dtype"]], data)
                packed = struct.pack("<f", float(v))
                if struct.unpack("<f", packed)[0] != v:
                    raise ValueError(f"{name}: {v} has no exact float32")
                data, t = packed, {**t, "dtype": "F32"}
            out[name] = {**t, "data_offsets": [off, off + len(data)]}
            blobs.append(data)
            off += len(data)
    if meta is not None:
        out["__metadata__"] = meta
    h = json.dumps(out, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)                       # the data starts 8-byte aligned
    tmp = dst + ".tmp"
    with open(tmp, "wb") as g:
        g.write(struct.pack("<Q", len(h)) + h)
        for d in blobs:
            g.write(d)
        g.flush()
        os.fsync(g.fileno())
    os.replace(tmp, dst)                            # only a complete file has the final name


if __name__ == "__main__":
    try:
        main(*sys.argv[1:3])
    except (ValueError, OSError) as e:
        sys.exit(f"lora_alpha_f32: {e}")
