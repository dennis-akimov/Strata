"""tools/video/test_lora_alpha_f32.py - the LoRA alpha converter on small hand-made safetensors files.

    python -m unittest tools.video.test_lora_alpha_f32 -v
"""
import json
import os
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))
import lora_alpha_f32 as conv  # noqa: E402


def st(tensors, header=None):
    """A safetensors file: tensors = [(name, dtype, shape, bytes)] in order; header, when given, replaces the one made."""
    h, off, data = {}, 0, b""
    for name, dtype, shape, raw in tensors:
        h[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + len(raw)]}
        off += len(raw)
        data += raw
    hb = (header if header is not None else json.dumps(h)).encode()
    return struct.pack("<Q", len(hb)) + hb + data


def tensors(path):
    b = open(path, "rb").read()
    n = struct.unpack("<Q", b[:8])[0]
    h = json.loads(b[8:8 + n])
    h.pop("__metadata__", None)
    return {k: (v["dtype"], b[8 + n + v["data_offsets"][0]:8 + n + v["data_offsets"][1]]) for k, v in h.items()}


class Convert(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.src, self.dst = os.path.join(self.d.name, "in.safetensors"), os.path.join(self.d.name, "out.safetensors")

    def tearDown(self):
        self.d.cleanup()

    def run_on(self, blob):
        open(self.src, "wb").write(blob)
        conv.main(self.src, self.dst)
        return tensors(self.dst)

    def test_integer_alphas_become_float32_and_the_rest_stays(self):
        w = struct.pack("<4f", 1, 2, 3, 4)
        out = self.run_on(st([("a.alpha", "I64", [], struct.pack("<q", 8)), ("a.lora_up.weight", "F32", [2, 2], w),
                              ("b.alpha", "U64", [], struct.pack("<Q", 16)), ("idx", "I32", [2], struct.pack("<2i", 7, 9))]))
        self.assertEqual(out["a.alpha"], ("F32", struct.pack("<f", 8.0)))
        self.assertEqual(out["b.alpha"], ("F32", struct.pack("<f", 16.0)))
        self.assertEqual(out["a.lora_up.weight"], ("F32", w))
        self.assertEqual(out["idx"], ("I32", struct.pack("<2i", 7, 9)))        # not an alpha: byte for byte
        n = struct.unpack("<Q", open(self.dst, "rb").read(8))[0]
        self.assertEqual((8 + n) % 8, 0)                                        # the data starts aligned

    def test_bad_input_is_refused_and_leaves_no_output(self):
        good = ("a.alpha", "I64", [], struct.pack("<q", 8))
        cases = {
            "inexact alpha": st([("a.alpha", "I64", [], struct.pack("<q", 16777217))]),
            "duplicate key": st([good], header='{"a.alpha":{"dtype":"I64","shape":[],"data_offsets":[0,8]},'
                                               '"a.alpha":{"dtype":"I64","shape":[],"data_offsets":[0,8]}}'),
            "overlap": st([good, good], header='{"a":{"dtype":"I64","shape":[],"data_offsets":[0,8]},'
                                               '"b":{"dtype":"I64","shape":[],"data_offsets":[0,8]}}'),
            "wrong size": st([("w", "F32", [2], struct.pack("<f", 1))]),
            "truncated": st([("w", "F32", [2], struct.pack("<2f", 1, 2))])[:-3],
            "unknown dtype": st([("w", "Q4", [1], b"x")]),
        }
        for why, blob in cases.items():
            with self.subTest(why):
                open(self.src, "wb").write(blob)
                with self.assertRaises(ValueError):
                    conv.main(self.src, self.dst)
                self.assertFalse(os.path.exists(self.dst) or os.path.exists(self.dst + ".tmp"))


if __name__ == "__main__":
    unittest.main()
