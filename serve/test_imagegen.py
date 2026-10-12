"""serve/test_imagegen.py - POST /v1/images/generations against a fake strata-image (no GPU, no model): the turn, the
memory rule, cancellation, the deadline and the engine's errors of serve/imagegen.py.

    python -m unittest serve.test_imagegen -v
"""
from __future__ import annotations

import base64
import json
import math
import os
import socket
import stat
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import imagegen                                               # noqa: E402
from serve.frontend import ChatTemplate                                  # noqa: E402
from serve.imagegen import ImageGen                                      # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve      # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PNG = b"\x89PNG\r\n\x1a\nfake"

# strata-image's protocol (metal/image/strata_image.cpp): JSON lines on stdout, the PNG at --out.  The prompt picks the
# behaviour: "fail" (an error line), "crash" (no line, status 2), "slow" (steps until stopped), "stubborn" (SIGTERM is
# ignored).  Every run is logged to $FAKE_ENGINE_LOG: its arguments, its prompt files and its pid.
FAKE = r'''
import json, os, signal, sys, time
a = dict(zip(sys.argv[1::2], sys.argv[2::2]))
prompt = open(a["--prompt-file"], encoding="utf-8").read()
negative = open(a["--negative-file"], encoding="utf-8").read()
with open(os.environ["FAKE_ENGINE_LOG"], "a") as f:
    f.write(json.dumps({"argv": sys.argv[1:], "prompt": prompt, "negative": negative, "pid": os.getpid()}) + "\n")
say = lambda o: (print(json.dumps(o)), sys.stdout.flush())
say({"phase": "text"})
if prompt == "fail":
    say({"error": "the text encoder failed"}); sys.exit(1)
if prompt == "crash":
    sys.stderr.write("Assertion failed: GGML_ASSERT\n"); sys.exit(2)
if prompt == "stubborn":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
steps = 150 if prompt in ("slow", "stubborn") else int(a["--steps"])
for i in range(steps):
    if prompt in ("slow", "stubborn"):
        time.sleep(0.2)
    say({"step": i + 1, "steps": steps, "s": 0.01})
open(a["--out"], "wb").write(PNG)
say({"done": True, "text_s": 0.1, "denoise_s": 0.2, "decode_s": 0.1, "total_s": 0.4})
'''.replace("PNG)", repr(PNG) + ")")


class UnloadableEngine(MockEngine):
    """A mock engine that can be unloaded, as the real one can."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.up, self.unloads = True, 0

    def alive(self):
        return self.up

    def unload(self):
        self.up, self.unloads = False, self.unloads + 1


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


class ImageGeneration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        exe = d / "strata-image"
        exe.write_text(f"#!{sys.executable}\n{FAKE}")
        exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
        for name in ("dit.gguf", "enc.gguf", "vae.gguf"):
            (d / name).write_bytes(b"GGUF")
        self.log = d / "runs.jsonl"
        os.environ["FAKE_ENGINE_LOG"] = str(self.log)
        tok = ByteTokenizer()
        self.engine = UnloadableEngine(tok, "hello", max_context=4096)
        self.svc = Service(self.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.svc.image = ImageGen.from_config({"exe": str(exe), "dit": str(d / "dit.gguf"), "encoder": str(d / "enc.gguf"),
                                               "vae": str(d / "vae.gguf"), "queue_s": 5, "timeout_s": 10})
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.tmp.cleanup()

    def runs(self):
        return [json.loads(x) for x in self.log.read_text().splitlines()] if self.log.exists() else []

    def wait_for(self, cond, s=10.0):
        end = time.monotonic() + s
        while not cond() and time.monotonic() < end:
            time.sleep(0.05)
        return cond()

    def post(self, body, path="/v1/images/generations", headers=None, timeout=30):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, dict(r.headers), json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers), json.loads(e.read())

    def post_async(self, body, headers=None):
        out = {}
        t = threading.Thread(target=lambda: out.setdefault("r", self.post(body, headers=headers)))
        t.start()
        return t, out

    def mem(self, *values):
        """available_gb() answers these in turn (the last one repeats)."""
        seq = list(values)
        return mock.patch.object(ImageGen, "available_gb", side_effect=lambda: seq.pop(0) if len(seq) > 1 else seq[0])

    def test_a_picture_is_made_in_the_servers_turn(self):
        with self.mem(100.0):
            t, _ = self.post_async({"prompt": "slow"}, headers={"X-Strata-Request-Id": "a" * 32})
            self.assertTrue(self.wait_for(lambda: self.runs()))
            self.assertTrue(self.svc.fifo.locked() and self.svc.image_gate.is_set())   # held while it runs
            self.post({"id": "a" * 32}, path="/cancel")
            t.join(30)
            self.assertEqual(self.post({"prompt": "x", "size": "1024x1536", "steps": 1})[0], 200)      # the largest
            code, _, body = self.post({"prompt": "a red apple", "size": "512x768", "steps": 3, "seed": 7,
                                       "negative_prompt": "blurry"})
        self.assertEqual(code, 200)
        self.assertEqual(base64.b64decode(body["data"][0]["b64_json"]), PNG)
        self.assertEqual((body["strata"]["steps"], body["strata"]["seed"]), (3, 7))
        run = self.runs()[2]
        self.assertEqual((run["prompt"], run["negative"]), ("a red apple", "blurry"))
        args = dict(zip(run["argv"][::2], run["argv"][1::2]))
        self.assertEqual((args["--width"], args["--height"], args["--steps"], args["--seed"], args["--cfg"]),
                         ("512", "768", "3", "7", "4.0"))
        self.assertNotIn("a red apple", run["argv"])                      # the prompt is a file, not an argument
        self.assertEqual(args["--watch-stdin"], "1")
        self.assertFalse(Path(args["--out"]).parent.exists())             # the temporary folder is gone
        self.assertFalse(self.svc.fifo.locked() or self.svc.image_gate.is_set())       # both given back
        self.assertEqual(self.engine.unloads, 0)                          # enough memory: nothing unloaded

    def test_short_memory_unloads_the_chat_model(self):
        with self.mem(2.0, 40.0):
            code, _, _ = self.post({"prompt": "a red apple"})
        self.assertEqual((code, self.engine.unloads), (200, 1))

    def test_memory_still_short_after_the_unload_is_503_and_no_picture(self):
        with self.mem(2.0):
            code, headers, body = self.post({"prompt": "x"})
        self.assertEqual((code, headers.get("Retry-After")), (503, "60"))
        self.assertIn("even without the chat model", body["error"]["message"])
        self.assertEqual(self.runs(), [])

    def test_a_chat_model_that_cannot_be_unloaded_and_short_memory_is_503(self):
        unload = UnloadableEngine.unload
        del UnloadableEngine.unload                                       # this engine has none
        try:
            with self.mem(2.0):
                code, _, body = self.post({"prompt": "x"})
        finally:
            UnloadableEngine.unload = unload
        self.assertEqual(code, 503)
        self.assertIn("cannot be unloaded", body["error"]["message"])
        self.assertEqual(self.runs(), [])

    def test_requests_waiting_for_the_turn_do_not_stop_the_unload(self):
        with self.svc.status_lock:
            self.svc.status["queued"] = 2                                 # chat requests queued behind the picture
        with self.mem(2.0, 40.0):
            code, _, _ = self.post({"prompt": "x"})
        self.assertEqual((code, self.engine.unloads), (200, 1))

    def test_a_parallel_request_past_the_gate_is_not_unloaded_under(self):
        with self.svc.status_lock:
            self.svc.entering = 1                                         # read the gate open, not yet "busy"
        self.assertEqual(self.svc.unload_held(waiting_ok=True), "busy")
        with self.svc.status_lock:
            self.svc.entering = 0
        self.assertEqual(self.svc.unload_held(waiting_ok=True), "unloaded")

    def test_a_running_chat_request_and_short_memory_is_503(self):
        self.svc.image.queue_s = 1.5
        with self.svc.status_lock:
            self.svc.status["busy"] = True
        with self.mem(2.0):
            code, headers, _ = self.post({"prompt": "x"})
        self.assertEqual((code, headers.get("Retry-After"), self.engine.unloads), (503, "60", 0))
        self.assertEqual(self.runs(), [])

    def test_a_bad_request_is_400_before_anything_is_unloaded(self):
        with self.mem(2.0, 40.0):
            for bad in ({}, {"prompt": " "}, {"prompt": "x", "steps": 0}, {"prompt": "x", "steps": True},
                        {"prompt": "x", "size": "100x100"}, {"prompt": "x", "size": "1000x1024"},
                        {"prompt": "x", "size": "big"}, {"prompt": "x", "size": "2048x2048"},
                        {"prompt": "x", "size": "1536x1040"}, {"prompt": "x", "n": 2}, {"prompt": "x", "seed": -1},
                        {"prompt": "x", "guidance_scale": math.inf}, {"prompt": "x", "negative_prompt": 3},
                        {"prompt": "x" * 4001}, {"prompt": "x", "response_format": "url"}, {"prompt": "\ud800"},
                        {"prompt": "x", "negative_prompt": "a\udfffb"}):
                with self.subTest(bad=bad):
                    self.assertEqual(self.post(bad)[0], 400)
        self.assertEqual((self.engine.unloads, self.runs()), (0, []))

    def test_no_turn_within_queue_s_is_503_with_retry_after(self):
        self.svc.image.queue_s = 1.0
        with self.svc.fifo:                                               # a chat request holds the turn
            code, headers, _ = self.post({"prompt": "x"})
        self.assertEqual((code, headers.get("Retry-After")), (503, "60"))
        self.assertEqual(self.runs(), [])

    def test_the_engines_error_line_is_the_answer(self):
        with self.mem(100.0):
            code, _, body = self.post({"prompt": "fail"})
        self.assertEqual((code, body["error"]["message"]), (500, "the text encoder failed"))
        self.assertFalse(self.svc.fifo.locked())

    def test_an_engine_that_dies_without_a_word_is_500_with_its_stderr(self):
        with self.mem(100.0):
            code, _, body = self.post({"prompt": "crash"})
        self.assertEqual(code, 500)
        self.assertIn("status 2", body["error"]["message"])
        self.assertIn("GGML_ASSERT", body["error"]["message"])

    def test_a_missing_engine_is_500_not_a_hang(self):
        self.svc.image.exe = str(Path(self.tmp.name) / "gone")
        with self.mem(100.0):
            code, _, body = self.post({"prompt": "x"})
        self.assertEqual(code, 500)
        self.assertIn("could not start", body["error"]["message"])
        self.assertFalse(self.svc.fifo.locked())

    def test_past_the_deadline_is_504_and_the_turn_waits_until_the_engine_has_exited(self):
        self.svc.image.timeout_s = 0.8
        with mock.patch.object(imagegen, "KILL_AFTER_S", 1.0), self.mem(100.0):
            t0 = time.monotonic()
            code, _, _ = self.post({"prompt": "stubborn"})                # ignores SIGTERM: SIGKILL 1 s later
        self.assertEqual(code, 504)
        self.assertGreater(time.monotonic() - t0, 1.7)
        self.assertFalse(alive(self.runs()[0]["pid"]))
        self.assertFalse(self.svc.fifo.locked())

    def test_a_client_that_hangs_up_stops_the_engine_and_then_the_turn_is_given_back(self):
        with self.mem(100.0):
            s = socket.create_connection(self.httpd.server_address)
            body = json.dumps({"prompt": "slow"}).encode()
            s.sendall(b"POST /v1/images/generations HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                      + f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
            self.assertTrue(self.wait_for(lambda: self.runs()))
            s.close()                                                     # the client leaves mid-picture
            pid = self.runs()[0]["pid"]
            self.assertTrue(self.wait_for(lambda: not alive(pid)))
            self.assertTrue(self.wait_for(lambda: not self.svc.fifo.locked()))

    def test_a_cancel_from_the_page_answers_the_connected_client(self):
        rid = "c" * 32
        with self.mem(100.0):
            t, out = self.post_async({"prompt": "slow"}, headers={"X-Strata-Request-Id": rid})
            self.assertTrue(self.wait_for(lambda: self.runs()))
            self.assertEqual(self.post({"id": rid}, path="/cancel")[0], 200)
            t.join(30)
        self.assertEqual(out["r"][0], 499)                                # an answer, not a silent close
        self.assertFalse(alive(self.runs()[0]["pid"]))

    def test_health_says_whether_pictures_can_be_made(self):
        with urllib.request.urlopen(self.base + "/health", timeout=5) as r:
            self.assertIs(json.loads(r.read())["image_generation"], True)
        self.svc.image = None
        with urllib.request.urlopen(self.base + "/health", timeout=5) as r:
            self.assertIs(json.loads(r.read())["image_generation"], False)
        self.assertEqual(self.post({"prompt": "x"})[0], 404)

    def test_the_config_entry_is_checked_at_start(self):
        d = Path(self.tmp.name)
        ok = {"exe": str(d / "strata-image"), "dit": str(d / "dit.gguf"), "encoder": str(d / "enc.gguf"),
              "vae": str(d / "vae.gguf")}
        self.assertIsNone(ImageGen.from_config(None))
        g = ImageGen.from_config({**ok, "need_gb": 8})
        self.assertEqual((g.need_gb, g.queue_s, g.timeout_s), (8.0, 600.0, 1800.0))
        for bad in ({**ok, "dit": str(d / "missing.gguf")}, {**ok, "vae": 3}, {k: v for k, v in ok.items() if k != "encoder"},
                    {**ok, "exe": str(d)}, {**ok, "need_gb": 0}, {**ok, "queue_s": True}, {**ok, "queue_s": math.nan},
                    {**ok, "timeout_s": math.inf}, []):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ImageGen.from_config(bad)


if __name__ == "__main__":
    unittest.main()
