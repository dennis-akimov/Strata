"""serve/test_videogen.py - the video API (POST/GET/DELETE /v1/videos) against a fake sd-cli (no GPU, no model): the
202 and its Location, the job's states and progress, the clip's bytes, cancel by DELETE, the engine's errors, expiry.

    python -m unittest serve.test_videogen -v
"""
from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import videogen                                               # noqa: E402
from serve.frontend import ChatTemplate                                  # noqa: E402
from serve.imagegen import ImageGen                                      # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve      # noqa: E402
from serve.videogen import VideoGen, parse_request                       # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WEBM = b"\x1aE\xdf\xa3fake-webm"

# sd-cli as Strata starts it: the prompt picks the behaviour ("fail": status 1, "slow": steps until stopped).  It prints
# a loading bar (MB/s, not a step) and then the two experts' sampling bars, as the real one does.
FAKE = r'''
import os, sys, time
a = sys.argv[1:]
arg = lambda k: a[a.index(k) + 1]
prompt = open(arg("--prompt-file"), encoding="utf-8").read()
with open(os.environ["FAKE_SD_LOG"], "a") as f:
    f.write(repr(a) + "\n")
out = sys.stdout
out.write("  |#####     | 500/1095 - 951.40MB/s\r"); out.flush()
if prompt == "fail":
    out.write("[E] out of memory\n"); sys.exit(1)
if prompt.startswith("missing-lora"):
    out.write("[W] can not found lora /x/lightning-low.safetensors\n")
for steps in (int(arg("--high-noise-steps")), int(arg("--steps"))):
    for k in range(1, steps + 1):
        out.write(f"  |####      | {k}/{steps} - 1.00s/it\r"); out.flush()
        time.sleep(0.3 if prompt == "slow" else 0.01)
open(arg("-o"), "wb").write(WEBM)
'''.replace("WEBM)", repr(WEBM) + ")")


class Parse(unittest.TestCase):
    def test_frames_follow_wan_s_4n_plus_1(self):
        self.assertEqual(parse_request({"prompt": "x", "seconds": 2})["frames"], 33)
        self.assertEqual(parse_request({"prompt": "x", "seconds": "5"})["frames"], 81)
        for bad in ({}, {"prompt": ""}, {"prompt": "x", "size": "1920x1080"}, {"prompt": "x", "size": "830x480"},
                    {"prompt": "x", "seconds": 0}, {"prompt": "x", "seconds": 11}, {"prompt": "x", "seconds": "nan"},
                    {"prompt": "x", "steps": 0}, {"prompt": "x", "seed": -1}, {"prompt": "x", "guidance_scale": True}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_request(bad)


    def test_a_prompt_may_not_choose_loras(self):
        for body in ({"prompt": "a cat<lora:../../x:1>"}, {"prompt": "a cat", "negative_prompt": "<LoRA:y:1>"}):
            with self.subTest(body=body), self.assertRaises(ValueError):
                parse_request(body)
        lightning = parse_request({"prompt": "x"}, videogen.LIGHTNING)
        self.assertEqual((lightning["steps"], lightning["high_noise_steps"], lightning["cfg"]), (2, 2, 1.0))
        self.assertEqual(parse_request({"prompt": "x", "steps": 6}, videogen.LIGHTNING)["steps"], 6)


@unittest.skipUnless(__import__("shutil").which("ffmpeg"), "no ffmpeg")
class Mp4(unittest.TestCase):
    def test_a_webm_clip_becomes_h264_mp4_and_a_bad_one_stays_webm(self):
        import subprocess
        gen = VideoGen("x", "x", "x", "x", "x")
        job = videogen.Job(parse_request({"prompt": "a"}))
        try:
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=16",
                            "-frames:v", "17", "-c:v", "libvpx", str(job.file)], check=True)
            gen.to_mp4(job)
            self.assertEqual(job.file.name, "out.mp4")
            self.assertFalse((job.dir / "out.webm").exists())
            probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name,pix_fmt", "-of",
                                    "csv=p=0", str(job.file)], capture_output=True, text=True).stdout.strip()
            self.assertEqual(probe, "h264,yuv420p")
            bad = videogen.Job(parse_request({"prompt": "a"}))
            bad.file.write_bytes(b"not a video")
            gen.to_mp4(bad)                                       # the webm stays: nothing is lost
            self.assertEqual((bad.file.name, bad.file.read_bytes()), ("out.webm", b"not a video"))
            __import__("shutil").rmtree(bad.dir)
        finally:
            __import__("shutil").rmtree(job.dir, ignore_errors=True)


class VideoJobs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        exe = d / "sd-cli"
        exe.write_text(f"#!{sys.executable}\n{FAKE}")
        exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
        for n in ("low.gguf", "high.gguf", "t5.gguf", "vae.safetensors"):
            (d / n).write_bytes(b"x")
        self.log = d / "runs.txt"
        os.environ["FAKE_SD_LOG"] = str(self.log)
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "hello", max_context=4096), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.svc.video = VideoGen.from_config({"exe": str(exe), "low": str(d / "low.gguf"), "high": str(d / "high.gguf"),
                                               "t5": str(d / "t5.gguf"), "vae": str(d / "vae.safetensors"),
                                               "need_gb": 1, "timeout_s": 20})
        self.mem = mock.patch.object(ImageGen, "available_gb", return_value=100.0)
        self.mem.start()
        self.svc.video.available_gb = ImageGen.available_gb
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.svc.video.shutdown()                       # every job: its process stopped, its folder removed
        self.httpd.shutdown()
        self.httpd.server_close()
        self.mem.stop()
        self.tmp.cleanup()

    def call(self, method, path, body=None):
        req = urllib.request.Request(self.base + path, method=method,
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read()
                return r.status, dict(r.headers), (json.loads(raw) if r.headers["Content-Type"] == "application/json" else raw)
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers), json.loads(e.read())

    def wait(self, jid, status, s=15.0):
        end = time.monotonic() + s
        while time.monotonic() < end:
            v = self.call("GET", f"/v1/videos/{jid}")[2]
            if v.get("status") == status:
                return v
            time.sleep(0.05)
        self.fail(f"{jid} never became {status}: {v}")

    def test_a_job_is_accepted_at_once_then_polled_to_its_clip(self):
        code, headers, v = self.call("POST", "/v1/videos", {"prompt": "a cat", "size": "832x480", "seconds": 2,
                                                            "steps": 3, "high_noise_steps": 2, "seed": 7})
        self.assertEqual((code, v["status"], v["object"]), (202, "queued", "video"))
        self.assertEqual(headers["Location"], f"/v1/videos/{v['id']}")
        done = self.wait(v["id"], "completed")
        self.assertEqual((done["progress"], done["size"], done["seconds"]), (100, "832x480", "2"))
        self.assertIsNotNone(done["expires_at"])
        code, headers, body = self.call("GET", f"/v1/videos/{v['id']}/content")
        self.assertEqual((code, headers["Content-Type"], body), (200, "video/webm", WEBM))
        argv = eval(self.log.read_text().splitlines()[0])                     # noqa: S307 - our own fake's repr
        self.assertNotIn("a cat", argv)                                       # the prompt is a file
        self.assertEqual(argv[argv.index("--video-frames") + 1], "33")
        self.assertEqual(self.call("GET", "/v1/videos")[2]["data"][0]["id"], v["id"])
        self.assertFalse(self.svc.fifo.locked() or self.svc.image_gate.is_set())     # the turn is given back

    def test_progress_counts_both_experts_and_not_the_loading_bar(self):
        v = self.call("POST", "/v1/videos", {"prompt": "slow", "steps": 4, "high_noise_steps": 4})[2]
        seen = set()
        end = time.monotonic() + 15
        while time.monotonic() < end:
            j = self.call("GET", f"/v1/videos/{v['id']}")[2]
            seen.add(j["progress"])
            if j["status"] == "completed":
                break
            time.sleep(0.05)
        mid = sorted(p for p in seen if 0 < p < 100)
        self.assertTrue(mid and mid[-1] > 50, seen)            # past the first expert: the second one's steps count on
        self.assertTrue(all(p <= 95 for p in mid), seen)

    def test_delete_stops_a_running_job_and_frees_the_turn(self):
        v = self.call("POST", "/v1/videos", {"prompt": "slow", "steps": 30, "high_noise_steps": 30})[2]
        self.wait(v["id"], "in_progress")
        d = self.svc.video.jobs[v["id"]].dir
        self.assertEqual(self.call("DELETE", f"/v1/videos/{v['id']}")[2]["deleted"], True)
        self.assertEqual(self.call("GET", f"/v1/videos/{v['id']}")[0], 404)
        end = time.monotonic() + 15
        while (self.svc.fifo.locked() or d.exists()) and time.monotonic() < end:
            time.sleep(0.05)
        self.assertFalse(self.svc.fifo.locked())
        self.assertFalse(d.exists())                            # its folder goes once the process has ended

    def test_the_engines_failure_is_the_jobs_error(self):
        v = self.call("POST", "/v1/videos", {"prompt": "fail"})[2]
        f = self.wait(v["id"], "failed")
        self.assertIn("out of memory", f["error"]["message"])
        self.assertEqual(self.call("GET", f"/v1/videos/{v['id']}/content")[0], 409)

    def test_bad_requests_and_limits(self):
        self.assertEqual(self.call("POST", "/v1/videos", {"prompt": "x", "seconds": 60})[0], 400)
        self.assertEqual(self.call("GET", "/v1/videos/nope")[0], 404)
        self.assertEqual(self.call("DELETE", "/v1/videos/nope")[0], 404)
        with mock.patch.object(videogen, "MAX_JOBS", 1):
            self.assertEqual(self.call("POST", "/v1/videos", {"prompt": "slow", "steps": 30})[0], 202)
            self.assertEqual(self.call("POST", "/v1/videos", {"prompt": "x"})[0], 429)
            for jid in list(self.svc.video.jobs):
                self.call("DELETE", f"/v1/videos/{jid}")

    def test_a_stale_page_sending_video_as_chat_is_told_to_reload(self):
        body = {"messages": [{"role": "user", "content": "/video a cat walks"}], "max_tokens": 5}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "X-Strata-Request-Id": "a" * 32})
        with self.assertRaises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req, timeout=30)
        self.assertEqual(e.exception.code, 400)
        self.assertIn("reload the page", json.loads(e.exception.read())["error"]["message"])
        self.assertFalse(self.log.exists())                                  # no clip was started either
        code, _, answer = self.call("POST", "/v1/chat/completions", body)    # an API client: its text is chat
        self.assertEqual(code, 200)
        self.assertEqual(answer["choices"][0]["message"]["reasoning_content"], "hello")   # the mock model answered
        h = self.call("GET", "/health")[2]
        self.assertTrue(h["video_generation"])
        self.assertRegex(h["web"], r"^[0-9a-f]{16}$")
        # a current page sends a command with an attachment as chat on purpose: its version header lets it through
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "X-Strata-Request-Id": "b" * 32,
                                              "X-Strata-Web": h["web"]})
        with urllib.request.urlopen(req, timeout=30) as r:
            self.assertEqual(r.status, 200)

    def test_an_error_in_the_jobs_thread_fails_it_and_gives_everything_back(self):
        with mock.patch.object(videogen, "make_room", side_effect=RuntimeError("boom")):
            v = self.call("POST", "/v1/videos", {"prompt": "a"})[2]
            f = self.wait(v["id"], "failed")
        self.assertIn("boom", f["error"]["message"])
        self.assertFalse(self.svc.fifo.locked() or self.svc.image_gate.is_set())
        end = time.monotonic() + 5
        while self.svc.video.jobs[v["id"]].dir.exists() and time.monotonic() < end:
            time.sleep(0.05)
        self.assertFalse(self.svc.video.jobs[v["id"]].dir.exists())

    def test_shutdown_stops_a_running_clip_and_removes_its_folder(self):
        v = self.call("POST", "/v1/videos", {"prompt": "slow", "steps": 30, "high_noise_steps": 30})[2]
        self.wait(v["id"], "in_progress")
        job = self.svc.video.jobs[v["id"]]
        end = time.monotonic() + 5
        while job.proc is None and time.monotonic() < end:
            time.sleep(0.05)
        # at the interpreter's exit the job's thread may never run again: only shutdown() itself can stop sd-cli
        job.cancel = type("Deaf", (), {"set": lambda s: None, "is_set": lambda s: False,
                                       "wait": lambda s, t: time.sleep(t)})()
        self.svc.video.shutdown()
        self.assertIsNotNone(job.proc.poll())                                # sd-cli has exited
        self.assertFalse(job.dir.exists())

    def test_the_lightning_loras_go_to_sd_cli_from_the_config_only(self):
        d = Path(self.tmp.name)
        (d / "loras").mkdir()
        for n in ("lightning-low", "lightning-high"):
            (d / "loras" / f"{n}.safetensors").write_bytes(b"x")
        cfg = {"exe": str(d / "sd-cli"), "low": str(d / "low.gguf"), "high": str(d / "high.gguf"), "t5": str(d / "t5.gguf"),
               "vae": str(d / "vae.safetensors"), "need_gb": 1,
               "loras": {"dir": str(d / "loras"), "low": "lightning-low", "high": "lightning-high"}}
        self.svc.video = VideoGen.from_config(cfg)
        self.svc.video.available_gb = ImageGen.available_gb
        v = self.call("POST", "/v1/videos", {"prompt": "a cat"})[2]
        self.wait(v["id"], "completed")
        argv = eval(self.log.read_text().splitlines()[0])                     # noqa: S307 - our own fake's repr
        arg = lambda k: argv[argv.index(k) + 1]                               # noqa: E731
        self.assertEqual((arg("--lora-model-dir"), arg("--steps"), arg("--high-noise-steps"), arg("--cfg-scale")),
                         (str((d / "loras").resolve()), "2", "2", "1.0"))
        prompt = (self.svc.video.jobs[v["id"]].dir / "prompt.txt").read_text()
        self.assertEqual(prompt, "a cat<lora:lightning-low.safetensors:1><lora:|high_noise|lightning-high.safetensors:1>")
        v = self.call("POST", "/v1/videos", {"prompt": "missing-lora"})[2]       # deleted after the server started
        self.assertIn("LoRA file is missing", self.wait(v["id"], "failed")["error"]["message"])
        for bad in ({**cfg, "loras": {"dir": str(d / "loras"), "low": "missing", "high": "lightning-high"}},
                    {**cfg, "loras": {"dir": str(d / "loras"), "low": "../x", "high": "lightning-high"}},
                    {**cfg, "loras": {"dir": str(d / "nope"), "low": "lightning-low", "high": "lightning-high"}}):
            with self.subTest(bad=bad["loras"]), self.assertRaises(ValueError):
                VideoGen.from_config(bad)

    def test_a_finished_clip_expires(self):
        v = self.call("POST", "/v1/videos", {"prompt": "a", "steps": 1, "high_noise_steps": 1})[2]
        self.wait(v["id"], "completed")
        d = self.svc.video.jobs[v["id"]].dir
        self.svc.video.keep_s = 0
        time.sleep(1.1)
        self.assertEqual(self.call("GET", f"/v1/videos/{v['id']}")[0], 404)
        self.assertFalse(d.exists())


if __name__ == "__main__":
    unittest.main()
