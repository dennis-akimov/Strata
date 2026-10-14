"""Videos from a prompt: OpenAI's video API (POST /v1/videos, GET /v1/videos/{id}, GET /v1/videos/{id}/content,
DELETE /v1/videos/{id}, GET /v1/videos), made by stable-diffusion.cpp's sd-cli (Wan 2.2 A14B on its ggml/Metal backend)
with the run config's "video":

    "video": {"exe": ".../sd-cli", "low": ".../Wan2.2-T2V-A14B-LowNoise-Q4_K_M.gguf",
              "high": ".../Wan2.2-T2V-A14B-HighNoise-Q4_K_M.gguf", "t5": ".../umt5-xxl-encoder-Q8_0.gguf",
              "vae": ".../wan_2.1_vae.safetensors", "need_gb": 40, "timeout_s": 3600, "keep_s": 3600,
              "args": ["--diffusion-fa"],
              "loras": {"dir": ".../loras", "low": "lightning-low", "high": "lightning-high"}}

"loras" (optional): lightx2v's Wan2.2-Lightning, distilled to 4 steps without CFG (tools/pull_model.sh WAN2.2).  With
it a job's defaults are 2 + 2 steps and guidance 1 (one pass a step): 832x480, 33 frames took 430 s, 2907 s without.

A clip takes minutes, so this is the asynchronous request-reply pattern (bible: system-design, Azure's "Asynchronous
Request-Reply"): POST answers 202 at once with the job and a Location to poll; the job waits for the server's turn,
then runs as a picture does (serve/imagegen.py: the chat model is unloaded first when memory is short, one process per
job, prompts in files in a private folder, SIGTERM then SIGKILL on cancel or past timeout_s).  Jobs live in this
process's memory: a restart forgets them and their files.  A finished clip is kept keep_s seconds (expires_at).
"""
import math
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

from serve.imagegen import ImageGen, _stop, make_room

MAX_PROMPT_CHARS = 4000
# sd-cli writes webm (VP8), which QuickTime, Finder and Photos do not play: each clip becomes H.264 MP4 when ffmpeg is
# there (libx264, else the Mac's hardware encoder), as OpenAI's video API returns.  Without ffmpeg: the webm.
MP4 = {"libx264": ["-c:v", "libx264", "-crf", "18", "-preset", "medium"],
       "h264_videotoolbox": ["-c:v", "h264_videotoolbox", "-b:v", "6M"]}
MAX_JOBS = 8                     # queued + running + kept: more is a 429
MAX_PIXELS = 1280 * 720
FPS = 16                         # Wan 2.2 A14B's training rate
MAX_SECONDS = 10
# sd-cli's sampling bar: "|#####   | 3/10 - 12.34s/it" (its loading bar ends in MB/s instead)
STEP = re.compile(r"\|\s*(\d+)/(\d+)\s*-\s*[\d.]+\s*(?:s/it|it/s)")


# the defaults of a job: Wan 2.2's own (sd.cpp's example), and the Lightning LoRAs' (4 steps, no CFG)
PLAIN = {"steps": 10, "high_noise_steps": 8, "guidance_scale": 3.5}
LIGHTNING = {"steps": 2, "high_noise_steps": 2, "guidance_scale": 1.0}


def parse_request(body, defaults=PLAIN) -> dict:
    """OpenAI's create-video fields (prompt, seconds, size) plus steps, high_noise_steps, guidance_scale, seed and
    negative_prompt.  A bad value is a ValueError: a 400, no job."""
    if not isinstance(body, dict):
        raise ValueError("the body must be a JSON object")
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > MAX_PROMPT_CHARS:
        raise ValueError(f"prompt must be a non-empty string of at most {MAX_PROMPT_CHARS} characters")
    # sd-cli reads <lora:NAME:WEIGHT> out of the prompt and loads that file: only the server's config chooses LoRAs
    for text in (prompt, body.get("negative_prompt", "")):
        if isinstance(text, str) and "<lora:" in text.lower():
            raise ValueError("a prompt may not contain <lora:...>: the server's config chooses the LoRAs")
    try:
        width, height = (int(v) for v in str(body.get("size", "832x480")).lower().split("x"))
    except ValueError:
        raise ValueError('size must look like "832x480"') from None
    if width % 16 or height % 16 or not 256 <= min(width, height) or max(width, height) > 1280 or \
            width * height > MAX_PIXELS:
        raise ValueError("size: multiples of 16, each side 256-1280, at most 1280x720 pixels in all")
    try:
        seconds = float(body.get("seconds", 2))
    except (TypeError, ValueError):
        raise ValueError("seconds must be a number") from None
    if not math.isfinite(seconds) or not 0 < seconds <= MAX_SECONDS:
        raise ValueError(f"seconds must be more than 0 and at most {MAX_SECONDS}")
    frames = int(round(seconds * FPS / 4)) * 4 + 1      # Wan's VAE: 4n + 1 frames
    out = {"prompt": prompt, "width": width, "height": height, "frames": max(frames, 5), "seconds": seconds}
    for k, lo, hi in (("steps", 1, 60), ("high_noise_steps", 1, 60)):
        v = body.get(k, defaults[k])
        if not isinstance(v, int) or isinstance(v, bool) or not lo <= v <= hi:
            raise ValueError(f"{k} must be an integer from {lo} to {hi}")
        out[k] = v
    g = body.get("guidance_scale", defaults["guidance_scale"])
    if not isinstance(g, (int, float)) or isinstance(g, bool) or not 1 <= g <= 20:
        raise ValueError("guidance_scale must be a number from 1 to 20")
    seed = body.get("seed", int(time.time() * 1000) % 2**31)
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**31:
        raise ValueError("seed must be an integer from 0 to 2^31-1")
    negative = body.get("negative_prompt", "")
    if not isinstance(negative, str) or len(negative) > MAX_PROMPT_CHARS:
        raise ValueError(f"negative_prompt must be a string of at most {MAX_PROMPT_CHARS} characters")
    for text in (prompt, negative):
        try:
            text.encode("utf-8")
        except UnicodeEncodeError:
            raise ValueError("prompt and negative_prompt must be valid Unicode text") from None
    out.update(cfg=float(g), seed=seed, negative=negative)
    return out


class Job:
    def __init__(self, p: dict):
        self.id = "video_" + uuid.uuid4().hex
        self.p = p
        self.status, self.progress, self.error = "queued", 0, None
        self.created_at, self.completed_at = int(time.time()), None
        self.cancel = threading.Event()
        self.proc = None                                  # sd-cli while it runs (shutdown() stops it)
        self.dir = Path(tempfile.mkdtemp(prefix="strata-video-"))   # 0700: this user's only
        self.file = self.dir / "out.webm"                    # out.mp4 once converted (to_mp4)

    def view(self, model, keep_s) -> dict:
        return {"id": self.id, "object": "video", "model": model, "status": self.status, "progress": self.progress,
                "created_at": self.created_at, "completed_at": self.completed_at,
                "expires_at": self.completed_at + int(keep_s) if self.completed_at and self.status == "completed" else None,
                "prompt": self.p["prompt"], "size": f"{self.p['width']}x{self.p['height']}",
                "seconds": f"{(self.p['frames'] - 1) / FPS:g}", "remixed_from_video_id": None,
                "error": {"code": "video_failed", "message": self.error} if self.error else None}


class VideoGen:
    MODEL = "wan2.2-t2v-a14b"

    def __init__(self, exe, low, high, t5, vae, need_gb=40.0, timeout_s=3600.0, keep_s=3600.0, queue_s=3600.0,
                 args=(), loras=None):
        self.exe, self.low, self.high, self.t5, self.vae = exe, low, high, t5, vae
        self.loras = loras                                  # {"dir", "low", "high"} or None
        self.defaults = LIGHTNING if loras else PLAIN
        self.need_gb, self.timeout_s, self.keep_s, self.queue_s = need_gb, timeout_s, keep_s, queue_s
        self.args = list(args)
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        self.available_gb = ImageGen.available_gb          # make_room's memory reading
        self.ffmpeg = shutil.which("ffmpeg")
        self.mp4 = None                                     # the encoder's ffmpeg arguments, chosen at the first clip

    @staticmethod
    def from_config(cfg):
        if cfg is None:
            return None
        if not isinstance(cfg, dict):
            raise ValueError('"video" must be {"exe": ..., "low": ..., "high": ..., "t5": ..., "vae": ...}')
        files = {}
        for k in ("exe", "low", "high", "t5", "vae"):
            v = cfg.get(k)
            if not isinstance(v, str) or not Path(v).expanduser().is_file():
                raise ValueError(f'"video".{k} must name an existing file')
            files[k] = str(Path(v).expanduser().resolve())
        nums = {}
        for k, default in (("need_gb", 40.0), ("timeout_s", 3600.0), ("keep_s", 3600.0), ("queue_s", 3600.0)):
            v = cfg.get(k, default)
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v <= 0:
                raise ValueError(f'"video".{k} must be a positive number')
            nums[k] = float(v)
        args = cfg.get("args", ["--diffusion-fa"])
        if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
            raise ValueError('"video".args must be a list of strings')
        loras = cfg.get("loras")
        if loras is not None:
            if not isinstance(loras, dict) or not isinstance(loras.get("dir"), str) or \
                    not Path(loras["dir"]).expanduser().is_dir():
                raise ValueError('"video".loras must be {"dir": <a folder>, "low": <name>, "high": <name>}')
            d = Path(loras["dir"]).expanduser().resolve()
            for k in ("low", "high"):
                n = loras.get(k)
                if not isinstance(n, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", n) or \
                        not (d / f"{n}.safetensors").is_file():
                    raise ValueError(f'"video".loras.{k} must name a .safetensors file in {d} (without ".safetensors")')
            loras = {"dir": str(d), "low": loras["low"], "high": loras["high"]}
        return VideoGen(**files, **nums, args=args, loras=loras)

    # ------------------------------------------------------------------ the jobs
    def sweep(self):
        """Forget expired clips (their files go too)."""
        now = time.time()
        with self.lock:
            for jid in [j.id for j in self.jobs.values()
                        if j.completed_at and now - j.completed_at > self.keep_s]:
                shutil.rmtree(self.jobs.pop(jid).dir, ignore_errors=True)

    def create(self, svc, req):
        """(status, body): 202 with the job, or 400 / 429."""
        p = parse_request(req, self.defaults)              # ValueError: the server's 400
        self.sweep()
        with self.lock:
            if len(self.jobs) >= MAX_JOBS:
                return 429, {"error": {"type": "rate_limit_error", "message": f"at most {MAX_JOBS} videos are kept: "
                                       "DELETE /v1/videos/{id} the finished ones, or wait for them to expire"}}
            job = Job(p)
            self.jobs[job.id] = job
        view = job.view(self.MODEL, self.keep_s)            # as created: queued (the thread may start it at once)
        try:
            threading.Thread(target=self.run, args=(svc, job), daemon=True, name=job.id).start()
        except RuntimeError as e:                           # no thread: no job either
            with self.lock:
                self.jobs.pop(job.id, None)
            shutil.rmtree(job.dir, ignore_errors=True)
            return 503, {"error": {"type": "server_error", "message": f"the video could not start ({e})"}}
        return 202, view

    def get(self, jid):
        self.sweep()
        with self.lock:
            return self.jobs.get(jid)

    def list(self):
        self.sweep()
        with self.lock:
            return [j.view(self.MODEL, self.keep_s) for j in sorted(self.jobs.values(), key=lambda j: j.created_at)]

    def delete(self, jid) -> bool:
        """Cancel it when queued or running (its thread removes the files once the process has exited), else remove."""
        with self.lock:
            job = self.jobs.pop(jid, None)
        if job is None:
            return False
        job.cancel.set()
        if job.status in ("completed", "failed"):
            shutil.rmtree(job.dir, ignore_errors=True)
        return True

    def shutdown(self):
        """At the server's exit: every job cancelled, its sd-cli stopped (its own session: it would outlive the
        server), its folder removed."""
        with self.lock:
            jobs = list(self.jobs.values())
            self.jobs.clear()
        for job in jobs:
            job.cancel.set()
            if job.proc is not None and job.proc.poll() is None:
                _stop(job.proc)
            shutil.rmtree(job.dir, ignore_errors=True)

    def run(self, svc, job: Job):
        """The job's thread.  Any error ends the job as failed, its turn given back and its folder cleaned."""
        def fail(msg):
            job.status, job.error, job.completed_at = "failed", msg, int(time.time())
            print(f"[strata] video {job.id} failed: {msg}", flush=True)
        try:
            self._run(svc, job, fail)
        except Exception as e:  # noqa: BLE001 - the job must end, whatever went wrong
            fail(f"internal error: {e}")
        finally:
            if job.status in ("queued", "in_progress"):
                fail("ended without a result")
            self._cleanup(job)

    def _run(self, svc, job: Job, fail):
        """The server's turn, room in memory, sd-cli, then the turn back."""
        deadline = time.monotonic() + self.queue_s
        got = False
        while not got and not job.cancel.is_set() and time.monotonic() < deadline:
            got = svc.fifo.acquire(timeout=1.0)
        if not got:
            fail("cancelled before it started" if job.cancel.is_set() else f"no turn within {self.queue_s:.0f} s")
            return
        svc.image_gate.set()                                 # chat requests wait while the clip runs
        try:
            room = make_room(svc, self, time.monotonic() + 600, job.cancel)
            if room == "cancelled":
                fail("cancelled before it started")
            elif room not in ("enough", "unloaded"):
                fail(f"memory: {room} ({self.available_gb()} GB free, {self.need_gb:.0f} GB needed)")
            else:
                job.status, job.progress = "in_progress", 1
                self.render(job, fail)
        finally:
            svc.image_gate.clear()
            svc.fifo.release()

    def _cleanup(self, job):
        with self.lock:
            gone = job.id not in self.jobs                    # deleted while it ran: nothing to keep
        if gone or job.status == "failed":
            shutil.rmtree(job.dir, ignore_errors=True)

    def to_mp4(self, job: Job):
        """The clip as H.264 MP4 (QuickTime plays it); on any failure the webm stays, and the log says why."""
        if not self.ffmpeg:
            return
        if self.mp4 is None:
            try:
                enc = subprocess.run([self.ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True,
                                     timeout=20).stdout
            except (OSError, subprocess.TimeoutExpired):
                enc = ""
            self.mp4 = next((MP4[e] for e in MP4 if f" {e} " in enc), [])
        if not self.mp4:
            return
        out = job.dir / "out.mp4"
        try:
            r = subprocess.run([self.ffmpeg, "-nostdin", "-v", "error", "-y", "-i", str(job.file), *self.mp4,
                                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
                               capture_output=True, text=True, timeout=300)
        except (OSError, subprocess.TimeoutExpired) as e:
            print(f"[strata] video {job.id}: kept as webm ({e})", flush=True)
            return
        if r.returncode == 0 and out.is_file() and out.stat().st_size > 0:
            job.file.unlink(missing_ok=True)
            job.file = out
        else:
            print(f"[strata] video {job.id}: kept as webm (ffmpeg: {r.stderr.strip()[-200:]})", flush=True)

    def command(self, job: Job) -> list:
        p, d = job.p, job.dir
        lora = []
        prompt = p["prompt"]
        if self.loras:   # the low-noise expert's LoRA, the high-noise one's: the checked files (sd-cli would take a .gguf first)
            prompt += f"<lora:{self.loras['low']}.safetensors:1><lora:|high_noise|{self.loras['high']}.safetensors:1>"
            lora = ["--lora-model-dir", self.loras["dir"]]
        (d / "prompt.txt").write_text(prompt, encoding="utf-8")
        (d / "negative.txt").write_text(p["negative"], encoding="utf-8")
        return [self.exe, *lora, "-M", "vid_gen", "--diffusion-model", self.low, "--high-noise-diffusion-model", self.high,
                "--t5xxl", self.t5, "--vae", self.vae, "--prompt-file", str(d / "prompt.txt"),
                "--negative-prompt-file", str(d / "negative.txt"), "--cfg-scale", repr(p["cfg"]),
                "--high-noise-cfg-scale", repr(p["cfg"]), "--sampling-method", "euler",
                "--high-noise-sampling-method", "euler", "--steps", str(p["steps"]),
                "--high-noise-steps", str(p["high_noise_steps"]), "-W", str(p["width"]), "-H", str(p["height"]),
                "--video-frames", str(p["frames"]), "--fps", str(FPS), "--seed", str(p["seed"]),
                "--disable-image-metadata", *self.args, "-o", str(job.file)]

    def render(self, job: Job, fail):
        total = job.p["steps"] + job.p["high_noise_steps"]
        t0 = time.monotonic()
        with open(job.dir / "log.txt", "wb") as log:
            try:
                proc = subprocess.Popen(self.command(job), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, start_new_session=True)
            except OSError as e:
                return fail(f"sd-cli could not start ({e})")
            job.proc = proc
            done = 0                                          # sampling steps finished, over both experts
            with proc:
                def read():
                    nonlocal done
                    last = None
                    buf = b""
                    for chunk in iter(lambda: proc.stdout.read1(4096), b""):
                        log.write(chunk)
                        buf += chunk
                        *lines, buf = re.split(rb"[\r\n]", buf)
                        for line in lines:
                            m = STEP.search(line.decode("utf-8", "replace"))
                            if m:
                                k, n = int(m.group(1)), int(m.group(2))
                                if last is not None and k < last:      # the second expert's bar started
                                    done += last
                                last = k
                                job.progress = min(95, 2 + int(90 * (done + k) / total))
                reader = threading.Thread(target=read, daemon=True)
                reader.start()
                why = None
                try:
                    while proc.poll() is None:
                        if job.cancel.is_set() or time.monotonic() - t0 >= self.timeout_s:
                            why = "cancelled" if job.cancel.is_set() else f"longer than {self.timeout_s:.0f} s"
                            break
                        job.cancel.wait(0.5)
                finally:
                    if proc.poll() is None:                   # cancelled, past the deadline, or an error here
                        _stop(proc)
                    proc.wait()
                reader.join(5)
        if why:
            return fail(why)
        if self.loras and b"can not found lora" in (job.dir / "log.txt").read_bytes():
            return fail("a LoRA file is missing: sd-cli ran without it (a 4-step clip without Lightning is noise)")
        if proc.returncode != 0 or not job.file.is_file():
            tail = (job.dir / "log.txt").read_bytes()[-400:].decode("utf-8", "replace").replace("\r", "\n").strip()
            return fail(f"sd-cli ended with status {proc.returncode}: {tail[-300:]}")
        self.to_mp4(job)
        job.status, job.progress, job.completed_at = "completed", 100, int(time.time())
        print(f"[strata] video {job.id}: {job.p['width']}x{job.p['height']}, {job.p['frames']} frames, "
              f"{time.monotonic() - t0:.0f} s", flush=True)
