"""Pictures from a prompt: POST /v1/images/generations, made by strata-image (metal/image: Nucleus-Image on Strata's own
ggml/Metal engine) with the run config's "image":

    "image": {"dit": ".../nucleus-dit-q4_0.gguf", "encoder": ".../qwen3vl-8b-text-bf16.gguf",
              "vae": ".../nucleus-vae.gguf", "need_gb": 16, "queue_s": 600, "timeout_s": 1800}

("exe" names the binary when it is not engine-metal/strata-image, which setup builds, or build-metal/metal/.)  strata-image is started for one picture and
exits after it, so a crash or an out-of-memory there does not take the chat model with it, and its exit is the proof that
its memory is back.  This server is the gate; its rule is that the chat model and the picture never both run (or sit) in
memory that cannot hold them:

1. the request is checked here first (parse_request): a bad one is a 400 while the chat model is still loaded;
2. the picture takes the server's turn (Service.fifo), as a chat request does, and closes the door to parallel chat
   requests (Service.image_gate) while it runs; no turn within queue_s: 503 with Retry-After;
3. with the turn, when psutil's available memory is under need_gb, the chat model is unloaded after the requests running
   on it have ended, and the memory is looked at again: still short (or the model could not be unloaded) is a 503;
4. strata-image gets the prompts as files in a private temporary folder (not its command line, which every user of the
   Mac can read) and writes the PNG there; a client that leaves (or POST /cancel) has it stopped (SIGTERM, then SIGKILL)
   and the answer is 499; past timeout_s (an absolute deadline) the same, and 504;
5. the turn is kept until the process has exited.
"""
import base64
import json
import math
import os
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_PROMPT_CHARS = 4000         # strata-image takes 16384 tokens: Qwen's byte-level BPE is at most 4 a character
MAX_PIXELS = 1536 * 1024        # 1024x1024: 1.5 GB of activations, 3 s a step; 2048x2048: 12.9 GB, 40 s
KILL_AFTER_S = 5.0               # SIGTERM, then SIGKILL this much later


def parse_request(body) -> dict:
    """OpenAI's images request (prompt, n, size, response_format) plus Nucleus' own steps, guidance_scale, seed and
    negative_prompt.  A bad value is a ValueError: a 400 before anything is loaded."""
    if not isinstance(body, dict):
        raise ValueError("the body must be a JSON object")
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise ValueError(f"prompt is longer than {MAX_PROMPT_CHARS} characters")
    if body.get("n", 1) != 1:
        raise ValueError("n must be 1 (one picture at a time)")
    if body.get("response_format", "b64_json") != "b64_json":
        raise ValueError("response_format must be b64_json (this server keeps no files to link to)")
    try:
        width, height = (int(v) for v in str(body.get("size", "1024x1024")).lower().split("x"))
    except ValueError:
        raise ValueError('size must look like "1024x1024"') from None
    if width % 16 or height % 16 or not 256 <= min(width, height) <= max(width, height) <= 2048:
        raise ValueError("size: width and height must be multiples of 16 from 256 to 2048")
    if width * height > MAX_PIXELS:          # the expert-choice routing's memory grows with the square of the pixels
        raise ValueError("size: at most 1536x1024 pixels in all (1024x1024, 1536x1024, 1248x1248 ...)")
    steps, cfg, seed = body.get("steps", 30), body.get("guidance_scale", 4.0), body.get("seed")
    if not isinstance(steps, int) or isinstance(steps, bool) or not 1 <= steps <= 100:
        raise ValueError("steps must be an integer from 1 to 100")
    if not isinstance(cfg, (int, float)) or isinstance(cfg, bool) or not 0 <= cfg <= 20:
        raise ValueError("guidance_scale must be a number from 0 to 20")
    if seed is None:
        seed = int(time.time() * 1000) % 2**31
    elif not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**31:
        raise ValueError("seed must be an integer from 0 to 2^31-1")
    negative = body.get("negative_prompt", "")
    if not isinstance(negative, str) or len(negative) > MAX_PROMPT_CHARS:
        raise ValueError(f"negative_prompt must be a string of at most {MAX_PROMPT_CHARS} characters")
    for text in (prompt, negative):          # JSON allows a lone surrogate ("\ud800"); the prompt files are UTF-8
        try:
            text.encode("utf-8")
        except UnicodeEncodeError:
            raise ValueError("prompt and negative_prompt must be valid Unicode text") from None
    return {"prompt": prompt, "negative": negative, "width": width, "height": height, "steps": steps,
            "cfg": float(cfg), "seed": seed}


def _error(msg: str, kind: str = "server_error") -> bytes:
    return json.dumps({"error": {"type": kind, "message": msg}}).encode()


class ImageGen:
    def __init__(self, exe, dit, encoder, vae, need_gb: float = 16.0, queue_s: float = 600.0,
                 timeout_s: float = 1800.0):
        self.exe, self.dit, self.encoder, self.vae = str(exe), str(dit), str(encoder), str(vae)
        self.need_gb, self.queue_s, self.timeout_s = need_gb, queue_s, timeout_s

    @staticmethod
    def from_config(cfg):
        """The config's "image" entry, or None (no picture generation).  A bad entry is a ValueError at start."""
        if cfg is None:
            return None
        if not isinstance(cfg, dict):
            raise ValueError('"image" must be {"dit": ..., "encoder": ..., "vae": ...} (docs/MACOS.md)')
        files = {}
        for k in ("dit", "encoder", "vae", "exe"):
            v = cfg.get(k, next((str(p) for p in (ROOT / "engine-metal/strata-image", ROOT / "build-metal/metal/strata-image")
                                 if p.is_file()), "strata-image") if k == "exe" else None)
            if not isinstance(v, str) or not Path(v).expanduser().is_file():
                raise ValueError(f'"image".{k} must name an existing file' +
                                 (" (run setup, or: cmake --build build-metal --target strata-image)" if k == "exe" else ""))
            files[k] = str(Path(v).expanduser().resolve())
        nums = {}
        for k, default in (("need_gb", 16.0), ("queue_s", 600.0), ("timeout_s", 1800.0)):
            v = cfg.get(k, default)
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v <= 0:
                raise ValueError(f'"image".{k} must be a positive number')
            nums[k] = float(v)
        return ImageGen(**files, **nums)

    @staticmethod
    def available_gb():
        """The memory macOS can give now (GB), or None when it cannot be read."""
        try:
            import psutil
            return psutil.virtual_memory().available / 1e9
        except Exception:  # noqa: BLE001 - psutil missing or failing
            return None

    def validate(self, req):
        """None when `req` is a picture this server can make; else (400, body)."""
        try:
            parse_request(req)
        except ValueError as e:
            return 400, _error(str(e), "invalid_request_error")
        return None

    def forward(self, req: dict, cancel: threading.Event):
        """The picture: (status, headers, body), after strata-image has exited (stopped when `cancel` is set: 499, or
        at the deadline: 504)."""
        p = parse_request(req)
        with tempfile.TemporaryDirectory(prefix="strata-image-") as d:      # 0700: this user's only
            d = Path(d)
            (d / "prompt.txt").write_text(p["prompt"], encoding="utf-8")
            (d / "negative.txt").write_text(p["negative"], encoding="utf-8")
            args = [self.exe, "--encoder", self.encoder, "--dit", self.dit, "--vae", self.vae,
                    "--prompt-file", str(d / "prompt.txt"), "--negative-file", str(d / "negative.txt"),
                    "--width", str(p["width"]), "--height", str(p["height"]), "--steps", str(p["steps"]),
                    "--cfg", repr(p["cfg"]), "--seed", str(p["seed"]), "--out", str(d / "out.png"), "--watch-stdin", "1"]
            with open(d / "stderr.txt", "wb") as err:
                try:
                    # stdin stays open and unwritten: strata-image ends itself when it closes (this server is gone)
                    proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err,
                                            start_new_session=True)
                except OSError as e:
                    return 500, {}, _error(f"strata-image could not start ({e})")
                with proc:                                  # closes its pipes and waits for it at the end
                    lines = []
                    reader = threading.Thread(target=lambda: lines.extend(_events(proc.stdout)), daemon=True)
                    reader.start()
                    deadline = time.monotonic() + self.timeout_s
                    why = None
                    while proc.poll() is None:
                        if cancel.is_set() or time.monotonic() >= deadline:
                            why = "cancelled" if cancel.is_set() else "deadline"
                            _stop(proc)
                            break
                        cancel.wait(0.25)
                    proc.wait()
                    reader.join(5)
            if why:                                           # a job's end is logged, not only its start
                print(f"[strata] picture stopped: {'cancelled' if why == 'cancelled' else 'past timeout_s'}", flush=True)
            if why == "cancelled":
                return 499, {}, _error("the picture was cancelled", "cancelled")
            if why == "deadline":
                return 504, {}, _error(f"strata-image took longer than {self.timeout_s:.0f} s")
            done = next((e for e in lines if e.get("done")), None)
            if proc.returncode != 0 or done is None or not (d / "out.png").is_file():
                msg = next((e["error"] for e in lines if isinstance(e.get("error"), str)), None)
                if msg is None:
                    tail = (d / "stderr.txt").read_bytes()[-600:].decode("utf-8", "replace").strip()
                    msg = f"strata-image ended with status {proc.returncode}" + (f": {tail}" if tail else "")
                print(f"[strata] picture failed: {msg}", flush=True)
                return 500, {}, _error(msg)
            png = (d / "out.png").read_bytes()
        info = {k: v for k, v in done.items() if k != "done"}
        info.update(steps=p["steps"], seed=p["seed"])
        print(f"[strata] picture {p['width']}x{p['height']}, {p['steps']} steps: {json.dumps(info)}", flush=True)
        return 200, {}, json.dumps({"created": int(time.time()), "data": [{"b64_json": base64.b64encode(png).decode()}],
                                    "strata": info}).encode()


def _events(stream):
    """strata-image's stdout: one JSON object a line (anything else is skipped)."""
    for raw in stream:
        try:
            e = json.loads(raw)
        except ValueError:
            continue
        if isinstance(e, dict):
            yield e


def _stop(proc):
    """SIGTERM to strata-image's process group, SIGKILL when it has not exited KILL_AFTER_S later."""
    for sig, wait in ((signal.SIGTERM, KILL_AFTER_S), (signal.SIGKILL, None)):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            return
        try:
            proc.wait(wait)
            return
        except subprocess.TimeoutExpired:
            continue


def make_room(svc, gen: ImageGen, deadline: float, cancel: threading.Event) -> str:
    """With svc.fifo held and svc.image_gate set: room for a picture.  "enough" (memory was), "unloaded" (the chat
    model was unloaded and the memory is there now), "cancelled", or a refusal: "busy" (the chat model's running
    requests had not ended by the deadline), "short" (memory still short), "unsupported" (it cannot be unloaded)."""
    avail = gen.available_gb()
    if avail is not None and avail >= gen.need_gb:
        return "enough"
    r = "busy"
    while not cancel.is_set():
        r = svc.unload_held(waiting_ok=True)
        if r != "busy" or time.monotonic() >= deadline:
            break
        time.sleep(0.5)
    if cancel.is_set():
        return "cancelled"
    if r in ("busy", "unsupported"):
        return r
    # unloaded (or not loaded at all): the memory takes a moment to come back - then it must be enough
    until = min(deadline, time.monotonic() + 15)
    while True:
        avail = gen.available_gb()
        if avail is None or avail >= gen.need_gb:
            return "unloaded" if r == "unloaded" else "enough"
        if time.monotonic() >= until or cancel.is_set():
            return "short"
        time.sleep(0.5)
