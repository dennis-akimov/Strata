"""metal/onboard_mac.py - `make setup` on a Mac: everything from a fresh macOS to a chat that answers.

bootstrap.py has already made sure of the Command Line Tools, Python 3.10+ and .venv (it runs this with .venv's
Python).
From there, in order, each step skipped when it is already done:

  1. this Mac: chip, memory, what Metal lets the GPU use, free disk
  2. the rest of the tools: the Python packages (requirements.txt: cmake, ninja, ...), Rust (rustup) for the server
     and the app, Homebrew and aria2 (downloads with 16 connections; without them tools/pull_model.py downloads alone)
  3. builds: the Metal engine, the server and its file tool, Strata.app; each put where the run config expects it
  4. the model: every chat model of the README with its download and whether it fits this Mac; the best one that fits
     is the default
  5. its download (resumable) and tokenizer pack, then the run config strata-<model>.json, with the context and the
     prefill batch the engine's own measurement (strata-metal --plan) says fit
  6. a first answer (a test start on a free port): if the GPU runs out of memory, a smaller context, then the reason
  7. Strata.app, opened on that model

    python3 bootstrap.py --onboard [--model GPT-OSS-20B] [--yes] [--data-dir DIR] [--no-open]
    .venv/bin/python metal/onboard_mac.py --config-only GPT-OSS-20B     (make pull: the run config of a pulled model)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENGINE = ROOT / "engine-metal"
GB = 1e9
GIB = 2**30

# The chat models of the README.  mem_full: memory at their full context (README "In memory", measured or
# estimated); mem_min: at the smallest context (about the weights + 1 GB).  how: "pull" = tools/pull_model.py, then
# the run config written here; "setup" = setup.py's own installer (it writes its run config).
MODELS = {
    "GPT-OSS-20B": dict(title="GPT-OSS 20B", about="the fastest; good for 16-24 GB Macs", download=12.1, mem_min=12.0,
                        mem_full=19, how="pull", gguf="gpt-oss-20b/gpt-oss-20b-MXFP4.gguf", pack="gpt-oss-20b",
                        name="gpt-oss-20b", context=131072, sampling={"temperature": 1.0, "top_p": 1.0}),
    "QWEN3.6-35B": dict(title="Qwen3.6-35B-A3B", about="35B, about 3B active per token", download=21.7, mem_min=22.5,
                        mem_full=29, how="pull", gguf="Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-MXFP4_MOE.gguf",
                        pack="qwen3.6-35b-a3b", name="qwen3.6-35b-a3b", context=131072,
                        sampling={"temperature": 0.6, "top_p": 0.95, "top_k": 20}),
    "Q2_0": dict(title="Qwen3.8-Flash-Next Q2_0", about="the default and the most tested on a Mac", download=66.4,
                 mem_min=52, mem_full=60, how="setup"),
    "GPT-OSS-120B": dict(title="GPT-OSS 120B", about="about 5B active per token", download=63.4, mem_min=62,
                         mem_full=70, how="pull", gguf="gpt-oss-120b/gpt-oss-120b-MXFP4.gguf", pack="gpt-oss-120b",
                         name="gpt-oss-120b", context=131072, sampling={"temperature": 1.0, "top_p": 1.0}),
    "IQ3_XXS": dict(title="Qwen3.8-Flash-Next IQ3_XXS", about="less compressed than Q2_0", download=75.8, mem_min=60,
                    mem_full=65, how="setup"),
    "IQ3_S": dict(title="Qwen3.8-Flash-Next IQ3_S", about="less compressed again", download=83.6, mem_min=68,
                  mem_full=74, how="setup"),
    "GLM-5.3": dict(title="GLM-5.3-Flash (Maya-S24)", about="the largest (320B), the slowest", download=94.7,
                    mem_min=88, mem_full=89, how="pull",
                    gguf="GLM-5.3-Flash-Maya-S24/GLM-5.3-Flash-Maya-S24-IQ2_XXS_S-00001-of-00003.gguf",
                    pack="glm-5.3-flash-maya-s24", name="glm-5.3-flash-maya-s24", context=32768, sampling={}),
}

GREEN, AMBER, RED, BOLD, DIM, END = ("\033[32m", "\033[33m", "\033[31m", "\033[1m", "\033[2m", "\033[0m") \
    if sys.stdout.isatty() else ("",) * 6


def step(n: int, total: int, what: str) -> None:
    print(f"\n{BOLD}[{n}/{total}] {what}{END}", flush=True)


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{END} {msg}", flush=True)


def note(msg: str) -> None:
    print(f"  {DIM}{msg}{END}", flush=True)


def warn(msg: str) -> None:
    print(f"  {AMBER}!{END} {msg}", flush=True)


def fail(msg: str, fix: str = "") -> None:
    print(f"\n{RED}✗ {msg}{END}" + (f"\n  {fix}" if fix else ""), flush=True)
    sys.exit(1)


def run(cmd, what: str = "", quiet: bool = False, **kw) -> None:
    """A command whose output the user may want to see (builds, downloads); stops onboarding when it fails."""
    if what:
        note(what)
    out = subprocess.DEVNULL if quiet else None
    r = subprocess.run(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT if quiet else None, **kw)
    if r.returncode != 0:
        fail(f"{' '.join(map(str, cmd))} failed (exit {r.returncode})", "Run `make setup` again after fixing it: finished steps are skipped.")


def ask(question: str, default: str, yes: bool) -> str:
    if yes or not sys.stdin.isatty():
        return default
    a = input(f"  {question} [{default}] ").strip()
    return a or default


def sysctl(name: str) -> str:
    return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True).stdout.strip()


def metal_limit() -> float:
    """Metal's recommendedMaxWorkingSetSize (bytes): what the GPU may use of the unified memory."""
    try:
        import ctypes
        objc = ctypes.cdll.LoadLibrary("/usr/lib/libobjc.dylib")
        metal = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/Metal.framework/Metal")
        metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
        objc.sel_registerName.restype = ctypes.c_void_p
        dev = metal.MTLCreateSystemDefaultDevice()
        send = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p)(("objc_msgSend", objc))
        return float(send(dev, objc.sel_registerName(b"recommendedMaxWorkingSetSize"))) if dev else 0.0
    except (OSError, AttributeError):
        return 0.0


def free_now() -> float:
    """Memory free or reclaimable right now (bytes): free + inactive + speculative + purgeable pages."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(re.search(r"page size of (\d+)", out).group(1)) if "page size" in out else 16384
    pages = {m[0]: int(m[1]) for m in re.findall(r"Pages (free|inactive|speculative|purgeable):\s+(\d+)", out)}
    return float(sum(pages.values()) * page)


def mac() -> dict:
    ram = float(sysctl("hw.memsize") or 0)
    limit = metal_limit() or 0.75 * ram
    return {"chip": sysctl("machdep.cpu.brand_string") or "Apple Silicon", "ram": ram, "limit": limit,
            "usable": min(limit, ram - max(3 * GB, 0.15 * ram)), "macos": platform.mac_ver()[0]}


def verdict(m: dict, me: dict) -> tuple[str, str]:
    usable = me["usable"] / GB
    if m["mem_full"] <= usable:
        return "fits", GREEN
    if m["mem_min"] <= usable:
        return "fits with a smaller context", AMBER
    return "too big for this Mac", RED


# ------------------------------------------------------------------ tools

def path_with_cargo() -> dict:
    env = dict(os.environ)
    cargo_bin = Path.home() / ".cargo" / "bin"
    if str(cargo_bin) not in env.get("PATH", ""):
        env["PATH"] = f"{cargo_bin}{os.pathsep}{env.get('PATH', '')}"
    return env


def tools(yes: bool) -> dict:
    env = path_with_cargo()
    # Python packages (cmake and ninja for the engine, the tokenizer pack's needs)
    if subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", "-r",
                       str(ROOT / "requirements.txt")], cwd=ROOT).returncode != 0:
        fail("pip could not install requirements.txt", "Check the internet connection, then run `make setup` again.")
    ok("Python packages (cmake, ninja, ...) in .venv")
    # Rust: the server, its file tool and the app
    if not shutil.which("cargo", path=env["PATH"]):
        note("Rust is missing: installing it with rustup (https://rustup.rs; into ~/.cargo, no sudo) ...")
        r = subprocess.run("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal",
                           shell=True)
        if r.returncode != 0 or not shutil.which("cargo", path=env["PATH"]):
            fail("rustup could not install Rust", "Install it by hand: https://rustup.rs, then run `make setup` again.")
    v = subprocess.run(["cargo", "--version"], capture_output=True, text=True, env=env).stdout.strip()
    ok(f"Rust ({v})")
    # aria2: 16 connections a file, from Homebrew (tools/pull_model.py downloads alone without it: one connection)
    brew = brew_path()
    if not shutil.which("aria2c") and not brew:
        if ask("Install Homebrew (https://brew.sh) for aria2, faster downloads? It asks for your password (y/n)",
               "y" if sys.stdin.isatty() else "n", yes).lower() == "y":
            note("Homebrew's installer says what it does, then asks for your password ...")
            script = subprocess.run(["curl", "-fsSL", "https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh"],
                                    capture_output=True, text=True).stdout
            if script and subprocess.run(["/bin/bash", "-c", script]).returncode == 0:
                brew = brew_path()
    if brew:
        ok("Homebrew")
        env["PATH"] = f"{Path(brew).parent}{os.pathsep}{env['PATH']}"
    if shutil.which("aria2c", path=env["PATH"]):
        ok("aria2 (downloads)")
    elif brew:
        run([brew, "install", "aria2"], "installing aria2 with Homebrew (faster downloads) ...", quiet=True)
        ok("aria2 (downloads)")
    else:
        warn("no aria2: downloads use one connection (slower, still resumable and checked)")
    return env


def brew_path() -> str | None:
    for b in ("/opt/homebrew/bin/brew", "/usr/local/bin/brew"):
        if os.access(b, os.X_OK):
            return b
    return shutil.which("brew")


def builds(env: dict) -> None:
    # make build: the Metal engine (strata-metal, strata-vision) into engine-metal/, after the server
    if not (ENGINE / "strata-metal").exists():
        note("building the Metal engine (once, 3-6 minutes) ...")
    run(["make", "--no-print-directory", "build"], env=env, quiet=False)
    ok("the Metal engine, the server and strata-files in engine-metal/")
    run(["make", "--no-print-directory", "app"], env=env, quiet=True, what="building Strata.app ...")
    ok("Strata.app")


# ------------------------------------------------------------------ the model

def choose(me: dict, wanted: str | None, yes: bool) -> str:
    if wanted:
        key = wanted.upper()
        if key not in MODELS:
            fail(f"unknown model {wanted}", "One of: " + ", ".join(MODELS))
        return key
    fitting = [k for k, m in MODELS.items() if verdict(m, me)[0] != "too big for this Mac"]
    # the default: the README's order of recommendation, the first that fits whole, else the first that fits at all
    prefer = ["Q2_0", "GPT-OSS-120B", "QWEN3.6-35B", "GPT-OSS-20B"]
    whole = [k for k in prefer if verdict(MODELS[k], me)[0] == "fits"]
    best = (whole or [k for k in prefer if k in fitting] or ["GPT-OSS-20B"])[0]
    print(f"  {'':3}{'model':32}{'download':>10}{'memory':>12}   for this Mac")
    keys = list(MODELS)
    for i, k in enumerate(keys, 1):
        m = MODELS[k]
        v, colour = verdict(m, me)
        mark = "▸" if k == best else " "
        print(f"  {mark}{i:>2} {m['title']:32}{m['download']:>8.1f} GB{m['mem_min']:>6.0f}-{m['mem_full']:<3.0f}GB   "
              f"{colour}{v}{END}  {DIM}{m['about']}{END}" + (f"  {BOLD}recommended{END}" if k == best else ""))
    if not fitting:
        warn(f"none fits this Mac's {me['ram'] / GB:.0f} GB well; the smallest, GPT-OSS 20B, needs about 12 GB free")
    a = ask(f"Which model? (1-{len(keys)})", str(keys.index(best) + 1), yes)
    try:
        key = keys[int(a) - 1]
    except (ValueError, IndexError):
        key = a.upper() if a.upper() in MODELS else best
    v, _ = verdict(MODELS[key], me)
    if v == "too big for this Mac" and ask(f"{MODELS[key]['title']} is too big for this Mac. Continue anyway? (y/n)",
                                           "n", yes).lower() != "y":
        return choose(me, None, yes)
    return key


def disk_free(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    return float(shutil.disk_usage(path).free)


def plan(gguf: Path, ctx: int, pb: int) -> dict:
    """The engine's own measurement of what the model needs (no weights read): see metal/strata_metal.cpp --plan."""
    r = subprocess.run([str(ENGINE / "strata-metal"), "--plan", "--gguf", str(gguf), "--max-context", str(ctx),
                        "--kv", "f16", "--prefill-batch", str(pb)], capture_output=True, text=True, timeout=300)
    line = next((l for l in r.stdout.splitlines() if l.startswith("{")), "")
    if not line:
        fail("the engine could not measure the model", (r.stderr or r.stdout).strip()[-800:])
    return json.loads(line)


def need(p: dict, ctx: int) -> tuple[float, float]:
    """(GPU bytes, all bytes) at a context of ctx tokens, from one plan (linear in the context)."""
    g, h = p["gpu"], p["host"]
    gpu = g["model"] + g["context"] + g["compute"] + p["per_token"] * (ctx - p["context"])
    return gpu, gpu + h["model"] + h["context"] + h["compute"]


def size_context(gguf: Path, want: int, me: dict) -> tuple[int, int, float]:
    """The largest context up to `want` that fits, with the prefill batch that allows it: (context, batch, bytes)."""
    best = (0, 2048, 0.0)
    for pb in (2048, 512, 256):
        p = plan(gguf, want, pb)
        want = min(want, p["n_ctx_train"] or want)
        lo, hi = 0, want
        for c in sorted({4096, 8192, 16384, 32768, 65536, 131072, 262144, want}):
            if c > want:
                break
            gpu, total = need(p, c)
            if gpu <= me["limit"] and total <= me["usable"]:
                lo = c
        if lo > best[0]:
            best = (lo, pb, need(p, lo)[1] if lo else 0.0)
        if lo >= min(32768, want):          # a big enough context with the faster batch: keep it
            break
    return best


def config_path(key: str) -> Path:
    return ROOT / f"strata-{MODELS[key]['name']}.json"


def write_config(key: str, data: Path, me: dict) -> Path:
    m = MODELS[key]
    gguf = data / "models" / m["gguf"]
    pack = data / "packs" / m["pack"] / "tokenizer"
    if not gguf.exists() or not pack.exists():
        fail(f"{m['title']} is not downloaded completely ({gguf if not gguf.exists() else pack} is missing)",
             f"make pull MODEL={key}")
    ctx, pb, total = size_context(gguf, m["context"], me)
    if not ctx:
        ctx, pb = 4096, 256
        warn(f"{m['title']} does not fit this Mac's memory even at a 4K context; written with the smallest settings")
    args = ["--gguf", str(gguf), "--max-context", str(ctx), "--kv", "f16"] + (["--prefill-batch", str(pb)] if pb != 2048 else [])
    cfg = {"backend": "metal", "exe": str(ENGINE / "strata-metal"), "cwd": str(ROOT), "args": args,
           "tokenizer": str(pack), "model_name": m["name"], "port": 8090, "host": "127.0.0.1",
           "log": str(ROOT / f"strata-{m['name']}.log")}
    if m["sampling"]:
        cfg["sampling"] = m["sampling"]
    path = config_path(key)
    old = json.loads(path.read_text()) if path.exists() else {}
    for k in ("api_key", "mcp_servers", "mcp", "firewall", "port", "host"):    # what the user set stays
        if k in old:
            cfg[k] = old[k]
    path.write_text(json.dumps(cfg, indent=1) + "\n")
    ok(f"{path.name}: context {ctx:,} tokens" + (f", prefill batch {pb}" if pb != 2048 else "") +
       (f", about {total / GB:.1f} GB of memory" if total else ""))
    return path


def download(key: str, data: Path, yes: bool, env: dict) -> Path | None:
    m = MODELS[key]
    if disk_free(data) < m["download"] * GB * 1.05 and not (data / "models" / m.get("gguf", "-")).exists():
        fail(f"{m['title']} needs {m['download']:.0f} GB of free disk; {data} has {disk_free(data) / GB:.0f} GB",
             "Free some space (or SETUP_ARGS=\"--data-dir /Volumes/...\"), then run `make setup` again.")
    if m["how"] == "setup":
        # setup.py's installer: the download, the MTP choice and its own run config (it asks its questions)
        run([sys.executable, "setup.py", "--setup", "--no-start", "--model", key, "--data-dir", str(data)] +
            (["--yes"] if yes else []), f"downloading {m['title']} ({m['download']:.0f} GB) with setup.py ...", env=env)
        found = sorted(ROOT.glob("strata-*.json"), key=lambda p: p.stat().st_mtime)
        return found[-1] if found else None
    run([sys.executable, "tools/pull_model.py", key, str(data)],
        f"downloading {m['title']} ({m['download']:.1f} GB; if it stops, `make setup` continues it) ...", env=env)
    ok(f"{m['title']} and its tokenizer pack in {data}")
    return None


# ------------------------------------------------------------------ the first answer

def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def first_answer(cfg_path: Path) -> tuple[bool, str]:
    """Start the server on a free port with this config, ask one question, stop it.  (ok, why not)."""
    port = free_port()
    log = cfg_path.with_suffix(".setup-test.log")
    with log.open("w") as out:
        srv = subprocess.Popen([str(ENGINE / "strata-server"), "--engine", "strata", "--config", str(cfg_path),
                                "--port", str(port)], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
    try:
        t0 = time.time()
        while time.time() - t0 < 900:
            if srv.poll() is not None:
                return False, "the engine stopped while loading: " + log.read_text()[-600:]
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read()
                break
            except OSError:
                time.sleep(2)
        body = json.dumps({"model": "x", "max_tokens": 300, "reasoning_effort": "low",
                           "messages": [{"role": "user", "content": "Say hello in five words."}]}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body, {"Content-Type": "application/json"})
        try:
            d = json.loads(urllib.request.urlopen(req, timeout=600).read())
        except urllib.error.HTTPError as e:
            d = json.loads(e.read() or b"{}")
        if "error" in d:
            return False, d["error"].get("message", str(d["error"]))
        msg = d["choices"][0]["message"]
        return True, (msg.get("content") or msg.get("reasoning_content") or "").strip()[:120]
    finally:
        srv.send_signal(signal.SIGTERM)
        try:
            srv.wait(120)
        except subprocess.TimeoutExpired:
            srv.kill()
        log.unlink(missing_ok=True)


def out_of_memory(why: str) -> bool:
    return any(s in why.lower() for s in ("out of memory", "insufficient memory", "llama_process failed (-3)", "failed (-3)"))


def shrink(cfg_path: Path) -> bool:
    """Half the context (down to 4096) and the smallest prefill batch: one more try after an out-of-memory."""
    cfg = json.loads(cfg_path.read_text())
    a = cfg["args"]
    ctx = int(a[a.index("--max-context") + 1])
    pb = int(a[a.index("--prefill-batch") + 1]) if "--prefill-batch" in a else 2048
    if ctx <= 4096 and pb <= 256:
        return False
    a[a.index("--max-context") + 1] = str(max(4096, ctx // 2))
    if "--prefill-batch" in a:
        a[a.index("--prefill-batch") + 1] = "256"
    else:
        a += ["--prefill-batch", "256" if pb <= 512 else "512"]
    cfg_path.write_text(json.dumps(cfg, indent=1) + "\n")
    note(f"trying again with a context of {max(4096, ctx // 2):,} tokens and a smaller prefill batch ...")
    return True


def test(cfg_path: Path, yes: bool) -> bool:
    if subprocess.run(["pgrep", "-f", "Strata.app/Contents/MacOS/Strata$"], capture_output=True).returncode == 0:
        if ask("Strata.app is running (its model uses memory). Quit it for the test? (y/n)", "y", yes).lower() == "y":
            subprocess.run(["make", "--no-print-directory", "app-stop"], cwd=ROOT)
    while True:
        note("a first answer (the model loads first: a few seconds to a minute or two) ...")
        good, said = first_answer(cfg_path)
        if good:
            ok(f"it answers: {said!r}")
            return True
        if out_of_memory(said):
            avail = free_now() / GB
            warn(f"the GPU ran out of memory ({avail:.1f} GB free right now)")
            if shrink(cfg_path):
                continue
            if not yes and sys.stdin.isatty():
                a = ask("Quit other apps (browsers, editors, Docker) to free memory, then press Enter to try again, "
                        "or type s to skip", "", False)
                if a.lower() != "s":
                    continue
            warn("it did not fit: quit other apps, or pick a smaller model (`make setup` again)")
            return False
        warn(f"the first answer failed: {said}")
        return False


def open_app(cfg_path: Path) -> None:
    settings = Path.home() / "Library/Application Support/Strata/app.json"
    settings.parent.mkdir(parents=True, exist_ok=True)
    s = json.loads(settings.read_text()) if settings.exists() else {}
    s["config"] = str(cfg_path)
    settings.write_text(json.dumps(s, indent=2))
    subprocess.run(["open", str(ROOT / "Strata.app")])
    ok("Strata.app is open on this model: press Start, then Chat")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="one of " + ", ".join(MODELS))
    ap.add_argument("--yes", action="store_true", help="take every default, ask nothing")
    ap.add_argument("--data-dir", default=str(ROOT.parent / "Strata-data"))
    ap.add_argument("--no-open", action="store_true", help="do not open Strata.app at the end")
    ap.add_argument("--no-test", action="store_true", help="skip the first answer")
    ap.add_argument("--config-only", metavar="MODEL", help="only write the run config of a downloaded model")
    a = ap.parse_args()
    data = Path(a.data_dir).expanduser().resolve()
    me = mac()
    if a.config_only:
        key = a.config_only.upper()
        if key in MODELS and MODELS[key]["how"] == "pull":
            write_config(key, data, me)
        return

    total = 6
    step(1, total, "This Mac")
    if platform.machine() != "arm64":
        fail("this Mac has an Intel processor", "Strata's Metal engine runs on Apple Silicon (M1 or newer) only.")
    ok(f"{me['chip']}, {me['ram'] / GB:.0f} GB of memory, macOS {me['macos']}")
    ok(f"the GPU may use {me['limit'] / GIB:.1f} GB; {free_now() / GB:.1f} GB is free right now; "
       f"{disk_free(data) / GB:.0f} GB free disk for {data}")

    step(2, total, "Tools")
    env = tools(a.yes)

    step(3, total, "Builds")
    builds(env)

    step(4, total, "The model")
    key = choose(me, a.model, a.yes)
    m = MODELS[key]
    ok(f"{m['title']}")

    step(5, total, "Download and run config")
    cfg = download(key, data, a.yes, env)
    if m["how"] == "pull":
        cfg = write_config(key, data, me)
    if not cfg:
        fail("no run config was written", "Run `make setup` again.")

    step(6, total, "First answer")
    good = True if a.no_test else test(cfg, a.yes)

    print()
    if good:
        print(f"{GREEN}{BOLD}Strata is ready.{END}")
    print(f"  model:   {m['title']}  ({cfg.name})\n  app:     open Strata.app   (or: make app-run)\n"
          f"  browser: make run PORT=8090, then http://127.0.0.1:8090\n  another model: make setup")
    if not a.no_open:
        open_app(cfg)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped. Run `make setup` again to continue where it stopped.")
        sys.exit(130)
