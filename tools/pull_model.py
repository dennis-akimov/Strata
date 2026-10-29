"""tools/pull_model.py NAME [DATA_DIR] - the models setup does not install, each as docs/MACOS.md does it by hand
(`make pull MODEL=NAME`; the chat quants Q2_0 ... UD-IQ4_XS go through setup instead).  Every file comes from a pinned
Hugging Face revision; a stopped download continues when you run the same line again.

  GLM-5.3        GLM-5.3-Flash Maya-S24 (94.7 GB) and its tokenizer pack
  GPT-OSS-120B   gpt-oss-120b MXFP4 (63.4 GB, sha-256 checked) and its tokenizer pack
  GPT-OSS-120B-HERETIC  llmfan46's gpt-oss-120b-heretic-v2, refusals removed (65.4 GB, sha-256 checked), requantized
                 to stock's layout (63.4 GB: BF16 attention/output -> Q8_0, +26% decode), and its tokenizer pack
  GPT-OSS-120B-DFLASH  z-lab's DFlash draft for gpt-oss-120b (1.6 GB, sha-256 checked), converted to one GGUF
                 (strata-metal --dflash; it needs metal/patches/core/0002)
  GPT-OSS-20B    gpt-oss-20b MXFP4 (12.1 GB, sha-256 checked) and its tokenizer pack
  QWEN3.6-35B    Qwen3.6-35B-A3B MXFP4_MOE (21.7 GB, sha-256 checked) and its tokenizer pack
  NUCLEUS        Nucleus-Image: ~50 GB downloaded, converted to 3 GGUFs (~21 GB), the downloads deleted
  HUNYUAN        HunyuanImage-3.0-Instruct-Distil: 169 GB streamed shard by shard into one 54 GB GGUF
  WAN2.2         Wan 2.2 T2V A14B for videos: both experts (Q4_K_M), UMT5 (Q8_0), the VAE, the Lightning LoRAs (29 GB),
                 and stable-diffusion.cpp's sd-cli built for Metal with metal/video/sdcpp-metal-conv3d.patch

Downloads: aria2c when installed (16 connections a file; brew install aria2), else Python's own, resumable
(an HTTP Range from what is on disk), with the same sha-256 check.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
PY = ROOT / ".venv" / "bin" / "python"
MTP_PY = ROOT / ".venv-mtp" / "bin" / "python"
NAMES = ("GLM-5.3", "GPT-OSS-120B", "GPT-OSS-120B-HERETIC", "GPT-OSS-120B-DFLASH", "GPT-OSS-20B", "QWEN3.6-35B",
         "NUCLEUS", "HUNYUAN", "WAN2.2")


def die(msg: str) -> None:
    print(msg, flush=True)
    sys.exit(1)


def sh(*cmd, quiet: bool = False, cwd=None) -> None:
    out = subprocess.DEVNULL if quiet else None
    if subprocess.run([str(c) for c in cmd], cwd=cwd, stdout=out).returncode != 0:
        die(f"failed: {' '.join(map(str, cmd))}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, dest: Path) -> None:
    """One file with Python, continued from what is on disk (Range); a progress line every few seconds."""
    have = dest.stat().st_size if dest.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                total = have + int(r.headers.get("Content-Length") or 0)
                mode = "ab" if r.status == 206 else "wb"
                done = have if r.status == 206 else 0
                t, last = time.time(), 0.0
                with dest.open(mode) as f:
                    for block in iter(lambda: r.read(1 << 20), b""):
                        f.write(block)
                        done += len(block)
                        if time.time() - last > 5:
                            last = time.time()
                            mbs = (done - have) / max(1e-3, last - t) / 1e6
                            print(f"  {dest.name}: {done / 1e9:.2f} of {total / 1e9:.2f} GB, {mbs:.0f} MB/s", flush=True)
            return
        except urllib.error.HTTPError as e:
            if e.code == 416:            # asked past the end: the file is complete already
                return
            err = e
        except (urllib.error.URLError, OSError) as e:
            err = e
        print(f"  {dest.name}: {err}; retrying ({attempt + 1}/5) ...", flush=True)
        time.sleep(5)
        have = dest.stat().st_size if dest.exists() else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    die(f"{dest.name}: the download did not finish; run the same command again (it continues)")


def get(d: Path, url: str, sha: str = "") -> Path:
    """One file into d under the URL's own name (Hugging Face redirects some to a CDN link whose name is a hash)."""
    d.mkdir(parents=True, exist_ok=True)
    dest = d / os.path.basename(url)
    if sha and dest.exists() and not (d / (dest.name + ".aria2")).exists() and sha256(dest) == sha:
        return dest                       # complete and checked
    if shutil.which("aria2c"):
        sh("aria2c", "-x16", "-s16", "-k8M", "-c", "--file-allocation=none", "--console-log-level=warn",
           "--summary-interval=0", "-d", d, "-o", dest.name, *([f"--checksum=sha-256={sha}"] if sha else []), url)
        return dest
    fetch(url, dest)
    if sha and sha256(dest) != sha:
        dest.unlink()
        die(f"{dest}: sha-256 differs from the pinned one (deleted; run the same command again)")
    return dest


def get_dir(repo: str, rev: str, path: str, dest: Path) -> None:
    """Every file under path (its sub-folders kept) into dest/path."""
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}/tree/{rev}/{path}?recursive=true", timeout=60) as r:
        files = [f["path"] for f in json.load(r) if f["type"] == "file"]
    for f in files:
        get(dest / os.path.dirname(f), f"https://huggingface.co/{repo}/resolve/{rev}/{f}")


def tokenizer(gguf: Path, out: Path) -> None:
    sh(PY, "tools/strata_tokenizer.py", "--gguf", gguf, "--out", out)


def llama_src() -> Path:
    found = sorted(glob.glob("build-metal/_deps/llama_cpp-src-*"))
    if not found:
        die("this model needs the Metal engine's llama.cpp: make build first")
    return Path(found[0])


def ninja() -> str:
    return shutil.which("ninja") or str(ROOT / ".venv" / "bin" / "ninja")


def cmake() -> str:
    v = ROOT / ".venv" / "bin" / "cmake"
    return str(v) if v.exists() else (shutil.which("cmake") or "cmake")


def ensure_venv() -> None:
    """The tokenizer packs and converters need .venv: made here when `make setup` has not run yet."""
    if PY.exists():
        return
    print("making .venv (Python packages for the tokenizer pack) ...", flush=True)
    sh(sys.executable, "-m", "venv", ".venv")
    sh(PY, "-m", "pip", "install", "-q", "--disable-pip-version-check", "-r", "requirements.txt")


def pull(name: str, data: Path) -> None:
    m = data / "models"
    if name == "GLM-5.3":
        d = m / "GLM-5.3-Flash-Maya-S24"
        r = "https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF/resolve/f16ee4ba50d8fd2bb726ea1862ca6ca14ba188da"
        for i in (1, 2, 3):
            get(d, f"{r}/Maya-S24/GLM-5.3-Flash-Maya-S24-IQ2_XXS_S-0000{i}-of-00003.gguf")
        get(d, f"{r}/vision/GLM-5.3-Flash-vocab.gguf")
        tokenizer(d / "GLM-5.3-Flash-vocab.gguf", data / "packs" / "glm-5.3-flash-maya-s24")
    elif name == "GPT-OSS-120B":
        d = m / "gpt-oss-120b"
        g = get(d, "https://huggingface.co/ggml-org/gpt-oss-120b-GGUF/resolve/238abdd290bb874b90a5da1b4549881b7d05c091/gpt-oss-120b-MXFP4.gguf",
                "582bd40f6886200101f4c4ed9f25f3fe80cc14c86e9e2b37746cd8904a0c622d")
        tokenizer(g, data / "packs" / "gpt-oss-120b")
    elif name == "GPT-OSS-120B-HERETIC":
        d = m / "gpt-oss-120b-heretic-v2"
        final = d / "gpt-oss-120b-heretic-v2-MXFP4_MOE.gguf"
        if final.exists():
            print(f"GPT-OSS-120B-HERETIC: already in {d}")
            return
        src = llama_src()
        g = get(d, "https://huggingface.co/llmfan46/gpt-oss-120b-heretic-v2-GGUF/resolve/58b83826e2f3e652fa404e493bb9982884bd7200/gpt-oss-120b-heretic-v2-MXFP4.gguf",
                "8f2bc3ba1f84707ce18bbc75a4cdbb211f365928e60a1ed076238dfd3de0c0f7")
        tokenizer(g, data / "packs" / "gpt-oss-120b-heretic-v2")
        # the published file keeps attention and the output layer in BF16 (5.0 GB read a token against stock's 3.6 GB:
        # 84 against 106 tok/s); MXFP4_MOE writes them as Q8_0 as stock does and copies the MXFP4 experts byte for byte
        sh(cmake(), "-S", src, "-B", "build-llama-tools", "-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={ninja()}",
           "-DCMAKE_BUILD_TYPE=Release", "-DGGML_METAL=ON", "-DLLAMA_CURL=OFF", "-DLLAMA_BUILD_TESTS=OFF", quiet=True)
        sh(cmake(), "--build", "build-llama-tools", "--target", "llama-quantize", quiet=True)
        tmp = Path(str(final) + ".tmp")
        sh("build-llama-tools/bin/llama-quantize", g, tmp, "MXFP4_MOE")
        tmp.rename(final)
        g.unlink(missing_ok=True)               # only the requantized file is used
    elif name == "GPT-OSS-120B-DFLASH":
        d = m / "gpt-oss-120b-DFlash"
        out = d / "gpt-oss-120b-DFlash-bf16.gguf"
        if out.exists():
            print(f"GPT-OSS-120B-DFLASH: already in {d}")
            return
        if not MTP_PY.exists():
            die("GPT-OSS-120B-DFLASH needs .venv-mtp (PyTorch) for llama.cpp's converter: setup makes it")
        src = llama_src()
        r = "https://huggingface.co/z-lab/gpt-oss-120b-DFlash/resolve/1278df34f0a7bd2c8588a27f49048aaa05c7db00"
        get(d / "hf", f"{r}/config.json")
        get(d / "hf", f"{r}/model.safetensors", "4546a44a2b4f3da784003964085f979ef1e59fb22955dafbac11535b7f413827")
        # the converter takes the draft's tokenizer from its target: gpt-oss-120b's config and tokenizer (28 MB)
        t = "https://huggingface.co/openai/gpt-oss-120b/resolve/b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
        for f in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                  "generation_config.json", "chat_template.jinja"):
            get(d / "target", f"{t}/{f}")
        sh(MTP_PY, src / "convert_hf_to_gguf.py", d / "hf", "--target-model-dir", d / "target",
           "--model-name", "gpt-oss-120b-DFlash", "--outtype", "bf16", "--outfile", out)
        shutil.rmtree(d / "hf", ignore_errors=True)       # only the GGUF is used
        shutil.rmtree(d / "target", ignore_errors=True)
    elif name == "GPT-OSS-20B":
        g = get(m / "gpt-oss-20b", "https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/resolve/ef9b12f2ff56c69cf32153a02784e7a3c88bf524/gpt-oss-20b-MXFP4.gguf",
                "27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901")
        tokenizer(g, data / "packs" / "gpt-oss-20b")
    elif name == "QWEN3.6-35B":
        g = get(m / "Qwen3.6-35B-A3B", "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/a483e9e6cbd595906af30beda3187c2663a1118c/Qwen3.6-35B-A3B-MXFP4_MOE.gguf",
                "2fdd20997c4d88ee25f70f500c61f8b999378d92ab055f9d450fc70d617158d3")
        tokenizer(g, data / "packs" / "qwen3.6-35b-a3b")
    elif name == "NUCLEUS":
        d = m / "Nucleus-Image"
        outs = [d / "nucleus-dit-q4_0.gguf", d / "nucleus-vae.gguf", d / "qwen3vl-8b-text-q8_0.gguf"]
        if all(o.exists() for o in outs):
            print(f"NUCLEUS: already in {d}")
            return
        if not MTP_PY.exists():
            die("NUCLEUS needs .venv-mtp (PyTorch) for its text encoder: setup makes it")
        get_dir("treadon/mlx-nucleus-image", "d5ac768cf398a25c4e157bffdcd193edf54f175b", "dit", d / "mlx")
        get_dir("treadon/mlx-nucleus-image", "d5ac768cf398a25c4e157bffdcd193edf54f175b", "vae", d / "mlx")
        get_dir("NucleusAI/Nucleus-Image", "5e963db4fd0a65c7e4faf53ca2d4eca567c4dcfa", "text_encoder", d / "hf")
        src = llama_src()
        sh(PY, "tools/image/nucleus_gguf.py", "--src", d / "mlx" / "dit", "--out", outs[0], "--experts", "q4_0")
        sh(PY, "tools/image/nucleus_gguf.py", "--vae", "--src", d / "mlx" / "vae", "--out", outs[1])
        sh(MTP_PY, src / "convert_hf_to_gguf.py", d / "hf" / "text_encoder", "--outtype", "q8_0", "--outfile", outs[2])
        shutil.rmtree(d / "mlx", ignore_errors=True)      # only the 3 GGUFs are used
        shutil.rmtree(d / "hf", ignore_errors=True)
    elif name == "HUNYUAN":
        d = m / "HunyuanImage-3"
        if not (d / "hunyuan-image3-q4_0.gguf").exists():   # its .tmp is not resumable: a stopped conversion starts over
            sh(PY, "tools/image/hunyuan_gguf.py", "--out", d / "hunyuan-image3-q4_0.gguf", "--work", d / "shards")
        get(d, "https://huggingface.co/tencent/HunyuanImage-3.0-Instruct-Distil/resolve/c8ffd07206f1b843697606968196e8f59f8ff38c/tokenizer.json")
    elif name == "WAN2.2":
        d = m / "Wan2.2-T2V-A14B"
        q = "https://huggingface.co/QuantStack/Wan2.2-T2V-A14B-GGUF/resolve/73eafba53a1a8f29254e4c77f92e74ea27d7cd6f"
        get(d, f"{q}/HighNoise/Wan2.2-T2V-A14B-HighNoise-Q4_K_M.gguf")
        get(d, f"{q}/LowNoise/Wan2.2-T2V-A14B-LowNoise-Q4_K_M.gguf")
        get(d, "https://huggingface.co/city96/umt5-xxl-encoder-gguf/resolve/b535255bee98c2b0a59ea7c0ae2dcd0c6657b3b7/umt5-xxl-encoder-Q8_0.gguf")
        get(d, "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/123acf1cc74bccbb9bfff8ac1ee72edc08c2341d/split_files/vae/wan_2.1_vae.safetensors")
        # lightx2v's Wan2.2-Lightning LoRAs (4 steps, no CFG; Apache-2.0): their integer alphas as float32 for sd-cli
        lora = "https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/18bccf8884ec0a078eed79785eb4ef13ea16ce1e/Wan2.2-T2V-A14B-4steps-lora-250928"
        for x in ("high", "low"):
            done = d / "loras" / f"lightning-{x}.safetensors"
            if done.exists():                   # the converter writes it only when complete
                continue
            raw = get(d / "loras", f"{lora}/{x}_noise_model.safetensors")
            sh(PY, "tools/video/lora_alpha_f32.py", raw, done)
            raw.unlink(missing_ok=True)
        # sd-cli: stable-diffusion.cpp at a pinned commit, with Strata's patch (Metal has no IM2COL_3D: the Wan VAE's
        # Conv3d as 2D convolutions per temporal tap, 13x faster decoding)
        b, sdc = Path("build-sdcpp"), "d637cfb935feb2254898a8f1524046c958f3df80"
        patch = "../../metal/video/sdcpp-metal-conv3d.patch"
        if not (b / "src" / ".git").exists():
            sh("git", "clone", "-q", "https://github.com/leejet/stable-diffusion.cpp", b / "src")
        head = subprocess.run(["git", "-C", str(b / "src"), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        patched = subprocess.run(["git", "-C", str(b / "src"), "apply", "--reverse", "--check", patch],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        if head != sdc or not patched:
            sh("git", "-C", b / "src", "fetch", "-q", "origin")
            sh("git", "-C", b / "src", "checkout", "-q", "-f", sdc)
            sh("git", "-C", b / "src", "submodule", "update", "-q", "--init", "--recursive")
            sh("git", "-C", b / "src", "apply", patch)
        sh(cmake(), "-S", b / "src", "-B", b / "build", "-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={ninja()}",
           "-DCMAKE_BUILD_TYPE=Release", "-DSD_METAL=ON", quiet=True)
        sh(cmake(), "--build", b / "build", "--target", "sd-cli")
        print(f"sd-cli: {ROOT / b / 'build' / 'bin' / 'sd-cli'}")
    print(f"{name}: done in {m}")


def main() -> None:
    name = (sys.argv[1] if len(sys.argv) > 1 else "").upper()
    data = Path(sys.argv[2] if len(sys.argv) > 2 else "../Strata-data").expanduser()
    if name not in NAMES:
        print("make pull MODEL=NAME: a setup quant (Q2_0 IQ2_XS IQ3_XXS IQ3_S IQ1_M UD-Q4_K_XL UD-IQ4_XS) or one of")
        print(" ".join(NAMES) + " (tools/pull_model.py)")
        sys.exit(2)
    ensure_venv()
    pull(name, data)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped: run the same command again to continue.")
        sys.exit(130)
