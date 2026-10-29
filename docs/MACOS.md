# Strata on a Mac (Apple Silicon)

The main document of this independently maintained, Apple Silicon-only fork of
[Strata](https://github.com/Niko1221/Strata) (the original's Windows and Linux engines are kept in the repository as
they were at the fork point, but not maintained here; see [docs/README.md](README.md)).

**Tested on one Mac:** a MacBook Pro M5 Max (40-core GPU, 128 GB) on macOS 26.4, with Qwen3.8-Flash-Next Q2_0, IQ3_XXS
and IQ3_S, GPT-OSS 120B and 20B, Qwen3.6-35B-A3B and GLM-5.3-Flash (Maya-S24). Other Apple Silicon Macs and other model files are untested. Read [Limits and warnings](#limits-and-warnings) before you install.

On a Mac, Strata serves the same web app and APIs (OpenAI, Anthropic, Responses, MCP) as on a PC. The engine under
them is different: `strata-metal` (`metal/`) runs the model on llama.cpp's Metal backend.

## Before you start

- An Apple Silicon Mac (M1 or newer; setup refuses Intel Macs). 16 GB runs GPT-OSS 20B at a small context with other
  apps closed; 24-32 GB runs it comfortably; Qwen3.6-35B wants 32 GB; the Qwen3.8 models, GPT-OSS 120B and GLM-5.3
  want 64-128 GB. Only an M5 Max with 128 GB and an M1 with 16 GB were tried.
- Free disk for the model (12-95 GB) on the internal SSD, and an internet connection for the first run.
- Nothing else: `make setup` installs what is missing.
- Expect the speed to drop after a few minutes of steady use (on the test Mac the GPU's clock went down; see below).

## Quick start

In Terminal:

```sh
git clone https://github.com/dennis-akimov/Strata.git
cd Strata
make setup
```

`make setup` (metal/onboard_mac.py, after bootstrap.py) goes step by step and says what it does:

| Step | What happens |
|---|---|
| Tools | Apple's Command Line Tools (an installer window opens: click Install; setup waits), Homebrew and Python 3.12 when no Python 3.10+ is there (Homebrew asks for your password), the Python packages in `.venv` (cmake, ninja, ...), Rust (rustup, into `~/.cargo`), aria2 (faster downloads; curl otherwise) |
| Builds | the Metal engine (3-6 minutes, once), the server and its file tool into `engine-metal/`, `Strata.app` |
| The model | every chat model with its download, its memory and whether it fits this Mac; the recommended one is preselected (Enter) |
| Download | resumable, SHA-256 checked, with its tokenizer pack, into `Strata-data/` next to the `Strata` folder |
| Run config | `strata-<model>.json` with the largest context (and prefill batch) that fits: the engine measures the model's memory (`strata-metal --plan`) against what Metal lets the GPU use and the Mac's memory |
| First answer | the model loads on a free port and answers one question; if the GPU runs out of memory, setup halves the context and tries again, then says what to close |
| Strata.app | opens on that model: Start, then Chat |

If anything stops, run `make setup` again: finished steps are skipped and a download continues. `make setup
MODEL=GPT-OSS-20B SETUP_ARGS=--yes` asks nothing; `SETUP_ARGS="--data-dir /Volumes/X/Strata-data"` puts the models
elsewhere; `make pull MODEL=...` downloads another model and writes its run config too. Setup.py's own installer
(the Qwen3.8 quants with the MTP choice and its questions) is still `python3 bootstrap.py --setup`.

`make run` keeps the model running in that Terminal window; Ctrl-C stops it. On the test Mac the engine compiled in
3-6 minutes (once) and later starts took about 35 seconds.

**Getting the model.** `make pull` downloads Q2_0, 66.4 GB in two files, from Hugging Face
([ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF),
pinned to one revision), plus the 0.9 GB image encoder if you answer yes to images. The files go to `Strata-data/`
next to the `Strata` folder; each one's size is checked against the server's when it finishes. If the download stops,
run the same command again: it continues where it stopped. Q2_0, IQ3_XXS (`make pull MODEL=IQ3_XXS`, 75.8 GB) and IQ3_S (`make pull MODEL=IQ3_S`, 83.6 GB) are the sizes tested on a Mac. Add
`SETUP_ARGS="--yes"` to skip the questions, and set `HF_ENDPOINT` to use a Hugging Face mirror.

`make run` alone also works the first time: it asks the same questions, downloads, then starts.

| To | Run |
|---|---|
| run in the background | `make start`, then `make status` / `make stop` (log: `strata-run.log`) |
| send one test message | `make chat PROMPT="Write a haiku"` |
| cap the thinking | `make chat PROMPT="..." EFFORT=high MAX_TOKENS=32768 REASONING_BUDGET=4096` (the thinking is closed there, then it answers if max tokens leaves room); in the web chat: Sampling > Thinking budget |
| use another port | `make run PORT=8090`, or put `PORT := 8090` in a file named `Makefile.local` |
| use a bigger context window | `make run CONTEXT=131072` (up to 262,144; it stays set, and needs more memory) |
| download without questions | `make pull MODEL=Q2_0 SETUP_ARGS="--yes"` |
| try another size (untested on a Mac) | `make pull MODEL=IQ2_XS`, then `make run` |
| turn on the MTP draft layer | `python3 bootstrap.py --setup --mtp on` (see below) |
| see every target | `make` |

If setup stops, it says what is missing and the command that fixes it.

## Limits and warnings

- **Memory.** A Mac's CPU and GPU share one memory. Metal sets a limit on how much the GPU may use at once
  (`recommendedMaxWorkingSetSize`); `make check` prints this Mac's value. On the test Mac it was 107.5 of 128 GB; it
  differs between Macs and macOS versions. Q2_0's weights take about 35 GB of it (IQ3_S's about 51 GB), plus the context's cache. Everything
  the Mac does shares the same memory, so close other large apps. Below 128 GB only the 64 GB floor applies: nobody
  has measured how close a 64 GB Mac gets.
- **Of the Qwen sizes, only Q2_0, IQ3_XXS and IQ3_S were tested** (and GPT-OSS 120B and 20B, Qwen3.6-35B-A3B and GLM-5.3-Flash Maya-S24 of the others). `make check` marks the other sizes "untested on a Mac". On a PC, the Unsloth sizes
  (UD-Q4_K_XL, UD-IQ4_XS) stream part of their experts from the SSD; the Mac engine cannot, so all of a model's experts
  must fit in Metal's limit. UD-Q4_K_XL (111 GB) is larger than the test Mac's default limit.
- **Speed.** On the test Mac: 13-17 tokens/s for the answer and 225-270 tokens/s to read a prompt, with other programs
  running (see [Measured](#measured)). Other Macs will differ; slower memory means fewer tokens per second.
- **It can halve after a few minutes** (in the *Automatic* energy mode; see *Power* above). On the test Mac, short IQ3_S answers with `--mtp on`, sent one after another,
  ran at 59-65 tokens/s in the first minute after a pause and at about 30-35 (at times 13-20) later, with the same
  answers. The slow phase came with a low GPU clock (600-830 MHz), the GPU busy about half the time, 8-11 W and a
  59-69 °C die. The cause was not isolated; macOS's power management is the likely one. A single long answer, Q2_0 and
  MTP off were not measured this way. See [Why the speed changes](#why-the-speed-changes);
  [macmon](https://github.com/vladkens/macmon) (`brew install macmon`, no sudo) shows the GPU's clock live.
- **Power: use *High Power*.** On the test Mac (16-inch, M5 Max) the speed of long answers changed up to 3x with the
  power setup, at only 60-95 °C. Decode tok/s, thinking off, 2026-10-08:

  | | *Automatic*, 96 W adapter | *High Power*, 96 W | *High Power*, 140 W |
  |---|---:|---:|---:|
  | GPT-OSS 120B, 900-word answer | 35 | 46-53\* | 93-100 |
  | GPT-OSS, a short answer right after it | 33 | 49\* | 93 |
  | Q2_0 (MTP on), 900-word answer | 24-27\*\* | 63-69 | 60-67 |
  | GPU clock (during the long answers) | 364-807 MHz | 1,300-1,620 MHz | 1,200-1,620 MHz |

  \* Measured seconds after switching to *High Power*. \*\* From [IQ3_S compared with Q2_0](#iq3_s-compared-with-q2_0)
  (the same MTP settings, but 233-353-token answers).

  The adapter made no difference for Q2_0 (63-69 on the 96 W one, 60-67 on the 140 W one); both were negotiated at the
  same 94 W (20 V, 4.69 A) over the USB-C cable used, so set *System Settings > Battery > Energy Mode* to *High Power*
  where your Mac has it (louder fans; the GPU reached 95 °C) and check the GPU's clock with
  [macmon](https://github.com/vladkens/macmon) when it seems slow. The same slow-down in *Automatic* is in llama.cpp
  [#10444](https://github.com/ggml-org/llama.cpp/issues/10444) (M3 Max, 96 W adapter). Earlier speed numbers on this page
  were measured in *Automatic*.
- **Disk.** The download is 66.4 GB (67.3 GB with the image encoder); installed with the engine, about 70 GB. `--mtp on`
  adds about 12 GB. Keep the model on the internal SSD: its 28 GB n-gram table is not loaded into memory, its rows are
  read from the model file as they are needed.
- **MTP is opt-in.** It makes answers faster (see Measured), costs about 12 GB of disk and keeps the KV cache 16-bit.
  With it on, a long answer can differ from the answer without it. Two requests at once (`"parallel"`) get no drafts.
- **Not on a Mac yet:** contexts past 262,144 tokens (rope scaling), the PC engine's expert cache and CPU experts.

### If a model does not fit

Pick a smaller size or a smaller context first. If you know what you are doing, you can raise Metal's limit; the
setting lasts until the Mac restarts:

```sh
sysctl -n iogpu.wired_limit_mb               # the current value in MB; 0 means macOS' default
sysctl -n hw.memsize | awk '{print $1/1048576 " MB of memory"}'
sudo sysctl iogpu.wired_limit_mb=$(( $(sysctl -n hw.memsize) / 1048576 - 16384 ))   # all memory but 16 GB
sudo sysctl iogpu.wired_limit_mb=0           # back to the default (or to the value the first line printed)
```

macOS' default is already close to this on a 128 GB Mac (107.5 GB there), so this helps most on smaller Macs, which
keep a larger share for the system. Never set more than the Mac's memory.

This is an undocumented macOS setting, and Strata never changes it. Leave macOS plenty of memory (Strata's rule of
thumb: 8-16 GB): set too high, the whole Mac can slow down or stop responding until it restarts.

## What works

| | On a Mac |
|---|---|
| Chat, tools, the web app, the APIs | yes, the same server as on a PC |
| Conversation reuse | yes: a follow-up reads only what is new |
| Pictures | yes: `strata-vision` runs on Metal |
| MTP draft layer | opt-in: `--mtp on` |
| GPT-OSS 120B (OpenAI's harmony format) | yes, set up by hand: see [GPT-OSS 120B](#gpt-oss-120b) |
| GLM-5.3-Flash (Maya-S24) | yes, set up by hand: see [GLM-5.3-Flash](#glm-53-flash) |
| Pictures from a prompt (`/image`, `/v1/images/generations`) | yes, on Strata's own engine (`strata-image`), with the model files converted by hand: see [Pictures from a prompt](#pictures-from-a-prompt-nucleus-image) |
| DFlash draft model (`--dflash`) | opt-in, for GPT-OSS 120B: faster on math, slower on chat ([below](#gpt-oss-120b-with-a-dflash-draft)) |
| EAGLE3 draft model (`--eagle3`) | opt-in engine flag, off: slower than no draft on GPT-OSS ([below](#eagle3-draft-model---eagle3)) |
| Several requests at once (`"parallel"`) | opt-in, the same as on a PC |
| Monitor tab | GPU load, memory and power, the chip's temperature (its die sensors), CPU, RAM; PCIe shows "n/a" (an integrated GPU has no PCIe link) |
| Expert cache, CPU experts, rope scaling, Intel Macs | no |

### The MTP draft layer (`--mtp on`)

The model's own draft head guesses the next 3 tokens, and the model checks them in one pass. Setup builds the head from
the original checkpoint: it downloads the 31 MTP tensors (5 GB, SHA-256 checked) and the embeddings and LM head (2.5 GB),
and llama.cpp's converter makes a 4.1 GB file of them. The converter needs PyTorch, so setup installs it once into
`.venv-mtp` (about 730 MB). llama.cpp pins numpy 2.2.6, which converts this head wrongly (no error, but no draft is ever
accepted), so setup uses numpy 2.4.0 and checks the result against a known SHA-256.

### GPT-OSS 120B

OpenAI's [gpt-oss-120b](https://huggingface.co/ggml-org/gpt-oss-120b-GGUF) (117B parameters, about 5B active per token,
MXFP4) runs in Strata on the same engine. Setup does not install it; by hand:

```sh
D=../Strata-data/models/gpt-oss-120b; mkdir -p $D
aria2c -x16 -s16 -c -d $D \
  "https://huggingface.co/ggml-org/gpt-oss-120b-GGUF/resolve/238abdd290bb874b90a5da1b4549881b7d05c091/gpt-oss-120b-MXFP4.gguf" \
  --checksum=sha-256=582bd40f6886200101f4c4ed9f25f3fe80cc14c86e9e2b37746cd8904a0c622d          # 63.4 GB
.venv/bin/python tools/strata_tokenizer.py --gguf $D/gpt-oss-120b-MXFP4.gguf --out ../Strata-data/packs/gpt-oss-120b
```

Then a run config like `strata-<model>.json` with `"args": ["--gguf", "<that file>", "--max-context", "131072", "--kv",
"f16"]`, `"tokenizer": "<the pack>/tokenizer"`, `"backend": "metal"` and OpenAI's suggested sampling `"sampling":
{"temperature": 1.0, "top_p": 1.0}`, and `serve/server.py --engine strata --config <it> --port 8090`. `GET /v1/models` (or
`/models`) lists the loaded model under the config's `"model_name"`; without one, under the GGUF's file name
(`gpt-oss-120b-mxfp4`).

What Strata does for it: its tokenizer (`pre` = `gpt-4o`, OpenAI's o200k) matched `openai/gpt-oss-120b`'s own
tokenizer on 2,178 strings (87,702 tokens, 0 differences); the server reads its replies in OpenAI's harmony format (the
`analysis` channel is the thinking, `final` the answer, a message `to=functions.NAME` a tool call) and closes the
thinking the harmony way when a thinking budget runs out; the engine keeps its conversation checkpoints at `<|start|>`.

| Measured on the test Mac, 2026-10-08 | |
|---|---|
| Answers, a 40-token thinking budget, a tool call and its result (OpenAI API) | correct |
| Decode, short answers, just after loading | 99-105 tok/s |
| Decode, 256-token answers, minutes later (GPU at ~990 MHz, see [Why the speed changes](#why-the-speed-changes)) | 20-33 tok/s |
| A follow-up that shares the start of the conversation | read only its new part (125 of 167 prompt tokens reused) |
| Memory | the 63 GB file, plus the context's cache |

Limits: GPT-OSS always thinks, so "Off" in the chat's Thinking setting means its low effort. A `tool_choice` that names a
tool is not forced (the model chooses); its template writes one tool call per message. The web chat's own defaults
(temperature 0.6, top-k 20) are Qwen's: set temperature 1.0 and top-p 1.0 there for GPT-OSS. Only the `qwen35` and
`gpt-4o` tokenizers are known to Strata; another model's pack is refused rather than tokenized wrongly.

### GPT-OSS 120B with a DFlash draft

z-lab's [gpt-oss-120b-DFlash](https://huggingface.co/z-lab/gpt-oss-120b-DFlash) (0.8B parameters, MIT) is a small draft
model that guesses a block of tokens in one pass from 5 of GPT-OSS's layers; GPT-OSS then checks the guesses in one pass
(`strata-metal --dflash <gguf> --spec N`, llama.cpp's `draft-dflash`). `make pull MODEL=GPT-OSS-120B-DFLASH` downloads
it from a pinned revision (SHA-256 checked), converts it with llama.cpp's converter (it needs `.venv-mtp` and a built
Metal engine) and keeps only `gpt-oss-120b-DFlash-bf16.gguf` (1.6 GB). The run config is GPT-OSS 120B's with
`"--dflash", "<that file>", "--spec", "3"` added. This draft has attention biases, which llama.cpp's DFlash code did not
load: `metal/patches/core/0002-dflash-attention-bias.patch` adds them.

| Measured through the server, 2026-10-13, greedy, 300 tokens, 2 runs each | Without the draft | With it (`--spec 3`) |
|---|---|---|
| Math (a step-by-step answer) | 99, 91 tok/s | **135, 122 tok/s** |
| Code (a Python function with doctests) | 102, 99 tok/s | 101, 98 tok/s |
| Chat (a 200-word explanation) | 97, 88 tok/s | 77, 71 tok/s |

Use it for math and other step-by-step answers, not for chat. Why: checking 4 tokens costs GPT-OSS about 1.9 times one
token (its experts differ per token), so drafts must be mostly right to pay off; the draft model's guesses were accepted
86% of the time on math, 53-64% on code and 43-46% on chat. `--spec 5` and `--spec 9` were slower in every case. A Q8_0 copy
of the draft was no faster than bf16. The answers start the same as without the draft, but a long greedy answer can
differ in places (the 4-token check rounds differently), as with EAGLE3.

### GPT-OSS 20B and Qwen3.6-35B-A3B

Two smaller models with few parameters active per token, on the same engine: OpenAI's
[gpt-oss-20b](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF) (21B parameters, about 3.6B active, MXFP4, 12.1 GB) and
Qwen's [Qwen3.6-35B-A3B](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) (35B, about 3B active, Unsloth's MXFP4_MOE,
21.7 GB). `make pull MODEL=GPT-OSS-20B` and `make pull MODEL=QWEN3.6-35B` download each from a pinned revision (SHA-256
checked) and make its tokenizer pack (`packs/gpt-oss-20b`, `packs/qwen3.6-35b-a3b`; each matched its GGUF's vocabulary on
16 test strings).

Then a run config as for [GPT-OSS 120B](#gpt-oss-120b) with that file and pack, `"--max-context", "131072"`, `"--kv",
"f16"`, and the model's own sampling: gpt-oss-20b `{"temperature": 1.0, "top_p": 1.0}`, Qwen3.6 `{"temperature": 0.6,
"top_p": 0.95, "top_k": 20}`.

| Measured on the test Mac, 2026-10-12 | gpt-oss-20b | Qwen3.6-35B-A3B |
|---|---|---|
| A short reasoning question (a trip from 14:35 to 17:10) | correct | correct |
| Decode, 800-token answers, 3 runs | 139, 128, 130 tok/s | 97, 90, 89 tok/s |
| GPU clock during those runs (throttled, so a floor) | 1.2-1.3 GHz | 1.2-1.5 GHz |
| Memory at 131,072 tokens of context (model, cache and work buffers) | 19 GB | 29 GB |

Qwen3.6 has fewer active parameters than GPT-OSS 120B but is no faster: 30 of its 40 layers are linear attention (Gated
DeltaNet), whose small steps for each token take time that fewer parameters do not save. This file has no MTP head.

### GLM-5.3-Flash

Z.ai's GLM-5.3-Flash (320B parameters, about 18B active per token) runs in Strata as
[Project Maya's](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF) Maya-S24: 2-bit routed experts, 94.7 GB
in 3 files; its model card rates it at 97.7% of the full model's zero-shot accuracy. Setup does not install it; by hand:

```sh
D=../Strata-data/models/GLM-5.3-Flash-Maya-S24; mkdir -p $D
R=https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF/resolve/f16ee4ba50d8fd2bb726ea1862ca6ca14ba188da
for f in 1 2 3; do aria2c -x16 -s16 -c -d $D "$R/Maya-S24/GLM-5.3-Flash-Maya-S24-IQ2_XXS_S-0000$f-of-00003.gguf"; done
aria2c -c -d $D "$R/vision/GLM-5.3-Flash-vocab.gguf"                                    # 9 MB, the tokenizer
.venv/bin/python tools/strata_tokenizer.py --gguf $D/GLM-5.3-Flash-vocab.gguf --out ../Strata-data/packs/glm-5.3-flash-maya-s24
```

If a part stops with a 403 near its end (Hugging Face's signed link expired), run the same line again: it continues.
Then a run config as for GPT-OSS, with `"--gguf", "<the -00001-of-00003 file>", "--max-context", "32768"`.

What Strata does for it: its tokenizer (`pre` = `glm4`) matched GLM-5.3-Flash's own on 2,179 strings (0 differences);
the server reads its thinking and its tool calls (GLM's `<arg_key>`/`<arg_value>` form) and stops a reply at
`<|user|>`, `<|observation|>` or `<|endoftext|>`. GLM always opens its thinking: the chat's efforts map to its template's
low / high / max ("Off" and "Low" are its low, which often writes no thinking at all).

| Measured on the test Mac, 2026-10-10, *High Power* | |
|---|---|
| A question, thinking (`high`), a tool call and its result (OpenAI API) | correct |
| Decode, short answers (127-346 tokens) | 29-32 tok/s |
| Decode, 900-word answers (930-1,220 tokens) | 13-26 tok/s (25.6, 13.4, 16.0, 19.9) |
| Decode, one 17,940-token answer (a 319-token prompt, 14.5 min) | 20.5 tok/s on average |
| A follow-up after a tool's result | read only its new part (199 of 217 prompt tokens reused) |
| Engine memory at a 32K context | 88.6 GB of Metal's 107.5 GB: weights 85.6, compute 2.6, cache 0.3 |
| Load | about 60 s |

These runs were not on a quiet Mac: Docker Desktop held about 40 GB and swap was full (34 of 35 GB), and the GPU ran
at 400-840 MHz, so read them as a floor. The model's own measurements: 13.5 tok/s on a 24 GB PC graphics card with
the rest in RAM. Its cache is small (0.3 GB at 32K), so larger contexts should fit, but only 32K was run.

### Pictures from a prompt (Nucleus-Image)

Strata makes pictures with [Nucleus-Image](https://huggingface.co/NucleusAI/Nucleus-Image) (a 17B mixture-of-experts
image model, Apache-2.0) on its own engine: `strata-image` (`metal/image/`, ggml's Metal backend, built by setup next
to `strata-metal`). It runs the three parts in turn, each freed before the next: the text encoder (Qwen3-VL-8B through
Strata's llama.cpp, the hidden state of layer 28), the DiT (64 experts a layer, expert-choice routing, both CFG
branches in one graph with their experts in one batched matmul) and the VAE decoder. The weights are memory-mapped:
they are the files' own pages, clean and file-backed, so macOS can drop them under pressure and read them again.

The model files, made once (~50 GB downloaded, ~21 GB kept): the DiT and the VAE from the BF16 safetensors of
[mlx-nucleus-image](https://huggingface.co/treadon/mlx-nucleus-image) (`dit/`, `vae/`), the text encoder from
`NucleusAI/Nucleus-Image`'s `text_encoder/` with llama.cpp's converter (PyTorch: `.venv-mtp` has it):

```sh
M=<a folder>; L=build-metal/_deps/llama_cpp-src-*                    # the llama.cpp setup fetched
.venv/bin/python tools/image/nucleus_gguf.py --src $M/mlx/dit --out $M/nucleus-dit-q4_0.gguf --experts q4_0   # 11.8 GB
.venv/bin/python tools/image/nucleus_gguf.py --vae --src $M/mlx/vae --out $M/nucleus-vae.gguf                 # 0.1 GB
.venv-mtp/bin/python $L/convert_hf_to_gguf.py $M/hf/text_encoder --outtype q8_0 --outfile $M/qwen3vl-8b-q8_0.gguf  # 8.7 GB
```

Then in the run config `"image": {"dit": "$M/nucleus-dit-q4_0.gguf", "encoder": "$M/qwen3vl-8b-q8_0.gguf", "vae":
"$M/nucleus-vae.gguf", "need_gb": 16}` (full paths) and start Strata again: `/health` says `"image_generation": true`,
`POST /v1/images/generations` (OpenAI's form: `prompt`, `size` "WxH" in steps of 16 from 256 to 2048, `n` 1; plus
`steps` (30), `guidance_scale` (4), `seed`, `negative_prompt`; at most 1536x1024 pixels in all) answers with the picture as `b64_json` and the run's
timings in `strata`, and the web chat makes one with `/image <prompt>` or "draw ..." (kept in the chat's history; not
sent to the chat model).

- **One process a picture.** Strata starts `strata-image` for each picture and waits for it to exit: a crash or an
  out-of-memory there leaves the chat model alone, and the exit is the proof its memory is back. The prompts go to it
  as files in a private temporary folder (not on its command line, which every user of the Mac can read); it ends
  itself when Strata's end of its stdin closes, so a server that dies does not leave a picture running.
- **Pictures only.** With `"lazy_load": true` as well, Strata starts without loading the chat model; it loads only if
  a chat request comes. On the test Mac `run-image.sh` starts such a Strata.
- **One model at a time when memory is short.** A picture takes the server's turn as a chat request does (chat
  requests wait for it). When less than `need_gb` GB is free (psutil's "available"), the chat model is unloaded first,
  after the requests running on it have ended; the next chat request loads it again (GLM-5.3-Flash: ~33 s).
- **Errors.** A bad request is a 400, before anything is unloaded. No turn within `queue_s` (600) seconds, or the chat
  model busy while memory is short: 503 with Retry-After; memory still short after the unload: 503 too. strata-image
  failing: 500 with its own message (or its last stderr). A client that hangs up (or the page's Stop): the process is
  stopped (SIGTERM, then SIGKILL 5 s later) and the turn is held until it has exited; past `timeout_s` (1800) the same,
  and 504. Every picture's end is in the server's log, failures and stops included.
- **Measured** (M5 Max, 128 GB, 2026-10-11; the GPU clock moved between 600 and 1600 MHz from run to run, so read
  timings as ranges, and compare engines only in interleaved runs):

| | step, 512x512 | step, 1024x1024 | 20 steps at 1024x1024, in all | memory the process holds | weights |
|---|---:|---:|---:|---:|---|
| strata-image (Q4_0 experts, Q8_0 encoder) | 0.38-0.5 s | 1.7-2.1 s (quiet), 2.5-3.2 s (busy) | 45 s | 0.4 GB + 0.3 / 1.5 GB of activations | 11.8 GB mapped, reclaimable |
| for comparison: the MLX worker (nucleus-mlx), 4-bit, experts in memory | 0.48 s | 1.5-2.1 s | 46-56 s | 14.6-18.2 GB | inside that |
| the MLX worker, 4-bit, experts streamed from the SSD | 0.9-1.1 s | 1.8-2.1 s | 54-64 s | 9.5-10.3 GB | streamed |

  A step is both CFG branches. CPU time for a 1024x1024 picture: 4 s (strata-image), 36-77 s (MLX, mostly PyTorch's
  text encoder). 1536x1024: 9 s a step, 2.6 GB of activations, 13 s to decode. The size is capped there: 2048x2048 ran
  at 40 s a step with 12.9 GB of activations (the expert-choice routing's one-hot grows with the square of the pixel
  count). The VAE needs 4.8 GB of scratch for its 1.8-2.7 s at 1024x1024, after the DiT is freed.

  What made the 1024x1024 step 1.6x faster (a per-op profile: each node of the graph alone, synchronized): flash
  attention on ggml's tensor-API kernel, which it takes only when the keys are a multiple of 64 (4119 keys took a slower
  kernel: the keys are padded and masked); RoPE as one `ggml_rope_multi` (MROPE sections 8/28/28, the converter
  de-interleaves the query and key projections: `rope_layout` "neox" in the GGUF) instead of a dozen small operations;
  the routed experts of the two branches as `[D, C, E, 2]` slabs; the routing scores gathered, not multiplied out.

  Quality, LPIPS against the float32 port's picture (same prompt, noise and settings; 8 prompts, 512x512, 30 steps;
  lower is closer, another seed is ~0.5): strata-image 0.174 on average, the MLX worker 0.199. The paired t-test
  (`image-model-testing` skill in the bible, `img-check paired`) finds no difference shown between the two (t = -0.56),
  nor between flash and float32 attention (t = 1.01); with 3 prompts the comparison had pointed both ways. The checks:
  `tools/image/check_dit.py` (every DiT block against the port), `strata-image --check-text` (the encoder against
  PyTorch: 1.1% relative, Q8_0), `--check-vae` (0.24-0.27% against the port's decoder).

### Pictures from a prompt (HunyuanImage-3.0)

`strata-image` also runs [HunyuanImage-3.0-Instruct-Distil](https://huggingface.co/tencent/HunyuanImage-3.0-Instruct-Distil)
(Tencent: an 80B mixture of experts, 64 experts a layer, top 8, ~13B active a token). It is one transformer for text
and picture: the prompt and the image tokens are one sequence, so there is no separate text encoder. The prompt's
keys and values are computed once. Each of the 8 steps then runs the 3 conditioning tokens (timestep, guidance,
timestep_r) and the h x w image tokens. CFG is distilled into the model: one pass a step. The 3D VAE decodes the
last of its 4 frames, as the pipeline does.

**License.** The [Tencent Hunyuan Community License](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0/blob/main/LICENSE)
does not apply in the European Union, the United Kingdom or South Korea. Above 100 million monthly active users,
you need Tencent's permission. Read it before you use the model.

The model file, made once: the converter streams the 32 BF16 shards (169 GB) one at a time, and deletes each shard
after it is converted. The disk needs room for one shard (5.4 GB) and the output (54 GB, Q4_0 experts). This takes
about 2 hours at 20 MB/s. `make pull MODEL=HUNYUAN` does both steps below (also `MODEL=NUCLEUS`, `GLM-5.3`,
`GPT-OSS-120B`, `GPT-OSS-120B-DFLASH`, `GPT-OSS-20B`, `QWEN3.6-35B`: `tools/pull_model.py`).

```sh
M=<a folder>
.venv/bin/python tools/image/hunyuan_gguf.py --out $M/hunyuan-image3-q4_0.gguf --work $M/shards   # 54.4 GB
curl -L -o $M/tokenizer.json https://huggingface.co/tencent/HunyuanImage-3.0-Instruct-Distil/resolve/main/tokenizer.json
```

Then in the run config: `"image": {"hunyuan": "$M/hunyuan-image3-q4_0.gguf", "tokenizer": "$M/tokenizer.json",
"need_gb": 70}`. The defaults change to the model's own: 8 steps and `guidance_scale` 2.5. `size` goes to the nearest
of the model's 37 trained sizes (1024x1024, 1280x768, 720x1280, ...). The server builds the token ids
(`tools/image/hunyuan_tokens.py`: the "instruct" template, the "en_unified" system prompt, no think or recaption
stage). The prompt is plain text: a prompt that spells `<img>` does not become that token. Alone:

```sh
.venv/bin/python tools/image/hunyuan_tokens.py $M/tokenizer.json "draw sweet cat" 1024x1024 > tokens.txt
build-metal/metal/strata-image --hunyuan $M/hunyuan-image3-q4_0.gguf --tokens tokens.txt --seed 42 --out cat.png
```

- **Measured** (M5 Max, 128 GB, 2026-10-12, GPU at a median 1160 MHz), 1024x1024 in all: 88 s. The prompt pass took
  17 s, mostly reading the mapped weights from the SSD. The 8 steps took 4.1-5.3 s each, with 1.5 GB of
  activations. The decode took 31 s, of which 17 s is the VAE itself. The process held 0.28 GB (peak footprint).
  The 54 GB of weights are the file's own pages, which macOS can drop and read again.
- **The VAE** is 3D: each Conv3d is 3 temporal taps of 2D convolutions. The three taps run as one kernel, so each
  frame's im2col is made once. The kernels are F16, as the pipeline decodes under fp16 autocast. This made the decode
  5.4x faster (92.8 s to 17.1 s), with at most 1 of 255 in a pixel changed.
- **Checked against Tencent's own modules** (float32, CPU), each part alone on the reference's input. The 80B model
  does not fit in memory whole, so `tools/image/hunyuan_ref.py dump` runs the parts, `strata-image --check-hunyuan`
  runs the same parts, and `hunyuan_ref.py compare` compares them. The experts are the GGUF's, dequantized. Relative
  error: the input tokens 2.2e-4, layer 0 3.2e-4, layer 31 4.4e-3, the image head 1.7e-4, the VAE 2.1e-4. The
  tokens and 2D RoPE positions are the reference tokenizer's and `build_2d_rope`'s, exactly, for 5 prompts and sizes.
- **Open:** Q4_0 experts have 8.6% relative error a weight matrix. With the BF16 model not runnable here, its effect
  on the pictures is not measured; text in pictures has single-letter errors ("STRASA" for "STRATA").

### Videos from a prompt (Wan 2.2 A14B)

Strata makes short clips with [Wan 2.2 T2V A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) (Apache-2.0): two
14B experts, one for the high-noise steps and one for the low-noise steps. It runs on
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)'s `sd-cli` (ggml, Metal), one process a clip,
built at a pinned commit with one patch, `metal/video/sdcpp-metal-conv3d.patch`. Metal has no `IM2COL_3D`, so the
Wan VAE's Conv3d took ggml's direct kernel. The patch runs it as 2D convolutions, one per temporal tap: decoding 17
frames went from 389 s to 30 s, and frame 8 matched the old output at 40.7 dB PSNR.

```sh
make pull MODEL=WAN2.2      # the 2 experts (Q4_K_M), UMT5 (Q8_0), the VAE, the Lightning LoRAs: 29 GB; builds sd-cli
```

Then `"video": {"exe": "<Strata>/build-sdcpp/build/bin/sd-cli", "low": "$D/Wan2.2-T2V-A14B-LowNoise-Q4_K_M.gguf",
"high": "$D/Wan2.2-T2V-A14B-HighNoise-Q4_K_M.gguf", "t5": "$D/umt5-xxl-encoder-Q8_0.gguf", "vae":
"$D/wan_2.1_vae.safetensors", "need_gb": 35}` in the run config. A clip takes minutes, so the API is a job, as
OpenAI's video API is (the asynchronous request-reply pattern):

- `POST /v1/videos` (`prompt`, `size` up to 1280x720, default 832x480, `seconds` up to 10, default 2; plus `steps`
  (10), `high_noise_steps` (8), `guidance_scale` (3.5), `seed`, `negative_prompt`) answers 202 at once, with the job
  and a `Location` to poll.
- `GET /v1/videos/{id}` gives `status` (queued, in_progress, completed, failed) and `progress` (0-100).
  `GET /v1/videos/{id}/content` gives the clip: H.264 MP4 (`video/mp4`, which QuickTime, Finder and Photos play) when
  ffmpeg is installed (`brew install ffmpeg`), else sd-cli's own WebM (`video/webm`), which only browsers play. `DELETE /v1/videos/{id}` stops it or removes it.
  `GET /v1/videos` lists the jobs.
- A job waits for the server's turn and unloads the chat model when memory is short, as a picture does. A finished
  clip is kept one hour (`keep_s`). The jobs live in the server's memory: a restart forgets them.
- In the web chat, `/video <prompt>` makes a 2 s clip, shows its progress and keeps the clip in the chat.

Measured (M5 Max, 128 GB, 2026-10-12; single runs, and the GPU clock moved between 600 and 1100 MHz):

| 832x480, 17 frames, 2 + 2 steps | sampling | VAE decode | in all | peak footprint |
|---|---:|---:|---:|---:|
| weights in memory (the default) | 117 s | 30 s | 254 s | 28.2 GB |
| `"args": ["--diffusion-fa", "--params-backend", "diffusion=disk"]` | 148 s | 26 s | 334 s | 18.2 GB |

**Faster: the Lightning LoRAs.** [lightx2v's Wan2.2-Lightning](https://huggingface.co/lightx2v/Wan2.2-Lightning)
(Apache-2.0) distils Wan 2.2 to 4 steps without CFG: one pass a step, 4 passes for a clip instead of 36. `make pull
MODEL=WAN2.2` gets them (their integer `alpha` tensors become float32: `tools/video/lora_alpha_f32.py`, as sd-cli
loads no integer tensor). With `"loras": {"dir": "$D/loras", "low": "lightning-low", "high": "lightning-high"}` in
`"video"`, a job's defaults are 2 + 2 steps and guidance 1. A prompt may not hold `<lora:...>` (sd-cli loads the file
it names): only the config chooses.

| 832x480, 33 frames (2 s) | steps | in all |
|---|---:|---:|
| the defaults without LoRAs (18 steps, CFG 3.5) | 36 passes | 2907 s |
| Lightning, 2 + 2 steps, guidance 1 | 4 passes, 75-116 s each | 430 s |

The frames passed `vid-check frames` (bible: video-model-testing). Measured and ruled out: the LoRAs at run time cost
nothing visible (a pass without them: 89-110 s); merging them into the Q4_K weights (`--lora-apply-mode immediately`)
aborts in sd-cli; llama.cpp's newer ggml (`SD_USE_UPSTREAM_GGML`) and `--disable-segmented-compute` were no faster.
In every run the GPU sat at 600-750 MHz and 40% busy while the Mac had 44 GB in swap: memory pressure from other
programs, not the model, sets the pace. Free memory first.

Through the API: 287 s for the same clip. Both experts stay in memory by default. With `diffusion=disk`, only the
expert in use is: 10 GB less at the peak, and 31% slower. **Flash attention is needed** (`--diffusion-fa`, the
default here): without it, every frame came out white. A step took 30-100 s; the defaults (18 steps, 33 frames) take
roughly 10-30 minutes a clip.

### EAGLE3 draft model (`--eagle3`)

`strata-metal --eagle3 <gguf> --spec N` drafts with an EAGLE3 model (it reads 3 of the target's layers) instead of an MTP
head; llama.cpp drives both the same way. It is off: on GPT-OSS with
[eagle3-gpt-oss-120b-Q8_0.gguf](https://huggingface.co/ggml-org/gpt-oss-120b-GGUF) (849 MB) it was slower than no draft.
`metal/bench/ab.py`, 4 alternating rounds, 3 prompts, 256 tokens, greedy:

| | No draft | `--eagle3 --spec 3` |
|---|---:|---:|
| Decode, median of the per-run medians | 31.5 tok/s | 23.2 tok/s (0.74x, slower in 10 of 12 runs) |
| Drafts accepted | | 34-52% |

Likely why: this EAGLE3 file keeps the full 201K-token output layer (no reduced draft vocabulary), and checking 4 tokens of
a mixture-of-experts model reads up to 4 tokens' worth of experts. On the 3,676-token prompt the greedy answer also differed
from the one without a draft (the 4-token check rounds differently). With EAGLE3 the engine rolls back one cell more than a
checkpoint holds: llama.cpp keeps EAGLE3's one-cell lag in a checkpoint only for a recurrent model, and without it the
draft would skip a position (`metal/strata_metal.cpp`, `rollback`).

## Measured

On the test Mac: Q2_0, 32K context, through the server's OpenAI API, greedy, thinking off, a fresh prompt each run,
medians of 3, 2026-10-06, with other programs running (the answer speed moved by ±3 tokens/s between runs).

| Prompt | Reads the prompt | Writes the answer | With `--mtp on` |
| ---: | ---: | ---: | ---: |
| 25 tokens (a story) | - | 16.2 tok/s | 17.2 tok/s |
| 3,686 tokens (code) | 241 tok/s | 12.8 tok/s | **21.6 tok/s** |
| 26,051 tokens (code) | 225 tok/s | 17.0 tok/s | 19.4 tok/s |

- The `--mtp on` column was measured on a later engine build (it also reads prompts faster: 272 and 252 tok/s), so
  its ratios to the other column mix two changes. MTP alone, on the same build: 1.23-1.50x on 3 short chat prompts,
  with 64-92% of drafts accepted; prose gained least.
- After the 26K prompt, a follow-up message started answering in 0.3 s (0.7 s with MTP).
- One picture (640×240, a 196-token prompt) was read and answered in 3.4 s in all.
- Two requests at once with MTP: 17.2 tok/s together against 18.7 one after the other (batch slots decode without
  drafts). llama.cpp's `batched-bench` without MTP: 13.2 tok/s for one sequence, 22.5 for two, 29.7 for four.
- No large overhead over llama.cpp was apparent: `llama-bench` on the same file gave 16.9 tok/s output.

### The models compared

On the test Mac, through the OpenAI API, thinking off, *High Power* (details in each model's section):

| | GPT-OSS 120B | Qwen3.8-Flash-Next Q2_0 | Qwen3.8-Flash-Next IQ3_XXS | GLM-5.3-Flash Maya-S24 |
|---|---:|---:|---:|---:|
| Parameters, active per token | 117B, ~5B | 180B, ~6B | 180B, ~6B | 320B, ~18B |
| Download | 63 GB | 66 GB | 76 GB | 95 GB |
| Engine memory | the 63 GB file + cache | ~60 GB at 128K (estimate) | 65 GB at 128K | 89 GB at 32K |
| Decode, 900-word answers | 93-100 tok/s | 60-69 tok/s (`--mtp on`) | 46-48 tok/s (`--mtp on`) | 13-26 tok/s\* |
| Decode, short answers | 93-103 tok/s | 92-102 tok/s (`--mtp on`) | 73-92 tok/s (`--mtp on`) | 29-32 tok/s\* |
| Measured | 2026-10-08 | 2026-10-08 | 2026-10-08 | 2026-10-10 |

\* With Docker using about 40 GB and swap full; the GPU ran at 400-840 MHz against 1,300-1,620 for the others, so this
column is a floor. GLM reads about 3x the active parameters per token of the others, so it would be slower on a quiet Mac
too.

### IQ3_S compared with Q2_0

IQ3_S (3.5 bits) has 46% more expert weights than Q2_0 (54.8 GB against 37.6 GB in the first file; the 28.8 GB
n-gram file is the same), so it was expected to write about 30% slower. Measured on the test Mac, 2026-10-08: the same
settings for both (128K context, f16 cache, `--mtp on`, 2 batch slots), greedy, thinking off, through the OpenAI API,
the two models loaded in turns (IQ3_S, Q2_0, IQ3_S, Q2_0), medians of 4 runs, with a Rust build, a virtual machine
and Docker running (load average 10-40).

| | Q2_0 | IQ3_S | IQ3_S / Q2_0 |
|---|---:|---:|---:|
| Writes the answer, short prompt (33 tokens) | 27.3 tok/s (25.7-29.7) | 24.3 tok/s (21.1-27.0) | 0.89 |
| Writes the answer, 3,775-token prompt | 24.4 tok/s (18.6-27.2) | 22.6 tok/s (22.3-24.0) | 0.93 |
| Reads the 3,775-token prompt | ~367 tok/s | ~346 tok/s | 0.94 |
| MTP drafts accepted | 60-64% | 53-58% | |
| Engine memory at a 128K context | | ~74 GB | |

- IQ3_S was only 7-11% slower in these runs despite 46% more expert bytes, so reading the expert weights is not most
  of a token's time; these runs do not show which costs are. Its drafts were accepted a little less often, a possible
  part of the gap.
- Q2_0's 18.6 tok/s run came when the load average reached 40; without it, the long-prompt range is 24.3-27.2.
- So on this Mac IQ3_S costs about a tenth of the speed and 16 GB more memory; its quality was not measured here
  (the model card rates it closest to the full model).

### IQ3_XXS compared with Q2_0

IQ3_XXS (3-bit) has a 47.0 GB first file against Q2_0's 37.6 GB (the n-gram file is the same: setup can share it).
Measured on the test Mac, 2026-10-08, *High Power*, the same settings for both (128K context, f16 cache, `--mtp on`, 2
batch slots), thinking off, through the OpenAI API, each model loaded on its own, an hour apart:

| | Q2_0 | IQ3_XXS |
|---|---:|---:|
| Writes the answer, 900-word answers | 63-69 tok/s | 46-48 tok/s |
| Writes the answer, short answers | 97-102 tok/s | 73-92 tok/s |
| MTP drafts accepted | 49-50% | 49% |
| GPU clock during the long answers | 1,300-1,620 MHz | 930-1,120 MHz |
| Engine memory | | 65.2 GB |

- IQ3_XXS was about 30% slower here, but the GPU ran at a lower clock during its run, so part of that gap is the Mac's
  state rather than the model: the interleaved IQ3_S/Q2_0 comparison above, with 46% more expert bytes, found 7-11%.
  Read it as 10-30% slower, for better quality than Q2_0 (not measured here).

### Why the speed changes

Measured on the test Mac, 2026-10-08, IQ3_S with `--mtp on` and `--spec 3`, the engine driven directly, one 256-token
answer to the same prompt again and again:

- In the first minute after a pause: 59-65 tok/s, about 44 ms per verify step. After a few minutes: 30-36 tok/s, about
  82 ms per step. The accepted drafts were the same total every time (165 of 273).
- [macmon](https://github.com/vladkens/macmon) in the slow phase: GPU 600-830 MHz, 40-50% busy, 8-11 W, die
  59-69 °C. A fast phase was not recorded with macmon. In both phases the engine's main thread spent most of its time
  (88% fast, 91% slow) waiting for the GPU to finish.
- Tried without a measurable change in the slow phase: fewer CPU threads and no spin-waiting (`--threads 1/4`,
  `--poll 0`), reading the n-gram file into the cache first, stopping a Time Machine backup and a busy System Settings
  storage scan. `GGML_METAL_NO_RESIDENCY=1` was tried only in a fast phase (56-66 against 59-65 tok/s).
- `--spec 3` stayed the default: `--spec 2` was about the same or slower, `--spec 4` slower.
- Where a verify step's time goes (`STRATA_PHASES=1` prints it per request to the engine's log; it adds two
  synchronizations, so use it to compare phases, not for speed): in a fast phase, about 51 ms a step = 41 ms for the
  target model to check 4 tokens, 7-8 ms for the 3 draft steps (each waits for the GPU), 2 ms for the MTP catch-up,
  under 1 ms sampling. So the draft round trips are at most about 15% of a step: removing them all would gain less than
  that, and most of their time is the draft head's own work (it has the full 248K-token output layer).
- Stopping the drafts early when the draft head is unsure (`p_min` 0.5) was 5% slower and changed the answer's tokens.
- `llama-bench` on the same file: a forward pass of 1, 2, 3, 4 and 8 tokens took about 21, 28, 34, 31 and 42 ms in
  one run, so checking 4 drafted tokens costs about 1.5x one token; in another run, minutes later, a 4-token pass took
  38 and then 169 ms. The engine's 41 ms for its 4-token check and `llama-bench`'s 31 ms came from different runs, so
  they do not show an overhead in the engine.
- What the engine's check needs beyond `llama-bench`'s pass, timed in one process against the same model: logits for
  all 4 tokens about +1-4%, the hidden rows the draft head reads about +0-3%, the recurrent-state snapshots that let
  rejected drafts be taken back about +6-9%. All three are needed for MTP. (The draft head's context builds a
  throwaway CPU thread pool for each call; in a profile that was 0.04% of the time.)
- The llama.cpp update of 2026-10-08 (55 upstream commits, among them few-row matrix kernels and a Metal fusion fix)
  was faster in two A/Bs: +11.5% (4 rounds, p = 0.46) and +8.3% (6 rounds, p = 0.054; 15 of 18 runs faster), with
  the same tokens every run. Which upstream change gives it was not isolated.
- Strata's two kernel patches (below) are now off by default: on the new commit, the build without them was 5.7%
  faster in one A/B (6 of 6 runs, the same tokens); a second A/B was swamped by a clock drop and showed nothing either
  way. They had not been measurably faster on the earlier commit either.
- Both changes together against the engine as it was that morning (6 alternating rounds, 3 prompts): the new one was
  faster in 12 of 18 runs, by 6% on average (paired geometric mean; +11.5% by medians), with the same tokens. During
  the run both fell from about 38 to about 17 tok/s as the GPU's clock dropped, so the gain is small next to the
  swings.

### Compared with MLX (mlx-lm)

Would Apple's MLX run this model faster? Measured on the test Mac, 2026-10-07. The same Q2_0 weights were converted to
MLX: the 2-bit experts copied bit for bit, the other tensors requantized one bit higher, the 28 GB n-gram table at 5
bits. They ran in mlx-lm on MLX 0.32.3 with the community port of this model
([mlx-lm#1788](https://github.com/ml-explore/mlx-lm/pull/1788), not merged as of that date). Each server ran alone, in
turn, on the same prompts: greedy, thinking off, MTP off, through its OpenAI API.

| | Strata (llama.cpp Metal) | mlx-lm (MLX) |
|---|---:|---:|
| Writes the answer, best run | 48.3 tok/s | 41.5 tok/s |
| Writes the answer, worst run (a virtual machine running) | 24.7 tok/s | 17.1 tok/s |
| Reads a 3,290-token prompt | 364-895 tok/s | 286-493 tok/s |
| Memory in use | ~35 GB of weights; the n-gram table stays in the file | ~81 GB, all of it in memory |
| Start | ~35 s | ~160 s |

- The load from other programs was not controlled and moved both engines by up to 2x (an Xcode build pulled MLX
  down to 15-19 tok/s while a GPU benchmark kept its full speed), so read the table as ranges, not as a ranking.
- The answers are not the same: 3 of 12 greedy answers matched; the rest parted after 2 to 58 tokens (the non-expert
  weights are requantized for MLX, and the two backends round differently; which matters more was not isolated).
- mlx-lm's server slowed down on repeated requests (40.7 to about 20 tok/s) until its prompt cache was turned off
  (`--prompt-cache-size 0`).
- Tried on the MLX side: the n-gram lookup on the GPU from one table, and fused hyper-connection steps (same tokens,
  no change beyond the noise); 8-bit hyper-connection weights were slower (26.3 against 32.7 tok/s) and changed the
  answer.
- So MLX was not faster here and needs more than twice the memory: Strata stays on llama.cpp, and the MLX conversion
  is not part of Strata.

## How it fits together

- `metal/strata_metal.cpp`: the engine. It speaks the CUDA engine's line protocol (`GEN`/`GENI`, `T`, `PP`, `RESUME`,
  `DONE`, `STOP`, `SAVE`/`RESTORE`, and `BGEN`/`BT`/`BDONE`/`BADM`/`BSTOP` for batch slots), so every server feature
  works through it. It adapts the protocol instead of porting the CUDA engine, whose kernels and expert cache are CUDA
  code; llama.cpp already runs this model on Metal.
  - Conversation reuse goes back to a checkpoint of the recurrent state, taken before each `<|im_start|>`. It never
    uses a state that does not match: tokens are compared by id, and picture cells by the picture's hash.
  - Pictures get the CUDA engine's M-RoPE positions.
- `metal/setup_mac.py`: setup's Mac steps. `setup.py` runs it by itself on macOS; it changes none of setup's other steps.
- `metal/mtp_gguf.py`: the MTP head for llama.cpp, from the checkpoint, checked by its SHA-256.
- `serve/telemetry.py`: the Monitor's GPU load comes from `ioreg`, its memory limit from Metal, its power from
  IOReport's energy counters and the temperature from the chip's die sensors, all without root. These are private
  macOS interfaces: if a macOS update changes them, the tiles show "–" instead of a wrong number.

## For developers

```sh
make build                                   # build-metal/: the engine and strata-vision
make test                                    # the tests that need no GPU and no model
make test-engine TEST_GGUF=<a small .gguf with <|im_start|>, e.g. LiquidAI LFM2-350M Q8_0>
```

llama.cpp comes at a pinned commit (`-DSTRATA_LLAMA_DIR=<checkout>` builds offline). Strata's own Metal kernels are
patches on that commit in `metal/patches/`, off by default; `-DSTRATA_METAL_KERNEL_PATCHES=ON` applies them. To A/B
the two:

```sh
make build-ab                                # build-metal-a (plain) and build-metal-b (patched)
make ab AB_GGUF=<shard 1> AB_PROMPTS=<json list of token-id lists>
```

`metal/bench/ab.py` runs both builds in turns and exits 1 when their greedy tokens differ. Results on the earlier pin
(2026-10-06), on Q2_0 on the test Mac:

- `0001-metal-fuse-scale-unary`: fuses a scale into the following unary op; the same tokens on the tested prompts, no
  measurable speedup.
- `0002-metal-short-row-mat-vec`: a mat-vec kernel for short rows; 1.8x faster on its own (15.2 vs 27.3 µs), +1.1%
  for the whole model, within the run-to-run noise. `GGML_METAL_MV_SHORT_DISABLE=1` turns it off.

`metal/test_strata_metal.py` speaks the protocol to the binary: a diverging conversation must give a fresh engine's
tokens, an extended prompt must read only its new part, a different picture with the same ids must not be reused, two
batch slots must each give what they give alone, and STOP, BSTOP, SAVE/RESTORE and every refusal must keep both sides
in step. Four of these checks were mutation-tested.
