<h1 align="center">Strata for Mac</h1>

**English** · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md) · [Português](README.pt-BR.md)

<p align="center"><b>Run 100-billion-parameter AI models on your own Mac</b><br>
Apple Silicon · tested on an M5 Max with 128 GB · free and open source</p>

> **This is an independently maintained fork of [Strata](https://github.com/Niko1221/Strata) for Apple Silicon Macs
> only**, maintained by [dennis-akimov](https://github.com/dennis-akimov). It adds a Metal engine and the Mac setup.
> The original's Windows and Linux engines (NVIDIA CUDA, AMD HIP, Intel SYCL) are still in this repository as they were
> at the fork point, but they are not built, tested or updated here.
> **For a Windows or Linux PC, use the original: [github.com/Niko1221/Strata](https://github.com/Niko1221/Strata).**

Strata runs large AI models on your own computer: they chat, write code, read pictures and work with your apps and
coding agents through the same APIs as cloud services. The model runs on your Mac; what leaves it is up to you (the
model downloads, and any app, tool or MCP server you connect).

It also makes pictures from a text prompt, on its own image engine ([Nucleus-Image](docs/MACOS.md#pictures-from-a-prompt-nucleus-image) or [HunyuanImage-3.0](docs/MACOS.md#pictures-from-a-prompt-hunyuanimage-30)). And short videos from a prompt ([Wan 2.2](docs/MACOS.md#videos-from-a-prompt-wan-22-a14b)).

## How fast is it?

Measured on one MacBook Pro with an M5 Max (40-core GPU, 128 GB), macOS 26.4, in the *High Power* energy mode,
2026-10-08, thinking off. A token is about ¾ of a word.

| Model | 900-word answers | Short answers | Model file in memory |
| --- | ---: | ---: | ---: |
| **[GPT-OSS 120B](docs/MACOS.md#gpt-oss-120b)** (MXFP4) | 93-100 tokens/s | 93-103 tokens/s | 63 GB |
| **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) Q2_0**, with `--mtp on` | 60-69 tokens/s | 92-102 tokens/s | 38 GB |
| **Qwen3.8-Flash-Next IQ3_XXS**, with `--mtp on` | 46-48 tokens/s | 73-92 tokens/s | 47 GB |
| **[GLM-5.3-Flash](docs/MACOS.md#glm-53-flash) Maya-S24**\* | 13-26 tokens/s | 29-32 tokens/s | 86 GB |

\* Measured 2026-10-10 while Docker used about 40 GB and swap was full (the GPU at a low clock), so a floor; GLM also reads about 3x the active parameters per token of the others.

Pictures and videos on the same Mac (2026-10-11/12):

| Model | Output | Time | Memory |
| --- | --- | ---: | ---: |
| **[HunyuanImage-3.0](docs/MACOS.md#pictures-from-a-prompt-hunyuanimage-30)** (80B, Q4_0 experts) | 1024x1024 picture, 8 steps | 88 s | 0.3 GB, its 54 GB file mapped |
| **[Nucleus-Image](docs/MACOS.md#pictures-from-a-prompt-nucleus-image)** (17B, Q4_0 experts) | 1024x1024 picture, 20 steps | 45 s | 2 GB, its 12 GB of files mapped |
| **[Wan 2.2 A14B](docs/MACOS.md#videos-from-a-prompt-wan-22-a14b)** + Lightning LoRAs | 832x480 video, 2 s (33 frames), 4 steps | 430 s | 30 GB at its peak |
| Wan 2.2 A14B without the LoRAs | the same video, 18 steps | 2907 s | about 28 GB |

The video times were measured while Docker held 76 GB and macOS was swapping (the GPU at 600-750 MHz), so a floor.

The Qwen numbers use its MTP draft layer, which is opt-in (`./setup.sh --setup --mtp on`, about 12 GB more disk).
The context's cache comes on top of the model file (a few GB; more for long contexts). In the *Automatic* energy mode
the same Mac wrote long answers 2-3x slower. Other Macs were not measured. Every number and how it was measured:
[docs/MACOS.md](docs/MACOS.md#measured).

## What you need

| | |
| --- | --- |
| **Mac** | Apple Silicon (M1 or newer). Only one M5 Max with 128 GB was tested; other chips and sizes are expected to work but were not tried. |
| **Memory** | Enough for the model file plus its cache, inside what macOS lets the GPU use (on the tested 128 GB Mac: 107.5 GB). Setup's check counts 64 GB as the minimum. `make check` estimates what fits on yours; see [Which model?](#which-model) |
| **Disk** | The model's download plus a few GB (70-100 GB per model), on the internal SSD. |
| **Software** | Apple's Command Line Tools (`xcode-select --install`, wait until it finishes) and Python 3.10 or newer (if it is missing, setup installs it with Homebrew when Homebrew is there). Tested on macOS 26.4. |

## Install

In Terminal:

```sh
git clone https://github.com/dennis-akimov/Strata.git
cd Strata
make check                 # what this Mac can run; installs nothing
make pull MODEL=Q2_0       # builds the engine and downloads the model (66 GB)
make run                   # loads the model and opens http://127.0.0.1:8080 when it is ready
```

`make pull` asks a few questions (context size, pictures); Enter takes the recommended answer. If the download stops,
run it again: it continues where it stopped. Loading takes a minute or so each start. `make run` keeps Terminal busy
while Strata runs; press Ctrl+C to stop it, or use `make start` / `make stop` to run it in the background. Every
option, the limits and the warnings: **[docs/MACOS.md](docs/MACOS.md)**.

**Updating:** `git pull`, then `make run` (setup rebuilds the engine when the sources changed; the model stays).
**Where things are:** models in `Strata-data/` next to the `Strata` folder, settings in `strata-<model>.json`, chats
in your browser.

## Which model?

| Model | Download | In memory, 128K context | Notes |
| --- | ---: | ---: | --- |
| **Qwen3.8-Flash-Next Q2_0** | 66 GB | about 60 GB (estimate) | The default and the most tested on a Mac. `make pull MODEL=Q2_0` |
| Qwen3.8-Flash-Next IQ3_XXS | 76 GB | 65 GB (measured) | Less compressed than Q2_0; 10-30% slower in our runs. `make pull MODEL=IQ3_XXS` |
| Qwen3.8-Flash-Next IQ3_S | 84 GB | 74 GB (measured) | Less compressed again; about 10% slower than Q2_0 in one A/B. `make pull MODEL=IQ3_S` |
| **GPT-OSS 120B** | 63 GB | about 70 GB (estimate) | The fastest here (about 5B parameters active per token). Install: `make pull MODEL=GPT-OSS-120B`, then [steps](docs/MACOS.md#gpt-oss-120b) |
| GLM-5.3-Flash (Maya-S24) | 95 GB | 89 GB at 32K (measured) | The largest here (320B parameters, about 18B active per token), and the slowest. Chat, thinking and tool calls work. Install: `make pull MODEL=GLM-5.3`, then [steps](docs/MACOS.md#glm-53-flash) |
| HunyuanImage-3.0 (pictures) | 169 GB, streamed (54 GB kept) | 54 GB mapped + 0.3 GB | 80B; about 90 s a picture. Its license excludes the EU, the UK and South Korea. `make pull MODEL=HUNYUAN` (about 2 hours) |
| Nucleus-Image (pictures) | about 50 GB (21 GB kept) | 12 GB mapped + 2 GB | 17B, Apache-2.0; about 45 s a picture. `make pull MODEL=NUCLEUS` |
| Wan 2.2 T2V A14B (videos) | 29 GB | 30 GB at its peak | Two 14B experts + the Lightning LoRAs; a 2 s video in about 7 minutes. `make pull MODEL=WAN2.2` |

"In memory" is the engine with the Qwen models' MTP draft layer and 2 request slots, on the tested Mac; a smaller
context needs less (`make run CONTEXT=32768`). It must fit in what macOS lets the GPU use, which `make check` prints
for your Mac (107.5 GB on the tested 128 GB one; less on smaller Macs). The Qwen models also read a 28 GB table from
the SSD while they run. The other Qwen sizes (IQ2_XS, the Unsloth ones) were not tried on a Mac.

## Using it

- **In the browser:** `http://127.0.0.1:8080` has **Chat**, a live **Monitor** of the model and the Mac (GPU load,
  memory, power, temperature) and **About**. Thinking effort (*Off* to *High*) sets how long the model thinks; a
  **Thinking budget** in the chat's **Sampling** settings caps it in tokens.
- **Your apps and coding agents:** an "OpenAI-compatible" provider with the base URL `http://127.0.0.1:8080/v1` (any
  model name; any API key unless you set one), the Anthropic API at `http://127.0.0.1:8080/v1/messages` (Claude Code:
  `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`), and the Responses API at `/v1/responses`. Tools and MCP work. Client
  examples: [docs/DETAILS.md](docs/DETAILS.md#using-it).
- **Pictures:** say yes to pictures in setup (Qwen models), then attach them in the chat or your app.
- **Making pictures:** `/image <prompt>` in the chat, or `POST /v1/images/generations` (OpenAI's images API), with
  [Nucleus-Image](https://huggingface.co/NucleusAI/Nucleus-Image) (17B) on Strata's own image engine. A 1024x1024
  picture takes about 45-70 s (20-30 steps) and a few GB of memory besides its 20 GB of model files, which macOS maps
  from the SSD. Install: `make pull MODEL=NUCLEUS`, then [steps](docs/MACOS.md#pictures-from-a-prompt-nucleus-image).
  Or [HunyuanImage-3.0](https://huggingface.co/tencent/HunyuanImage-3.0-Instruct-Distil) (Tencent, 80B mixture of experts; one model for the prompt and the picture, so no
  text encoder): 8 steps, about 90 s at 1024x1024, 0.3 GB of private memory besides its 54 GB file, which macOS maps
  from the SSD. Install: `make pull MODEL=HUNYUAN` (about 2 hours), then [steps](docs/MACOS.md#pictures-from-a-prompt-hunyuanimage-30). Its license excludes the EU,
  the UK and South Korea.
- **Making videos:** `/video <prompt>` in the chat, or `POST /v1/videos` (OpenAI's video API: a job you poll), with
  [Wan 2.2 T2V A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) (two 14B experts) on [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)'s Metal backend and
  [lightx2v's Lightning LoRAs](https://huggingface.co/lightx2v/Wan2.2-Lightning) (4 steps instead of 18, one pass a step). A 2 s 832x480 video takes about
  7 minutes (49 without the LoRAs) and 30 GB of memory at its peak. Install: `make pull MODEL=WAN2.2` (29 GB, the
  LoRAs included), then [steps](docs/MACOS.md#videos-from-a-prompt-wan-22-a14b).
- **One request at a time** by default; others wait. A follow-up in the same conversation reads only what is new, so
  it starts quickly; a long new prompt takes a while to read before the answer starts.
- **From another device on your network:** in `strata-<model>.json` set `"host": "0.0.0.0"` and an `"api_key"`,
  start Strata again, then use `http://<your Mac's IP>:8080` with that key. The traffic is not encrypted: keep it on a
  network you trust (or a VPN).

## Something went wrong?

- **It is slow.** Use the *High Power* energy mode where your Mac has it (laptops: *System Settings > Battery > Energy
  Mode*; desktops: *System Settings > Energy*) and keep a laptop plugged in. To see the GPU's clock: `brew install
  macmon`, then `macmon`.
- **It thinks for a very long time.** Set a lower thinking effort, or a **Thinking budget** (4096-8192): when the
  budget runs out the server closes the thinking and the model writes its answer, if max tokens leaves room for one.
- **It says the port is in use.** Another program listens on 8080. If it is another Strata, `make stop` or starting a
  `run-<model>.sh` offers to stop it; otherwise use another port (`make run PORT=8090`). Two models at once need the
  memory for both.
- **The model does not fit.** Close large apps, choose a smaller model or a smaller context (`make run CONTEXT=32768`).

More: [docs/MACOS.md](docs/MACOS.md#limits-and-warnings). Problems with this Mac fork:
[issues](https://github.com/dennis-akimov/Strata/issues). A security problem: report it privately, see
[SECURITY.md](SECURITY.md).

## How does it work?

On a Mac, Strata's own engine (`strata-metal`, in [`metal/`](metal/)) runs the model on the GPU through
[llama.cpp](https://github.com/ggml-org/llama.cpp)'s Metal backend, and the Strata server sits on top of it: the web
app, the OpenAI / Anthropic / Responses / MCP APIs, conversation reuse, the model's draft layer that guesses a few
tokens ahead, and the chat formats of Qwen, GPT-OSS (harmony) and GLM. The CPU and the GPU share the Mac's memory, so
the model's weights sit in it once, for both. The design and the measurements:
[docs/MACOS.md](docs/MACOS.md#how-it-fits-together). The documents of the original are kept in
[`docs/`](docs/README.md), with which ones apply on a Mac.

Pictures from a prompt run in a second engine, `strata-image` (in [`metal/image/`](metal/image/)): Nucleus-Image's text
encoder, its 17B mixture-of-experts image model and its decoder on the same Metal backend, one process per picture. HunyuanImage-3.0 runs there too: the prompt's keys and values once, then 8 steps of the image tokens through
the 80B transformer and its 3D VAE.

Videos run in `sd-cli` from stable-diffusion.cpp, built with a Strata patch that makes the Wan VAE's 3D convolutions
13x faster on Metal; the server runs each clip as a job.

## Credits and license

**[Strata](https://github.com/Niko1221/Strata) is the work of Niko1221 and the Strata contributors**
([all credits](docs/HOW_IT_WORKS.md#credits)); this fork adds the Mac support. The models are by the Qwen team
([Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), compressed by
[ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)), OpenAI
([GPT-OSS](https://huggingface.co/openai/gpt-oss-120b)) and Z.ai (GLM-5.3-Flash, compressed by
[Project Maya](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF)). Strata uses parts of
[llama.cpp / ggml](https://github.com/ggml-org/llama.cpp). Open source under the [MIT License](LICENSE) (Copyright (c)
2026 Niko1221 and the Strata contributors); a few parts and every model have their own licenses
([which ones](docs/HOW_IT_WORKS.md#license)).

Pictures: [Nucleus-Image](https://huggingface.co/NucleusAI/Nucleus-Image) by NucleusAI (Apache-2.0), converted from the
[mlx-nucleus-image](https://huggingface.co/treadon/mlx-nucleus-image) port. [HunyuanImage-3.0](https://huggingface.co/tencent/HunyuanImage-3.0-Instruct-Distil) by Tencent
([Tencent Hunyuan Community License](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0/blob/main/LICENSE): not licensed in the EU, the UK or South Korea).
Videos: [Wan 2.2](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) by Alibaba's Wan team (Apache-2.0), run with [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) (MIT).

If this Mac fork is useful to you, you can support its maintainer:
[buymeacoffee.com/dennisakimov](https://buymeacoffee.com/dennisakimov).
