<h1 align="center">Strata for Mac</h1>

**English** · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md) · [Português](README.pt-BR.md)

<p align="center"><b>Run 100-billion-parameter AI models on your own Mac</b><br>
Apple Silicon · tested on an M5 Max with 128 GB · free and open source</p>

> **This is an independently maintained fork of [Strata](https://github.com/Niko1221/Strata) for Apple Silicon Macs
> only**, maintained by [dennis-akimov](https://github.com/dennis-akimov). It adds a Metal engine and the Mac setup.
> The original's Windows and Linux engines (NVIDIA CUDA, AMD HIP, Intel SYCL) are still in this repository as they were
> at the fork point, but they are not built, tested or updated here. **Linux on an Apple Silicon Mac** (Asahi Linux, or
> Debian with Asahi's kernel) is supported: the same engine on the Mac's GPU through Vulkan, and `strata-tui`, the app
> in a terminal ([below](#linux-on-an-apple-silicon-mac)).
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
| **[GPT-OSS 120B with a DFlash draft](docs/MACOS.md#gpt-oss-120b-with-a-dflash-draft)**‡ | 122-135 tokens/s on math, 71-77 on chat | - | 65 GB |
| **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) Q2_0**, with `--mtp on` | 60-69 tokens/s | 92-102 tokens/s | 38 GB |
| **Qwen3.8-Flash-Next IQ3_XXS**, with `--mtp on` | 46-48 tokens/s | 73-92 tokens/s | 47 GB |
| **[GLM-5.3-Flash](docs/MACOS.md#glm-53-flash) Maya-S24**\* | 13-26 tokens/s | 29-32 tokens/s | 86 GB |
| **[GPT-OSS 20B](docs/MACOS.md#gpt-oss-20b-and-qwen36-35b-a3b)** (MXFP4)† | 128-139 tokens/s | - | 12 GB |
| **[Qwen3.6-35B-A3B](docs/MACOS.md#gpt-oss-20b-and-qwen36-35b-a3b)** (MXFP4_MOE)† | 89-97 tokens/s | - | 22 GB |

\* Measured 2026-10-10 while Docker used about 40 GB and swap was full (the GPU at a low clock), so a floor; GLM also reads about 3x the active parameters per token of the others.

† Measured 2026-10-12 on 600-word (800-token) answers, with the GPU throttled to 1.2-1.5 GHz after earlier runs, so a floor.

‡ Measured 2026-10-13 on 300-token answers, against 91-99 tokens/s without the draft in the same runs: the draft helps math, not chat.

On a 16 GB Mac: a 13-inch MacBook Pro with the M1 (2020), 2026-10-29/30, short answers, on macOS and on Linux on the
same Mac:

| Model | macOS 27 (Metal) | Linux, Debian + Asahi kernel 7.1.13 (Vulkan, Mesa 26.1) | Model file |
| --- | ---: | ---: | ---: |
| **GPT-OSS 20B** (MXFP4) | 16.8-19.9 tokens/s at an 8-16K context § | 13.3-13.7 tokens/s at a 32K context ¶ | 12 GB |
| **LFM2.5-8B-A1B** (LiquidAI Q8_0) | 27.6 tokens/s at a 32K context; prompt 528-574 tokens/s | 22.9 tokens/s at a 128K context; prompt 185 tokens/s | 9.0 GB |
| **LFM2.5-24B-A2B** (mradermacher Q3_K_M) | 28.7 tokens/s at a 16K context; prompt 248-267 tokens/s | 12.2 tokens/s at a 32K context; prompt 61 tokens/s ‖ | 11.5 GB |

§ Only after raising macOS's GPU memory limit (`sudo sysctl iogpu.wired_limit_mb=13312`, until the next restart): by
default macOS lets the GPU of a Mac up to 36 GB hold about two thirds of the memory, too little for the model's 11 GB of
weights. `make setup` and Strata.app offer it.

¶ Linux's Apple GPU driver (Honeykrisp) gives the GPU half the memory by default (7.65 GB here); setup writes
`"HK_SYSMEM"` into the run config to give it 13 GB.

‖ Avoid IQ2/IQ3 files on Linux: their experts' mat-vec runs 15-30x slower on Honeykrisp than the K-quants'. Unsloth's
LFM2.5-24B UD-Q3_K_XL (IQ3_XXS experts) answered at 1.3 tokens/s, 91% of each token in that one step; mradermacher's
Q3_K_M of the same model at 12.2 (Qwen3.6-35B's 2-bit UD-IQ2_M, since taken out of `make pull`, at 2.4). The LFM2.5 models (`make pull MODEL=LFM2.5-8B` / `LFM2.5-24B`, LiquidAI, both
think before answering) passed the M1's checks (arithmetic, a fact, letter counting, Russian, code, a tool call): the
server reads their `<|tool_call_start|>[f(a=1)]<|tool_call_end|>` calls and their template's `{% generation %}` tag.

On Linux the GPU sets the pace (the engine uses under 10% of the CPU; the GPU's clock is its firmware's, the same top
1278 MHz as under macOS, with no Linux setting). What was measured on the M1, 2026-10-29:

| Change | Answer speed | Kept |
| --- | ---: | --- |
| `metal/patches/core/0003-honeykrisp-mat-vec.patch` (llama.cpp's mat-vec: large workgroups, 2 rows each) | 12.2 -> 13.7 tokens/s | yes, built in |
| CPU governor `performance` (`p` in the TUI's Monitor) | +3% | your choice |
| llama.cpp b12000 (newest, 487 commits later) | 12.8 tokens/s | no |
| Mesa 26.2.6, or 26.2.6 with the unmerged mesa!44041 | 13.7-13.8 tokens/s | no |
| `GGML_VK_FORCE_MMVQ`, `GGML_VK_DISABLE_FUSION`, `GGML_VK_DISABLE_GRAPH_OPTIMIZE`, `HK_PERFTEST=batch` | 11.8-13.3 tokens/s | no |

The rest of the gap to Metal is the driver: Honeykrisp has no integer dot product and no cooperative matrix, which
llama.cpp's fast paths need; 57% of the GPU's time per token is the MXFP4 experts' mat-vec. Details and how to measure
(`GGML_VK_PERF_LOGGER=1`): [docs/MACOS.md](docs/MACOS.md#linux-on-an-apple-silicon-mac-asahi).

On a 16 GB Mac: GPT-OSS 20B, or on Linux LFM2.5-8B-A1B (the fastest there) and LFM2.5-24B-A2B.
The first one or two answers after starting it are slower (8-10 tokens/s on the M1): macOS is still reading the 11 GB
model from the SSD into memory; from then on 17-20 tokens/s, through the chat page or the API alike. The thinking
effort (*High* makes it think longer, not slower per token) and the sampling settings (top-k 0 or 40: 16.8 and 17.4
tokens/s) do not change the speed. An 8K context with a prefill batch of 256 leaves macOS the most room on 16 GB.

**The screen glitches while it answers?** The model and the window server share the GPU's memory limit
(`iogpu.wired_limit_mb`). On the 16 GB M1 at 13,312 MB, gpt-oss-20b at a 16K context with a prefill batch of 512 took
12.97 GB of it, leaving the screen 0.34 GB: WindowServer's frames failed (`log show` has
`kIOGPUCommandBufferCallbackErrorOutOfMemory` and IOGPUFamily `FAILED To Free ... total accelerator = max`) and answers
fell to 10.6-15 tokens/s. Strata keeps 1 GB of the limit for the screen; Strata.app now shrinks a config that does not
leave it (the largest context that fits, written into the config and its log) before it starts the model.

Pictures and videos on the same Mac (2026-10-11/12):

| Model | Output | Time | Memory |
| --- | --- | ---: | ---: |
| **[HunyuanImage-3.0](docs/MACOS.md#pictures-from-a-prompt-hunyuanimage-30)** (80B, Q4_0 experts) | 1024x1024 picture, 8 steps | 88 s | 0.3 GB, its 54 GB file mapped |
| **[Nucleus-Image](docs/MACOS.md#pictures-from-a-prompt-nucleus-image)** (17B, Q4_0 experts) | 1024x1024 picture, 20 steps | 45 s | 2 GB, its 12 GB of files mapped |
| **[Wan 2.2 A14B](docs/MACOS.md#videos-from-a-prompt-wan-22-a14b)** + Lightning LoRAs | 832x480 video, 2 s (33 frames), 4 steps | 430 s | 30 GB at its peak |
| Wan 2.2 A14B without the LoRAs | the same video, 18 steps | 2907 s | about 28 GB |

The video times were measured while Docker held 76 GB and macOS was swapping (the GPU at 600-750 MHz), so a floor.

The Qwen numbers use its MTP draft layer, which is opt-in (`python3 bootstrap.py --setup --mtp on`, about 12 GB more disk).
The context's cache comes on top of the model file (a few GB; more for long contexts). In the *Automatic* energy mode
the same Mac wrote long answers 2-3x slower. Other Macs were not measured. Every number and how it was measured:
[docs/MACOS.md](docs/MACOS.md#measured).

## What you need

| | |
| --- | --- |
| **Mac** | Apple Silicon (M1 or newer). Only one M5 Max with 128 GB was tested; other chips and sizes are expected to work but were not tried. |
| **Memory** | Enough for the model file plus its cache, inside what macOS lets the GPU use (on the tested 128 GB Mac: 107.5 GB). 16 GB runs GPT-OSS 20B at a small context with other apps closed; 24-32 GB runs it comfortably; the Qwen3.8 models want 64 GB or more. `make setup` shows what fits yours; see [Which model?](#which-model) |
| **Disk** | The model's download plus a few GB (70-100 GB per model), on the internal SSD. |
| **Software** | Nothing beforehand: `make setup` installs Apple's Command Line Tools, Homebrew and Python (when missing), Rust, the build tools and aria2. Tested on macOS 26.4 and 27.0. |

## Install

In Terminal:

```sh
git clone https://github.com/dennis-akimov/Strata.git
cd Strata
make setup                 # everything: tools, builds, a model that fits this Mac, a first answer, Strata.app
```

`make setup` installs what is missing (Apple's Command Line Tools open an installer window; Homebrew asks for your
password), builds the engine, the server and Strata.app, lists the models with whether each fits this Mac (the best
one is preselected: Enter takes it), downloads it, writes its run config with the largest context that fits (the
engine measures it), checks that the model answers, and opens Strata.app on it. If anything stops, run `make setup`
again: finished steps are skipped and downloads continue. `make setup MODEL=GPT-OSS-20B SETUP_ARGS=--yes` asks
nothing. Without the app: `make run PORT=8090` and http://127.0.0.1:8090. Loading takes a minute or so each start. `make run` keeps Terminal busy
while Strata runs; press Ctrl+C to stop it, or use `make start` / `make stop` to run it in the background. Every
option, the limits and the warnings: **[docs/MACOS.md](docs/MACOS.md)**.

### Linux on an Apple Silicon Mac

On Asahi Linux, or Debian / Ubuntu with Asahi's kernel (M1 and M2; tried on an M1 with Debian forky):

```sh
git clone https://github.com/dennis-akimov/Strata.git
cd Strata
python3 bootstrap.py --onboard    # the first time, before make is installed; later: make setup
make tui                          # Strata.app in the terminal: Ctrl+S starts the model, type, Enter
```

Setup installs the system packages with sudo (the compiler, make, cmake, the Vulkan build parts, Mesa's Apple GPU
driver, aria2), adds you to the GPU's groups (`render`, `video`; from your next login), installs Rust, builds the
engine with Vulkan, the server and `strata-tui`, downloads the model, writes its run config and checks that it answers.
`python3 bootstrap.py --detect` shows what it found (the chip, and whether the GPU's driver is there). Details:
[docs/MACOS.md](docs/MACOS.md#linux-on-an-apple-silicon-mac-asahi).

**Updating:** `git pull`, then `make run` (setup rebuilds the engine when the sources changed; the model stays).
**Where things are:** models in `Strata-data/` next to the `Strata` folder, settings in `strata-<model>.json`, chats
in your browser.

## Which model?

| Model | Download | In memory, 128K context | Notes |
| --- | ---: | ---: | --- |
| **Qwen3.8-Flash-Next Q2_0** | 66 GB | about 60 GB (estimate) | The default and the most tested on a Mac. `make pull MODEL=Q2_0` |
| Qwen3.8-Flash-Next IQ3_XXS | 76 GB | 65 GB (measured) | Less compressed than Q2_0; 10-30% slower in our runs. `make pull MODEL=IQ3_XXS` |
| Qwen3.8-Flash-Next IQ3_S | 84 GB | 74 GB (measured) | Less compressed again; about 10% slower than Q2_0 in one A/B. `make pull MODEL=IQ3_S` |
| **GPT-OSS 120B** | 63 GB | about 70 GB (estimate) | About 5B parameters active per token. Install: `make pull MODEL=GPT-OSS-120B`, then [steps](docs/MACOS.md#gpt-oss-120b) |
| GPT-OSS 120B + DFlash draft | 1.6 GB more | about 72 GB (estimate) | Faster on math and step-by-step answers (about +35%), slower on chat (about -20%). `make pull MODEL=GPT-OSS-120B-DFLASH`, then [steps](docs/MACOS.md#gpt-oss-120b-with-a-dflash-draft) |
| GPT-OSS 20B | 12 GB | 19 GB (measured) | The fastest here (about 3.6B parameters active per token), less capable than the 120B. `make pull MODEL=GPT-OSS-20B`, then [steps](docs/MACOS.md#gpt-oss-20b-and-qwen36-35b-a3b) |
| Qwen3.6-35B-A3B | 22 GB | 29 GB (measured) | About 3B parameters active per token, but no faster than GPT-OSS 120B here. `make pull MODEL=QWEN3.6-35B`, then [steps](docs/MACOS.md#gpt-oss-20b-and-qwen36-35b-a3b) |
| Apple's on-device model (AFM 3 Core on macOS 27) | none (part of macOS) | macOS's own | 3B dense, 4,096-token context on M1/M2 (8,192 on M3 and later); about 30 tokens/s on the 16 GB M1. Small: on the M1's checks it got arithmetic, Russian and code right, Australia's capital and the r's in "strawberry" wrong, and it does not call tools through Strata yet. Needs macOS 26 or later and Apple Intelligence on. `make afm` (it writes `strata-apple-afm.json`), then pick it in Strata.app |
| LFM2.5-8B-A1B | 9.0 GB | about 13 GB at 128K (measured) | LiquidAI's, 1.5B parameters active per token, thinks before answering; 22.9 tokens/s on the 16 GB M1 under Linux. `make pull MODEL=LFM2.5-8B` (it writes its run config) |
| LFM2.5-24B-A2B | 11.5 GB | about 13 GB at 32K (measured) | LiquidAI's, 2.45B active, thinks before answering; 12.2 tokens/s on the M1 under Linux. `make pull MODEL=LFM2.5-24B` |
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
- **macOS Reminders (Strata.app):** the chat's settings (gear) > **macOS Reminders**. The model then has
  `reminders_lists`, `reminders_list`, `reminders_add` (title, notes, due `YYYY-MM-DD [HH:MM]` with an alert at that
  time, list) and `reminders_complete`: "remind me to call Anna tomorrow at 9", "what is on my list?". macOS asks once
  for Strata; it is the MCP server `reminders` (`engine-metal/strata-reminders`, metal/reminders, EventKit), saved in
  the run config, off again with the same switch. Tried on the 16 GB M1 with LFM2.5-24B: listed, added and completed.
- **Browser beside the chat (Strata.app):** the bar's **Web** button splits the window: the chat on the left, a
  browser on the right (back, forward, reload; the address field takes an address, a site like `news.ycombinator.com`,
  or words to search with DuckDuckGo). **Send page to chat** puts the page you read (its title, address and text, up
  to 20,000 characters) into the message box as an attached file, to ask about it. A full WebKit view: sign-ins
  stay like in Safari. The chat's globe button opens and closes it too.
- **The model browses (Strata.app):** with the chat's MCP switch on, the model has `browser_search` (Google's results
  page, read compactly, with today's date), `browser_open`, `browser_read` (the main text in parts of 3,500 characters
  and up to 40 links, buttons and fields numbered `[n]`), `browser_click`, `browser_type`, `browser_scroll` and
  `browser_back`, in that same pane, in front of you. After a search it opens only addresses from the results or pages
  it read (guessed ones were 404s); at the tool-round limit (`mcp.max_rounds`, 8) it answers from what it has, and
  when tool results outgrow the context the oldest ones are compacted to a note (auto-compact) instead of failing. On
  the 16 GB M1 with LFM2.5-24B: "Paris weather today" 30 s (one search), "the most recent F1 race" 127 s (right, 2026);
  a page read went from ~5,300 tokens (~25 s to read) to ~1,500. Google is the start page and the search. An answer from the web always ends with **Sources** (the pages the browser read, or the search, as links: the
  server adds the ones the answer did not link), and links in the chat are clickable (in Strata.app they open in the
  browser pane; a bare address becomes a link). The page scripts are tested in headless Chrome (`cargo test` in
  app/, skipped without Chrome). Typing waits for your
  **Allow** in the browser's strip (it says what goes into which field on which site), a password field is refused
  (you sign in yourself), and **Stop model** there refuses its browser tools until you let it again. Strata.app serves
  these tools (app/src/browser_mcp.rs: MCP on 127.0.0.1 with a key made at each start) and hands them to the server it
  starts as the MCP server `browser` (not saved; only there while the app runs). Tried on the M1 with LFM2.5-24B:
  opened example.com, read it, clicked "Learn more", read IANA's page. Opening a page, clicking, scrolling and going
  back answer with the page itself, and each tool result says how many tool calls are left for the answer.
- **Google Sheets and My Maps (Strata.app):** through the same browser pane, signed in to Google as you (no Google
  API, no Cloud project): `sheets_create` (title and header row), `sheets_read` (rows numbered as in the sheet),
  `sheets_append` (under the last filled row; rows whose first cell is FALSE/TRUE get tick boxes), `sheets_write` (from
  a cell), `sheets_delete_rows`, `sheets_clear`, `sheets_trash`, and `maps_create`, `maps_import_sheet` (a sheet's rows
  as places: an address column to place them, a column to title them, every column kept in each place) and
  `maps_trash`. The trash tools wait for your **Allow** in the browser's strip. They run scripts in Google's pages
  (app/src/google.rs: a cell range by the address's `#range=`, a paste event for writing, a copy event for reading,
  the menus by mouse events). My Maps' Import opens Google's file picker from another origin (docs.google.com); a
  script the browser puts in every frame wakes only there and answers My Maps' page for a token made at each start, so
  `browser_read`/`click`/`type` also work inside the picker. Tried on the M1 (30 Oct 2026): a sheet of 18 downtown San
  Diego studios and a map of them; create 10 s, append 6 s, read 2–3 s, map create 7 s, import 9 s. Google's pages
  change: when a tool says it found no button or menu item, that is where.
- **Scheduled prompts:** the calendar button by the chat's message box. A prompt that runs at
  set times: once, every N minutes (5 or more), hourly, daily, weekdays, weekly, monthly, or a cron expression
  (`minute hour day-of-month month day-of-week`, with `*`, lists, ranges, `*/N`, names; both day fields set: either
  matches, as POSIX says). Each run asks this server's chat (with the tools, when allowed: "every weekday at 8:00, my
  open reminders and the weather in Paris"); its answer comes as a macOS notification and is kept (the last 10), with
  Run now, Pause, Edit, **Open in chat** (go on from it) and Delete. They run in the server, whatever started it; a run
  missed while it was off or the Mac asleep runs once when it is back. Kept in `.strata-schedules.json` (git ignores
  it); `GET /schedules`, `POST /schedules` (`{"run": id}` runs one now), `DELETE /schedules/<id>`. For a job outside
  Strata, macOS's own scheduler is launchd (`~/Library/LaunchAgents`, `StartCalendarInterval`).
- **`.env` (optional, a convenience):** `cp .env.example .env; chmod 600 .env`. `API_KEY=` is one more key the
  server accepts for `/v1` and the chat page, next to the run config's `"api_key"`, `--api-key` and
  `STRATA_API_KEY` (none of them is replaced; Strata.app and the TUI send it when their config has none; it is never
  printed). `MCP_<NAME>_URL=` and `MCP_<NAME>_API_KEY=` add the MCP server `<name>` (`Authorization: Bearer`, or
  `MCP_<NAME>_HEADER=X-API-Key`); one of the same name in the config, `--mcp-config` or added on the page wins, and
  `.env`'s are read-only on the page and never written to the config. Read when the server starts.
- **Voice ("Hey Strata", macOS 26 or later):** the microphone button in the chat's bar turns it on and off (lit while
  on); the small arrow next to it opens its settings. Say the wake phrase and your
  question: with the chat page open, the question goes into the chat as yours and the answer streams there like a
  typed one (the chat keeps both, with its history and settings), then is read aloud in macOS's voice; with no page
  open, the helper asks the model itself and reads the answer. Voice stays on across restarts. Apple's on-device dictation
  hears it (`SpeechAnalyzer`, no network, no Apple Intelligence and none of its memory); the model you run answers. The
  panel has the switch, **Allow microphone & speech** (macOS asks once for Strata.app; refused, it opens the page in
  System Settings), and the **wake phrase** (1-4 words; its last word wakes it as the first or second word said, heard
  a little off too: "Heyden" as "Hayden", "Haydn" or "hey den"). `engine-metal/strata-voice` (`make build`,
  metal/voice/strata_voice.swift) runs as the server's child; `GET /voice` (`?page=1`: an open
  page takes the questions) and `POST /voice` (`{"on": true}`, `{"phrase": "..."}`, `{"say": "..."}`, `{"hush": true}`,
  `{"permissions": true}`) are what the button uses. The panel shows what was heard last: a wake phrase heard as
  another word shows there. No sound at all from the Mac (`say` hangs, coreaudiod logs `'what'`, 2003329396): `sudo
  killall coreaudiod`. Tested on the 16 GB M1 with LFM2.5-24B: "Hey
  Strata, what is two plus two?" answered in about 3 s.
- **Free memory for the model** (Strata.app's **System** button, or its card when the server is stopped): toggles that
  keep Apple Intelligence & Siri off (its on-device model holds about 0.4 GB; Apple Intelligence's own switch opens
  in System Settings; leave it on to use Apple's model in Strata), keep the moving (aerial) wallpaper off, and set the GPU's memory limit at every startup (a
  LaunchDaemon for `iogpu.wired_limit_mb`; macOS asks for your password). While the app runs it keeps the first two
  off. The memory card also lists the apps using the most memory, with **Quit**.
- **In the terminal:** `make tui` is Strata.app for a terminal (Linux on an Apple Silicon Mac, or over SSH): a bar with
  the state and the memory, **Chat** (the thinking dimmed), **Monitor**, **Memory** (the context to pick, the programs
  using the most memory) and **Log**. Ctrl+S starts or stops the model, Ctrl+Q quits and stops it.
- **One request at a time** by default; others wait. A follow-up in the same conversation reads only what is new, so
  it starts quickly; a long new prompt takes a while to read before the answer starts.
- **From another device on your network:** in `strata-<model>.json` set `"host": "0.0.0.0"` and an `"api_key"`,
  start Strata again, then use `http://<your Mac's IP>:8080` with that key. The traffic is not encrypted: keep it on a
  network you trust (or a VPN).

## Something went wrong?

- **It is slow.** Use the *High Power* energy mode where your Mac has it (laptops: *System Settings > Battery > Energy
  Mode*; desktops: *System Settings > Energy*) and keep a laptop plugged in. To see the GPU's clock: `brew install
  macmon`, then `macmon`. Right after starting, the first answer or two are slower while the model is read into
  memory. The Monitor's live speed covers the last few seconds and dips while the window redraws; the request table
  shows each answer's average. On a 16-32 GB Mac, Strata.app's **System** toggles leave the model more memory.
- **It thinks for a very long time.** Set a lower thinking effort, or a **Thinking budget** (4096-8192): when the
  budget runs out the server closes the thinking and the model writes its answer, if max tokens leaves room for one.
- **It says the port is in use.** Another program listens on 8080. If it is another Strata, `make stop` or starting a
  `run-<model>.sh` offers to stop it; otherwise use another port (`make run PORT=8090`). Two models at once need the
  memory for both.
- **The model does not fit.** Close large apps, choose a smaller model or a smaller context (`make run CONTEXT=32768`).
  Strata.app's memory card and `make tui`'s Memory view show what it needs, what the GPU may hold and what is free.
- **It runs out of GPU memory on a 16-32 GB Mac although memory is free.** macOS: the GPU's limit is about two
  thirds of the memory by default; `sudo sysctl iogpu.wired_limit_mb=<the memory in MB less 3072>` raises it until the
  next restart (Strata.app's **Raise it** button does it). Linux: the run config's `"env": {"HK_SYSMEM": "<bytes>"}`
  sets the GPU's share (setup writes it).

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
