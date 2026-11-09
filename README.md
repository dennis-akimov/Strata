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

**Several agents at once.** One model can answer several requests at the same time: `"parallel": N` in the run config
(2-8), or the *agents* menu in Strata.app. It is one engine on one port with one copy of the weights; each agent gets
its own conversation cache, and the engine writes all of their answers in the same passes. Qwen3.6-35B-A3B (MXFP4_MOE)
on the M5 Max, 2026-11-01, 256-token answers to four different 590-token chat prompts, thinking off, the engine driven
directly:

| Agents at once | Tokens/s together | Tokens/s each | Memory at a 128K context |
| ---: | ---: | ---: | ---: |
| 1 | 98.6 | 98.6 | 26.5 GB |
| 2 | 148.7 | 74.4 | |
| 3 | 171.6 | 57.2 | 34.8 GB |
| 4 | 190.5 | 47.6 | 37.5 GB |

- Four agents write 1.9x what one does, each at about half the speed. Three copies of the model instead would hold
  the 22 GB of weights three times and share the same GPU.
- An agent's new prompt is read before the others go on: an 18,605-token one took 11 s (about 1,700 tokens/s), and
  the two agents already answering wrote 3 tokens meanwhile. A follow-up message reads only what is new, so these
  pauses are usually short.
- Each agent's context gets its own cache (here 128K each); the memory column is the engine's own measure (`--plan`).

**Agents, context and the conversation cache.** `server/target/release/strata-bench CONFIG.json --agents 1,2,3,4`
restarts the config's server for every mix of agents, context (16K-256K each) and `--conversation-cache-mib`
(0-32 GB), and has each agent write three 128-token answers through the OpenAI API: one to a new 8K-token
conversation, one to a second, short conversation (which pushes the first out of the GPU), and a follow-up in the first.
It puts Strata's own config back with `--restore CONFIG.json`. M5 Max, 2026-11-02, Docker running (results in
`bench/results/2026-11-02-agents-matrix*`, about 20% apart from run to run):

| Model | Agents | Context each | Cache | Follow-up: first token | Tokens/s together | GPU memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B (MXFP4_MOE) | 3 | 16K-256K | 0 | 6-8 s | 65-92 | 27-48 GB |
| | 3 | 16K-256K | 8-32 GB | 0.33-0.5 s | 115-155 | 27-48 GB |
| | 4 | 64K | 16 GB | 0.42 s | 165 | 33 GB |
| | 4 | 256K | 8-32 GB | 0.5-0.6 s | 149-153 | 54 GB |
| Qwen3-Coder-30B-A3B (Q8_0) | 1 | 128K-256K | 16 GB | 0.35-0.4 s | 80 | 49-62 GB |
| | 2 | 64K | 16 GB | 0.7 s | 86 | 54 GB |
| | 4 | 32K | 16 GB | 1.1 s | 89 | 52 GB |
| | 4 | 32K | 8 GB | 9 s (25% kept) | 44 | 51 GB |
| | 4 | 128K | 16 GB | 19 s | 33 | 100 GB |
| Qwen3-Coder-30B-A3B (Q4_K_M) | 1 | 16K-256K | 8-32 GB | 0.3-0.45 s | 90-105 | 22-48 GB |
| | 2 | 32K-128K | 16 GB | 0.55-0.75 s | 95-103 | 31-61 GB |
| | 3 | 64K | 16 GB | 0.8 s | 110 | 48 GB |
| | 4 | 16K | 16-32 GB | 1.1 s | 99-107 | 29 GB |
| | 3 | 256K | any | no answer | - | 110 GB |

- Without the cache every follow-up reads its whole conversation again (6-15 s). With it the follow-up reads only the
  new message. 8 GB is enough for Qwen3.6 here, but Qwen3-Coder's larger cache per token needs 16 GB for 3-4 agents.
- A longer context costs memory, not speed, until it no longer fits: Qwen3-Coder at 2 × 256K or 4 × 128K takes
  100-113 GB and slows down, and 3-4 × 256K does not answer at all (Q4_K_M too: 3 × 256K needs 110 GB, over the
  107.5 GB macOS gives the GPU).
- Qwen3-Coder (Q8_0) writes about 85-90 tokens/s together however many agents run: its weights are twice as heavy to
  read per token as Qwen3.6's MXFP4. For several agents at once, Qwen3.6 writes about twice as much.
- Qwen3-Coder at Q4_K_M (measured on one requantized from the Q8_0 with `llama-quantize`, the same size and types as
  Unsloth's that `make pull MODEL=QWEN3-CODER-30B-Q4` fetches) is about 40% faster than
  Q8_0 for one agent and 20-30% for 3-4, and takes 14 GB less memory. Its conversation memory per token does not
  shrink, so 3-4 agents still stop at about 110 tokens/s together.
- Strata.app's default for Qwen3.6 is 4 agents, 64K each, 16 GB of cache.

**How many agents, and how to set them up.** In the model's run config (`strata-<model>.json`), or the *agents* menu
in Strata.app:

```
"parallel": 4,
"args": [..., "--max-context", "65536", "--conversation-cache-mib", "16384"]
```

All agents use the same port, API key and model name; point each coding agent (or each subagent) at it as usual.
A request beyond the N slots waits in the server's queue, it is not refused. `--max-context` is per agent and is held
in memory from the start, so agents × context must fit in what macOS lets the GPU use (`make check`). The
conversation cache is shared by all agents: it keeps the conversations they are not working on right now, so a
subagent's side question does not cost the main conversation its place. More in [docs/BATCHING.md](docs/BATCHING.md)
and [docs/DETAILS.md](docs/DETAILS.md).

The same bench at a 64K context each and 16 GB of cache, follow-up answers (tokens/s together / each):

| Model | 1 agent | 2 agents | 3 agents | 4 agents | Memory, 3 agents at 128K |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B (MXFP4_MOE) | 99 (2026-11-01) | 149 / 74 (2026-11-01) | 135 / 45 | 165 / 41 | 37 GB |
| Qwen3-Coder-30B-A3B (Q8_0) | 68 | 86 / 43 | 82 / 27 | 68 / 17 (89 / 22 at 32K) | 87 GB |
| Qwen3-Coder-30B-A3B (Q4_K_M) | 87 | 95 / 48 | 110 / 37 | 85 / 21 (91 / 23 at 32K) | 74 GB |
| GPT-OSS 120B (MXFP4) * | 53 | 81 / 40 | 89 / 30 | not run | 86 GB |

\* While the Mac checked a large download and Docker ran, so a floor: one agent wrote 96-99 tokens/s at a 16K context
in the same run.

- For agents that mostly wait on tools (tests, builds, searches), 3-4 agents of Qwen3.6 keep the GPU busy; for one
  agent that has to be fast, one slot is quickest.
- Qwen3-Coder and GPT-OSS 120B gain little past 2-3 agents and need much more memory per agent: 2-3 agents at
  64-128K is their sweet spot on a 128 GB Mac. For Qwen3-Coder take the Q4_K_M: 3 agents at 64K with 16 GB of cache
  wrote the most (110 tokens/s together, first token in 0.8 s, 48 GB).
- The follow-up's first token comes in 0.3-0.7 s with 1-2 agents and about 1 s with 3-4, while every agent's
  conversation stays in the cache.

**Agents of different sizes: own slots, one shared pool, or a router.** By default each agent holds its whole context
from the start (4 agents at 128K = 4 x 128K of K/V). Two other ways, measured on the M5 Max, 2026-11-03, Qwen3.6-35B-A3B,
four agents at once with ~20K / 40K / 50K / 100K-token conversations (docs/CONTEXT_POOL.md):

| Layout | Memory | Reading the 4 prompts (~210K tokens) | First token, cold: 20K / 40K / 50K / 100K | Follow-up's first token | Decode, 4 at once | Decode, each |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| Own slots, 4 x 128K (the default) | 35.5 GB | 273 s | 14 / 107 / 68 / 258 s | 0.3-1.5 s | **58.1 tok/s** | 14.7-15.2 |
| A shared pool of 288K, each agent up to 128K | **29.2 GB** | 255 s | 14 / 234 / 200 / 153 s | 0.6-2.4 s | 47.5 tok/s | 12.0-12.5 |
| A router: servers of 32K, 2 x 64K and 128K | three engines | 259 s | 42 / 86 / 219 / 247 s | 0.4-4.0 s | 47.2 tok/s | 12.2-14.5 |

- **Own slots decode fastest** (22% faster than a pool): each agent's attention reads only its own conversation. Take
  them when the agents' contexts fit in memory (Qwen3.6 keeps K/V on 10 of its 40 layers, so they easily do).
- **A pool holds the same agents in less memory** (6.3 GB less here) and lets any agent grow to the cap, but each
  token's attention walks the whole pool. Take it when the contexts would not fit as own slots: a model with a big
  K/V per token (GLM-5.3, GPT-OSS 120B) at long contexts. In Strata.app: the *pool* menu beside *agents* (off, or a
  half / three quarters / all of context x agents, kept in proportion when you change either); in a run config:
  `"args": [..., "--kv-unified", "--kv-pool", "294912"]`.
- **A router** (`strata-router --backend URL=CONTEXT ...`) sends each request to the smallest server it fits; on one
  Mac its engines compete for the GPU instead of sharing a batch, so it was slowest. It is for spreading agents over
  several Macs.
- Reading prompts dominates a first request in every layout (they are read one at a time); follow-ups reuse ~100%.

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

Several agents at once on the M1 (`"parallel": 3`, see [above](#how-fast-is-it)): LFM2.5-8B-A1B (Q8_0) on macOS, 2026-11-01,
through the OpenAI API, three different chat prompts at a time, each answer its full 512 tokens (the model thinks first):

| Agents at once | Tokens/s together | Tokens/s each | Memory at a 32K context each |
| ---: | ---: | ---: | ---: |
| 1 | 30.1 | 30.1 | 10.65 GB |
| 2 | 34.4 | 17.2 | |
| 3 | 36.8 | 12.3 | 11.86 GB |

The M1 gains much less from writing several answers at once than the M5 Max (1.22x for three, against 1.74x for
Qwen3.6-35B there): its 8-core GPU is already busy with one. Three agents fit easily (LFM2.5's hybrid layers keep
small caches: 1.6 GB for three 32K contexts), so on the M1 they are for working at the same time, not for more output.

The agents bench (`strata-bench`, as above: each agent's own ~3,000-token conversation, a short side question each, then
the conversation again plus a turn) on the M1, 2026-11-03, Sheared-LLaMA-2.7B (Q5_K_M, 1.9 GB; a 2023 base model with
SentencePiece's tokenizer, which Strata reads since that day; its 4K context, 128-token answers):

| Agents | Answer, tokens/s each | Follow-up's first token, no cache | ... with a 4 GB conversation cache | Memory |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 19.6-20.5 | 9.8 s | 0.6 s (99% of it from the cache) | 3.9 GB |
| 2 | 15.0-17.5 | 15.0 s | 26.5 s (50%) | 6.7 GB |
| 3 | 14.7-18.9 | 22.5 s | 45.3 s (0%) | 8.1 GB |
| 4 | 15.9-18.8 | 37.7 s | 46.5 s (25%) | 9.4 GB |

Its prompts read at 210-268 tokens/s. One agent's conversation fits the cache and comes back in 0.6 s; with 2-4 the
cache holds too few of them (this 2023 model keeps a big state per token: about 1 GB a conversation) and a follow-up
waited longer than without one.

**No tools: `supports_tools: false`.** Sheared-LLaMA is a base model: it continues text and was never trained to
follow instructions or call tools, and its chat template (`User:` / `Assistant:`) has no place for tools (`/props`
says `"chat_template_caps": {"supports_tools": false, ...}`).  Asked to "use the web to find today's news" it writes
more made-up `User:` and `Assistant:` turns.  The web page's tools (MCP) are refused with an HTTP 400 that says so; a
client's own tools are dropped, and the server log says so once.  For tools on a small Mac take LFM2.5-8B or
Qwen3-4B.

**Qwen3-4B-Instruct-2507 on the M1** (`make pull MODEL=QWEN3-4B`: Unsloth's Q8_0, 4.3 GB; a 32K context, about 11 GB
in all), 2026-11-03, one agent:

| Prompt | Read, tokens/s | Answer, tokens/s |
| ---: | ---: | ---: |
| 26 tokens | - | 12.6 |
| 1.9K | 189 | 11.2 |
| 4.5K | 112 | 8.5 |
| 15K (12.7K of it reused) | 65 (the 2.2K new) | 8.7 |
| 19K | 107 | - |

It calls tools in Hermes' JSON form (`<tool_call>{"name": ..., "arguments": {...}}</tool_call>`), which the server
reads.  Asked on Strata's page, with tools on, for today's top 3 technology headlines with sources, it searched
(Strata.app's browser), opened Reuters pages in 8 rounds and answered with three dated headlines and their links in
351 s; 120 s of that were two page clicks that timed out (60 s each).  It writes at about 40% of LFM2.5-8B's speed
(30 tokens/s here: 1.5B parameters active a token against Qwen3-4B's 4B), and needs no thinking first.

Several agents on the M1, `make pull MODEL=QWEN3-4B-Q4` (Q4_K_M, 2.5 GB; 1.6x the Q8_0's writing speed for one) against
LFM2.5-8B (Q8_0): the agents bench (each agent's ~3,000-token conversation, a short side question, the conversation
again plus a turn), 8K context each, no conversation cache, 2026-11-03:

| Agents | Prompt read, tok/s: 4B Q4 / 8B | Follow-up's first token | Follow-up written together, tok/s | GPU memory |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 187 / 321 | 11.4 / 4.3 s | 20.3 / 19.7 | 5.3 / 10.5 GB |
| 2 | 179 / 374 | 20.8 / 9.3 s | 18.7 / 19.9 | 7.7 / 10.7 GB |
| 3 | 121 / 366 | 35.0 / 14.1 s | 17.3 / 25.1 | 8.9 / 10.8 GB |
| 4 | 126 / 315 | 42.5 / 17.1 s | 22.4 / 23.5 | 10.1 / 10.9 GB |

- LFM2.5-8B reads prompts 2-3x faster (a mixture of experts, 1.5B active) and so answers a follow-up 2.5x sooner;
  both write about 20 tokens/s together. LFM2.5-8B thinks before it answers, Qwen3-4B does not.
- Qwen3-4B keeps a large state per token (147 KB at f16), so its memory grows with the agents: 4 x 8K is about what
  fits a 16 GB M1 (10.1 GB). LFM2.5-8B's hybrid layers keep its memory flat at about 10.5-11 GB for 1-4 agents.
- With the cache on, a follow-up reuses its conversation instead of reading it again (see above).

**LFM2.5-8B in Q4_K_M** (`make pull MODEL=LFM2.5-8B-Q4`, LiquidAI's own, 5.2 GB against the Q8_0's 9.0) on the M1,
2026-11-03: six checks (17 x 23, a capital, the r's in strawberry, Russian, a Python function that is run, a tool
call) all pass at Q8_0, Q4_K_M and Q4_0 alike, writing 32.9 / 50.5 / 52.6 tokens/s on them.  The same agents bench
(8K each, no cache), Q4_K_M against Q8_0:

| Agents | Prompt read, tok/s | Follow-up's first token | Follow-up written together, tok/s | GPU memory |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 510 / 321 | 4.5 / 4.3 s | 52.7 / 19.7 | 6.6 / 10.5 GB |
| 2 | 440 / 374 | 7.5 / 9.3 s | 32.0 / 19.9 | 7.1 / 10.7 GB |
| 3 | 420 / 366 | 10.5 / 14.1 s | 37.5 / 25.1 | 6.9 / 10.8 GB |
| 4 | 412 / 315 | 13.6 / 17.1 s | 33.2 / 23.5 | 7.2 / 10.9 GB |

On the M1 the Q4_K_M is the one to take: 1.3-2.7x the writing speed, a faster read, 4 GB less memory, the same
answers on these checks.

**LFM2.5-24B in i1-IQ3_XXS** (`make pull MODEL=LFM2.5-24B-IQ3`, mradermacher's, 9.3 GB against the Q3_K_M's 11.5;
macOS only, see ‖) on the M1, 2026-11-03, the same six checks, one run each:

| Quant | Size | Checks | Writing, tok/s |
| --- | ---: | ---: | ---: |
| Q3_K_M | 11.5 GB | 6/6 | 30.1 |
| **i1-IQ3_XXS** | 9.3 GB | 6/6 | **36.2** |
| UD-Q2_K_XL (Unsloth) | 8.0 GB | 6/6 | 35.6 |
| i1-Q2_K | 8.8 GB | 6/6 | 29.5 (rambles: 2.6x the tokens) |

The model is bound more by its compute than by reading its weights, so the smaller files gain little; i1-IQ3_XXS is
the M1's 24B.

Apple's on-device model (`make afm`, `"engine": "afm"`) took about 1.5 GB loaded and 3.2 GB while answering on the
M1, estimated from the Mac's wired memory (its system service's own memory is readable by root only); the Monitor
shows the same estimate.

The newer models keep far smaller states, and there 2 GB of conversation cache holds every agent's conversation. The
same bench, 8K context each, the follow-up's first token without the cache / with the automatic one (2 GB here; 99% of
the prompt from the cache), 2026-11-03:

| Model (M1, 16 GB) | 1 agent | 2 agents | 3 agents | 4 agents | Answer, tokens/s each |
| --- | ---: | ---: | ---: | ---: | ---: |
| LFM2.5-8B-A1B (Q8_0) | 4.3 s / 0.43 s | 6.5 s / 0.70 s | 8.8 s / 1.1 s | 10.9 s / 1.2 s | 34 (one), 29-31 (several) |
| LFM2.5-24B-A2B (Q3_K_M) | 10.1 s / 0.50 s | 15.1 s / 0.78 s | 20.3 s / 1.1 s | 25.3 s / 1.4 s | 31 (one), 22-25 (several) |
| GPT-OSS 20B (MXFP4) | 12.5 s / 0.83 s | 25.0 s / 1.4 s | does not fit | does not fit | 22 (one), 20-27 (two) |

The server sizes that cache itself when the config sets none, by the RAM beside the model: 2 GB on a 16 GB Mac for
these, 4 GB for Sheared-LLaMA (one agent's follow-up 9.8 -> 3.7 s; with 2-4 its conversations do not fit and it is
slower either way), 16 GB on a 128 GB Mac with Qwen3.6 (`"--conversation-cache-mib"` in the args sets it, `0` turns
it off; Strata.app's menu has 0.2-32 GB and auto). GPT-OSS 20B with 3-4 agents at 8K is past the M1's GPU memory
(13 GB) with or without the cache.

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

## How good is it?

`server/target/release/strata-eval CONFIG.json [CONFIG2.json ...] [--n 15] [--effort none]` scores models on the same
fixed tasks of public benchmarks, graded by code: HumanEval+ in Python, MultiPL-E's HumanEval in Rust and C++ (compiled
and run against their tests), BFCL v3 tool calls (simple, choose one of several, parallel) and IFEval instructions.
Overall = mean of code (Python, Rust, C++), tools and instructions. `strata-eval --selftest` checks the graders
against the benchmarks' own answers. M5 Max, 2026-11-02, 15 tasks per area, thinking off, greedy (results in
`bench/results/2026-11-02-eval`; with 15 tasks a score moves about ±10 between close models, `--n 40` separates them):

| Model | Overall | Code (Python / Rust / C++) | Tool calls | Instructions | Time |
| --- | ---: | ---: | ---: | ---: | ---: |
| GLM-5.3-Flash (Maya-S24), 2 slots | 93 | 93 (93 / 93 / 93) | 100 | 87 | 576 s |
| Qwen3.6-35B-A3B (MXFP4_MOE) | 91 | 93 (93 / 87 / 100) | 80 | 100 | 193 s |
| Qwen3-Coder-30B-A3B (Q4_K_M) | 90 | 91 (87 / 93 / 93) | 93 | 87 | 163 s |
| Qwen3.8-Flash-Next (Q2_0) | 88 | 84 (73 / 80 / 100) | 80 | 100 | 245 s |
| GPT-OSS 120B heretic (MXFP4) | 74 | 96 (87 / 100 / 100) | 53 | 73 | 301 s |

- GPT-OSS writes the best code here but makes one tool call where a task needs two or three at once, and breaks
  rules like "no commas"; Qwen3.6 sometimes answers from memory instead of calling the tool; Qwen3-Coder calls tools
  best.

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
| Qwen3-Coder-30B-A3B (Q8_0) | 32.5 GB | 49 GB at 128K, 1 agent (measured) | Unsloth's, made for code, no thinking; about 80 tokens/s for one agent and no more for several ([above](#how-fast-is-it)). `make pull MODEL=QWEN3-CODER-30B` (it writes its run config) |
| Qwen3-Coder-30B-A3B (Q4_K_M) | 18.6 GB | 35 GB at 128K, 1 agent (measured) | About 90-105 tokens/s for one agent, 110 together for three; the better pick of the two. `make pull MODEL=QWEN3-CODER-30B-Q4` (it writes its run config) |
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
  back answer with the page itself, and each tool result says how many tool calls are left for the answer. Clicks and
  typing are the mouse and keyboard's own (AppKit events into the pane: trusted, as sites such as Zillow need): the
  pointer moves onto the element and presses there, and typing goes in as a paste into the field with its text
  selected, then Enter. A page that asks to prove a person is there (a press-and-hold, a captcha, "unusual traffic")
  is said as such: the model asks you to do it in the pane, and never tries to pass it. The model is told to get
  around a site by its own search field and links rather than made-up addresses.
- **Tables on this Mac, synced to a Google Sheet:** `table_upsert` / `table_read` / `table_delete` keep a table as a
  CSV file in `Strata-data/sheets/<name>.csv` (RFC 4180: quoted fields with commas, quotes and line breaks), rows by a
  key column (Link by default): instant, no browser, a file to diff and back up. `table_sync` puts it in its Google
  Sheet: first your own columns (Checkbox, Confirmation; `mine` to change them) and rows you added there come into the
  file, then the whole table goes in with one paste, rows it no longer has are removed, and the Checkbox column gets
  tick boxes. Rows removed with `table_delete` are remembered (`<name>.csv.deleted`) so a sync does not bring them
  back. A paste, not File > Import: an import resets the cells' formats. `maps_export` keeps the map on this Mac too: its KML
  (My Maps' own export: each place with the point Google placed it at and all its columns) as
  `Strata-data/maps/<name>.kml`, a GeoJSON copy (RFC 7946, longitude first), and the coordinates into the table of
  the same name as Lat / Lng columns by Link, so the table knows where each place is.
- **The model finishes what it says and checks it:** every MCP tool call is checked against the tool's input schema
  before it runs (a missing required argument, a wrong type; an unknown name when the schema allows none, or beside
  another error): it is not run, and the model reads what the tool takes ("map_url is missing; it has no url. It
  takes: …"). An answer that ends by announcing what it will do ("I'll add all 17 listings…") with no tool call is
  told once to do it. An answer that changed something (create, append, write, delete, trash, import, add…) gets a
  check: **Checking the work…** under it, where the model reads back what it changed and marks each part of the
  request DONE or NOT DONE, fixing what is not (at most two checks, 4 more tool rounds). The log's `tool …` lines
  show each call's arguments. Tried on the M1 with LFM2.5-24B: a sheet created, two rows added, read back, all DONE,
  105 s. A tool that gives the same error twice in a row says so under the second ("repeating it will not
  help"), and a third such call is not run: a model once imported into a trashed map 8 times. My Maps' tools say
  when a map is in the trash, and list the maps there are with their addresses; a Google tool always answers (75 s at
  most), and one whose page was replaced under it runs again only when it had changed nothing (reading, the start of
  an import).
- **Google Sheets and My Maps (Strata.app):** through the same browser pane, signed in to Google as you (no Google
  API, no Cloud project): `sheets_create` (title and header row), `sheets_read` (rows numbered as in the sheet),
  `sheets_append` (under the last filled row, each value by its column's name, `{"Address": …}`: a name that is no
  column, or a list of cells shorter than the header, is refused with the columns, never shifted; first cells FALSE/TRUE
  become tick boxes), `sheets_write` (one row's cells by column name, the rest of the row kept), `sheets_delete_rows`, `sheets_clear`, `sheets_trash`, and `maps_create` (a map with that title already: its address, not a second map), `maps_import_sheet` (a sheet's rows
  as places: an address column to place them, a column to title them, by name or letter; a wrong one is answered
  at once with the sheet's columns; the map is loaded fresh for each import; every column kept in each place) and
  `maps_trash`. The trash tools wait for your **Allow** in the browser's strip. They run scripts in Google's pages
  (app/src/google.rs: a cell range by the address's `#range=`, a paste event for writing, a copy event for reading,
  the menus by mouse events). My Maps' Import opens Google's file picker from another origin (docs.google.com); a
  script the browser puts in every frame wakes only there and answers My Maps' page for a token made at each start, so
  `browser_read`/`click`/`type` also work inside the picker. Tried on the M1 (30 Oct 2026): a sheet of 18 downtown San
  Diego studios and a map of them; create 10 s, append 6 s, read 2–3 s, map create 7 s, import 9 s. Google's pages
  change: when a tool says it found no button or menu item, that is where. The log's `tool …` line has the call's
  arguments (cut at 300 characters) and, for an error, what the tool said. Tests: `cargo test` in app/ runs the tools'
  scripts in headless Chrome against a small fake Google Sheet (served on 127.0.0.1, the editor, Name Box, `#range=`,
  paste, copy, Delete and menus as they behaved live), one Chrome at a time.
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
