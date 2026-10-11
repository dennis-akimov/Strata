# Documentation

This repository is an **independently maintained, Apple Silicon-only fork of [Strata](https://github.com/Niko1221/Strata)**
(maintained by dennis-akimov). Its documents fall in three groups.

## Maintained here: the Mac

- **[MACOS.md](MACOS.md)**: install, limits and warnings, the models tested on a Mac (Qwen3.8-Flash-Next, GPT-OSS 120B),
  every measurement, and the developer notes for the Metal engine.

## From the original, about the shared server

These describe the Strata server and its APIs, which the Mac fork shares with the original. They are kept as the
original wrote them at the fork point. Where they mention graphics cards, CUDA, VRAM or speeds, that part is about PCs;
the Mac's own limits (memory, MTP, several requests at once) are in [MACOS.md](MACOS.md#limits-and-warnings).

| Document | What it covers |
| --- | --- |
| [DETAILS.md](DETAILS.md#using-it) | The APIs (OpenAI, Anthropic, Responses), settings and options |
| [MCP_SERVER.md](MCP_SERVER.md) | Strata's MCP server for AI assistants |
| [BATCHING.md](BATCHING.md) | Several requests at once (`"parallel"`) |
| [LOGIT_BIAS.md](LOGIT_BIAS.md) | Per-request token biases |
| [LLAMA_SWAP.md](LLAMA_SWAP.md) | Running Strata behind llama-swap |
| [MESSAGE_BOUNDARY_CACHE.md](MESSAGE_BOUNDARY_CACHE.md), [PROMPT_CACHE_TAIL.md](PROMPT_CACHE_TAIL.md) | How conversations are reused |
| [HOW_IT_WORKS.md](HOW_IT_WORKS.md) | The design, the credits and the licenses |

## From the original, for Windows and Linux PCs: not maintained here

The PC engines (NVIDIA CUDA, AMD HIP, Intel SYCL) are still in this repository as they were at the fork point, but they
are not built, tested or updated here. For a PC, use [the original](https://github.com/Niko1221/Strata) and
[its current documents](https://github.com/Niko1221/Strata/tree/main/docs); the copies kept here may be out of date:
[INSTALL.md](INSTALL.md), [MODELS.md](MODELS.md), [TROUBLESHOOTING.md](TROUBLESHOOTING.md), [AI_SETUP.md](AI_SETUP.md),
[MULTI_GPU.md](MULTI_GPU.md), [SECOND_GPU.md](SECOND_GPU.md), [OLDER_GPUS.md](OLDER_GPUS.md),
[NVIDIA_V100.md](NVIDIA_V100.md), [AMD_HIP.md](AMD_HIP.md), [AMD_HIP_PERFORMANCE.md](AMD_HIP_PERFORMANCE.md),
[STRIX_HALO.md](STRIX_HALO.md), [INTEL.md](INTEL.md), [INTEL_ARC.md](INTEL_ARC.md), [A770_SPEED_SPEC.md](A770_SPEED_SPEC.md),
[UNSLOTH_Q4.md](UNSLOTH_Q4.md), [UNSLOTH_Q6.md](UNSLOTH_Q6.md), [ORCA.md](ORCA.md), [ORCA_Q4_K_S.md](ORCA_Q4_K_S.md),
[VRAM_ELASTIC.md](VRAM_ELASTIC.md), [BATCHED_DMA.md](BATCHED_DMA.md), [DISJOINT_EXPERT_CACHE.md](DISJOINT_EXPERT_CACHE.md),
[EXCHANGE_ROTATION.md](EXCHANGE_ROTATION.md), [KV_PREFETCH.md](KV_PREFETCH.md), [MMVQ_IL_TABLE.md](MMVQ_IL_TABLE.md),
[RESEARCH_RUNS.md](RESEARCH_RUNS.md), [COMMUNITY_BENCHMARKS.md](COMMUNITY_BENCHMARKS.md),
[TEST_REQUESTS.md](TEST_REQUESTS.md), and [the paper](paper/Strata-Paper.pdf).
