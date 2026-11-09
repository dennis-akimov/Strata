# Slots of different sizes in one engine (own slots + router): feasibility spike

Status: spike done 2026-11-03.  Verdict: **no-go** on llama.cpp's current kernels.  Nothing kept in the code.

## Idea

One engine with slots of different sizes (say 32K + 64K + 64K + 128K).  Each request goes to the smallest free slot
that holds it, the way strata-router picks a server (docs/CONTEXT_POOL.md).  Agents share one batch, and each slot's
attention walks only its own cells.  The hope was the pool's memory (288K summed) at the own-slots speed.

## What it needs, and what was measured

**1. Memory per slot.** Own slots are one K/V tensor per layer, `[embd x size x streams]`, so every slot is as large
as the largest.  Could oversized slots cost nothing until they are used?  A scratch Metal test said no.  Allocating a
16 GB buffer committed 0 GB.  The GPU's first read of 1 GB of it committed **all 16.1 GB**.  Ragged slots therefore
need a separate buffer (tensor) per slot and layer, in llama-kv-cache.cpp:
- the K/V tensors (lines 233-244);
- `get_k`/`get_v` views, whose strides assume one size;
- the `set_rows` index into the flattened cache;
- the masks;
- stream copy (`seq_cp` between streams of different sizes);
- the state save/load.

**2. Attention per slot.** Batched flash attention takes one 4-D K/V tensor of a single length.  Separate
tensors mean one attention call per slot, with the outputs concatenated.  This was tested as a temporary edit to
`build_attn_mha` behind `STRATA_ATTN_PER_STREAM=1`.  The edit kept the shared tensor but split attention by stream.
The engine suite passed 23/23 with it on.  The decode test was the CONTEXT_POOL.md decode run: Qwen3.6-35B-A3B,
4 slots x 128K, conversations of 20K / 40K / 50K / 100K, all 4 writing 512 tokens, M5 Max.

| Run | Batched attention | One call per slot |
| --- | ---: | ---: |
| 1 (batched first) | 53.0 tok/s | 48.4 tok/s |
| 2 (per-slot first) | 60.5 tok/s | 36.0 tok/s |
| CONTEXT_POOL.md, for reference | own slots 58.1 | pool 47.5 |

Per-slot attention costs **9-40%** (the runs are noisy).  That is never better than the pool's 18%.  Raw output is in
this doc only: the spike's rows were not kept.

## Verdict

Ragged slots would match the pool on memory and at best tie it on speed.  The build would be a large patch to
llama-kv-cache, carried against every llama.cpp update.  So, no.

The real fix would be a ragged flash-attention kernel: one launch, with a K/V pointer and length per sequence, as
in vLLM's PagedAttention.  That is a Metal kernel in ggml, not a Strata change.

What stays:
- own slots where speed matters (Qwen3.6: small K/V);
- `--kv-unified --kv-pool` where memory decides (GLM-5.3, GPT-OSS 120B at long contexts);
- strata-router only across machines.

There is one upside the spike did not measure.  Ragged slots would let a 20K slot walk 20K cells instead of the longest
slot's length.  That pays only once attention, not reading the weights, dominates decode.  It does not at these lengths
on Qwen3.6.
