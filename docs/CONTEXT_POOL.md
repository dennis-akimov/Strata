# Agents with different context needs: a router, or one shared pool

Status: both implemented 2026-11-03; A/B below.

## Problem

`"parallel": N` with `--max-context C` gives every agent its own C tokens of K/V, held from the start.  Agents differ
(a 20K helper, a 100K main agent); sizing every slot for the largest wastes memory, sizing them smaller refuses the
big agent (HTTP 400, "request exceeds the context window", seen 2026-11-03 at 63,912 + 32,000 > 65,536).

## Options

**B - a router** (`strata-router`, server/src/bin/strata-router.rs): servers with different contexts (say 1 x 32K,
2 x 64K, 1 x 128K, the weights one mapped file), each request to the smallest whose context holds its prompt (bytes /
3) and answer (max_tokens).  Costs: agents in different engines do not share a batch (each engine reads the weights
for its own tokens), and a conversation that outgrows its server moves to a larger one and is read there again.

**C - one pool** (`strata-metal --kv-unified --kv-pool N`): llama.cpp's unified K/V cache, the way vLLM's
PagedAttention and llama-server work: one pool of N cells all sequences draw from, each request capped at
--max-context.  One batch for all agents; any agent can be the big one.  Copying a sequence (an idle slot's
conversation to another slot) shares its cells (llama_memory_seq_cp) instead of writing a state.  When the pool is
full a batch window fails and its requests end (ponytail: no admission against the pool yet).

## The "unified cache reads wrong" note, resolved

strata-metal kept own cells per sequence since a copied sequence rolled back to a checkpoint "read wrong in the
unified cache (measured)".  Rechecked: with real prompts, two slots decoded together match their solo answers 3 of 4
times in **both** modes (Qwen3-Coder-30B Q4: plain differs at token 16 of 40 in one, unified at token 37 in one) -
batching changes the arithmetic's last bits in either mode.  The engine test's made-up prompts (ids 300-340...) tie
their top tokens, so that alone flipped greedy answers after a few tokens.  The engine suite passes 23/23 in both
modes; test_batch_slots compares the first 4 tokens under `--kv-unified` (a slot reading another's row shows from the
first token).

## A/B

4 agents at once (~20K / 40K / 50K / 100K-token conversations, a first request and a follow-up of 128 tokens each),
Qwen3.6-35B-A3B MXFP4, M5 Max, conversation cache off:

| Layout | Memory (engine) | Wall | First token, cold: 20K / 40K / 50K / 100K | Follow-up's first token |
| --- | ---: | ---: | --- | --- |
| A: 4 slots x 128K | 35.5 GB | 273 s | 14 / 107 / 68 / 258 s | 0.4-1.1 s |
| B: router, 1 x 32K + 2 x 64K + 1 x 128K | (74.5 GB summed: each engine counts the shared weights) | 259 s | 42 / 86 / 219 / 247 s | 0.7-1.7 s |
| C: pool 288K, each up to 128K | **29.2 GB** | 255 s | 14 / 234 / 200 / 153 s | 0.8-2.4 s |

Reading ~210K prompt tokens dominates: the cold first token is mostly a request's turn in the queue (prompts are read
one at a time in every layout).  Follow-ups reuse ~100% everywhere.  C holds the same agents in 6.3 GB less than A,
any of them up to 128K; B gains nothing on one GPU (its engines compete instead of batching: the 20K agent's first
token 42 s against 14 s).  Qwen3.6 keeps K/V on 10 of its 40 layers, so the gain is small here; models with a big K/V
per token (GLM-5.3, GPT-OSS 120B) are where a pool matters.  One run each: differences under ~10% are noise.
Raw rows: bench/results/2026-11-03-context-ab.jsonl.

### Decode (the conversations read first, then all 4 agents write up to 512 tokens at once)

| Layout | Together | Each | Follow-up's first token |
| --- | ---: | ---: | --- |
| A: 4 slots x 128K | **58.1 tok/s** | 14.7-15.2 | 0.3-1.5 s |
| B: router | 47.2 tok/s | 12.2-14.5 | 0.4-4.0 s |
| C: pool 288K | 47.5 tok/s | 12.0-12.5 | 0.6-2.2 s |

The pool decodes 18% slower: in llama.cpp's unified cache each sequence's attention walks the whole pool's cells
(the others' masked), where a slot of its own walks only its own.  So: a pool where memory decides (a model whose K/V
for the agents' contexts would not fit as fixed slots: GLM-5.3, GPT-OSS 120B at long contexts), fixed slots where
speed does (Qwen3.6: small K/V).  The router is slower in both tests on one GPU.  Raw: bench/results/2026-11-03-context-decode.jsonl.

