# Monitor: what it shows, and the seven additions of 2026-11-03

Status: implemented.  Sources: vLLM's production metrics (docs.vllm.ai/en/stable/usage/metrics), OpenTelemetry's GenAI
semantic conventions (gen_ai.server.time_to_first_token / time_per_output_token, histograms in seconds), and the
sudo-free Apple Silicon readers (macmon, siltop: IOReport energy counters, GPU P-state residency, the pmgr DVFS table).

## Problem

The Monitor showed the engine's speed and the hardware's load, but not what a user waits on or what went wrong:

1. A request the server refused (HTTP 4xx/5xx) left no trace outside the server's stdout, which Strata.app shows and
   then drops: three refusal-test runs ended on HTTP 400 with no way to see why.
2. Latency only as means and the last request; latency is read as percentiles (p50/p95), and the wait a user has is
   queue + prompt reading + the first token, not the decode speed.
3. "Context fill" was the last request's, while 4 agents each fill their own slot of the context.
4. Power was the GPU's alone (12 W at 100% load on an M5 Max is right for the GPU; the CPU drew 10 W and DRAM 7.6 W
   beside it), and nothing showed the GPU's clock (it ran in P2-P3 of 15 while "100% busy": decode waits on memory),
   swap, memory pressure or the thermal state.
5. PC-only cards on a Mac: PCIe "n/a", "Experts in VRAM –", an always-empty "VRAM hit rate" column.
6. MCP tools: 25 connected, no count of calls, failures or their time.
7. The conversation cache counted parks and restores but not their time, so a restore slower than reading the prompt
   again (the M1's 2-4 agents) could not be seen.

## Design

| # | Server (`/metrics` JSON, Prometheus) | Monitor tab |
| --- | --- | --- |
| 1 | Every JSON answer with status >= 400 counted by (status, type) and kept (last 50: time, path, status, type, message); logged as `HTTP <status> <path>: <message>`. `strata:http_errors_total{status,type}` | "Failed" rows in Recent requests (the message on hover), failures in the Latency card |
| 2 | Each request's `queue_s` (arrival to start, model load included) and `ttft_s` (arrival to first token) in its record; `latency`: p50/p95 of time to first token, queue, end to end, and the time per output token, over the kept requests | A Latency card |
| 3 | (the slots are in `live.slots` already) | One bar per agent: its prompt + answer of the context |
| 4 | IOReport "Energy Model": GPU, CPU and DRAM watts; "GPU Stats / GPU Performance States" residency x the `voltage-states9` DVFS table of `pmgr`: the GPU's clock (MHz, its top); `vm.swapusage`; `kern.memorystatus_vm_pressure_level`; the notify state `com.apple.system.thermalpressurelevel` | Power card: GPU, then CPU and DRAM; a GPU clock card instead of PCIe on a Mac; swap and pressure under RAM; the thermal state under the temperature |
| 5 | - | PCIe, Experts in VRAM and VRAM hit rate hidden where they cannot have a value |
| 6 | Per tool: calls, failures, total ms, the last error; `strata:mcp_tool_calls_total{tool,ok}` | Calls, failures and mean time beside each tool |
| 7 | The engine's cache log line gains `ms=` (the state copy out or in); parks/restores timed | "Park / restore time" (mean) |

Each value that cannot be read is null and its card or line is left out; nothing here can stop the server.

## Verification

Unit tests for the pure parts (percentiles, the DVFS table and the clock from residencies, the failure log, the cache
log's `ms=`, the MCP counters); then the live server: a refused request shows up in `/metrics`, the Prometheus text and
the Monitor; the clock, powers, swap and pressure read on this Mac.
