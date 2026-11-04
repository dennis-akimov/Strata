# A client's own engine: route a client by its User-Agent to a small engine of its own

Status: implemented 2026-11-03.

## Problem

Zed's requests (code completion, short prompts, often) wait behind the long agents in the main engine's slots, and the
main engine's context (32K-128K a slot) is far more than they need.  Wanted: one small engine for such a client, its
context capped (6K), configured in the run config.

## Design

The run config's `"clients"`: a list, each entry one client's engine.

```json
"clients": [
  {"name": "zed", "user_agent": "Zed/", "max_context": 6144, "parallel": 2}
]
```

| Key | Meaning | Default |
| --- | --- | --- |
| `name` | in logs and the Clients tab | `client1`, ... |
| `user_agent` | the requests whose User-Agent contains this text (case ignored) go to this engine | required |
| `paths` | only these API paths | every generation path: `/v1/chat/completions`, `/v1/completions`, `/v1/messages`, `/v1/responses` |
| `max_context` | the engine's context (prompt + answer): larger requests get the usual HTTP 400 there | 8192 |
| `parallel` | its slots (2 for Zed: its chat and its completions each keep their prompt) | 1 |
| `cache_mib` | its conversation cache | 0 (off) |
| `port` | its port (127.0.0.1) | the main port + 100 + its place in the list |
| `url` | send to a server already running there instead of starting one | none |

**The engine** (no `url`): the main server starts a second `strata-server` with the same config except `max_context`,
`parallel`, `cache_mib` and the port.  The pool (`--kv-unified`, `--kv-pool`), the vision encoder and the MCP servers
are left out.  The weights are the same mapped file, so it costs its context and compute buffers (Qwen3.6 at 6K: under
1 GB).  The child server skips the firewall history, voice and scheduled prompts (the main server runs those).  It
stops with the main server, and on its own when the main server is gone (it checks its parent every 2 s).

**The routing** (http.rs): a POST to one of the entry's paths whose User-Agent matches goes through to that engine,
headers and body as sent, and its answer streams back.  The first entry that matches wins.  Only a request that fits
the engine goes there: its body counted with the model's tokenizer (a little over the prompt), plus room to answer
(`max_tokens`, or 256 when the run config has `"fit_max_tokens": true`), within `max_context`.  A bigger one (a long
Zed agent thread) goes to the main engine instead of failing there with a 400.  Measured 2026-11-04: Zed's chat
requests (`/v1/chat/completions`, 2.8K tokens, 19 written) waited 9 s on average for a first token in the main engine
behind deep research and agents, when the entry routed only `/v1/completions`; leave `"paths"` out to route them all.  The engine still loading,
or gone: HTTP 503 with `Retry-After`, so the client's limit is never lifted by falling back to the main engine.

**The Clients tab:** the routed requests count for their client as usual: tokens from the answer's `usage` (and
Strata's `timings`), plus an Engine column (`main`, or the entry's name).

## VIP: the client first

`"vip"` (default `true`): while any of the client's requests runs, the main server sends its engine `VIP 1` and
the engine gives way between its passes (prompt pieces, batch windows, solo decode steps) until `VIP 0`.  This is
capped at 30 s a wait, so a lost `VIP 0` never hangs it.  Only while `VIP 1` holds does it read prompts in
`--prefill-share` pieces (512); otherwise in whole 2,048-token chunks, so a client that arrives mid-chunk waits for
that chunk's end (about 1 s deep into a long prompt) and then has the GPU.
macOS has no GPU priority between processes; this is the main engine stepping aside.

Measured 2026-11-03 on the M5 Max, Qwen3.6 main engine, Zed's `/v1/completions` (a ~1.5K-token prompt, 24 tokens),
while the main engine read a new 34K-token prompt:

| | Zed alone | Zed during the main read | The main read alone |
| --- | ---: | ---: | ---: |
| `"vip": false` | 0.24 s | 1.1-1.6 s | 17.2 s |
| `"vip": true` | 0.29-0.36 s | 0.27-0.64 s | 27 s (512-token pieces) |

The "vip: true" row read every prompt in pieces (the first version); pieces always cost Qwen3.6 a third of its
prompt speed, so now they are used only while the client's request runs.  `"vip": false` turns all of it off.

## Limits

- The two engines do not share a batch: each reads the weights for its own tokens.  A busy small engine slows the main
  one, as with strata-router (docs/CONTEXT_POOL.md: three engines at once read prompts ~3x slower).
- Blue-green (`make app-run`): the new main server starts its own child with `--handoff` on the same port, so the old
  child drains like the old main server.
- The match is a User-Agent substring: a client that hides its User-Agent cannot be routed.  The research found no
  better fixed signal; LiteLLM and Envoy's AI gateway route by configured header matches the same way.
