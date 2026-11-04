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
  {"name": "zed", "user_agent": "Zed/", "max_context": 6144}
]
```

| Key | Meaning | Default |
| --- | --- | --- |
| `name` | in logs and the Clients tab | `client1`, ... |
| `user_agent` | the requests whose User-Agent contains this text (case ignored) go to this engine | required |
| `paths` | only these API paths | every generation path: `/v1/chat/completions`, `/v1/completions`, `/v1/messages`, `/v1/responses` |
| `max_context` | the engine's context (prompt + answer): larger requests get the usual HTTP 400 there | 8192 |
| `parallel` | its slots | 1 |
| `cache_mib` | its conversation cache | 0 (off) |
| `port` | its port (127.0.0.1) | the main port + 100 + its place in the list |
| `url` | send to a server already running there instead of starting one | none |

**The engine** (no `url`): the main server starts a second `strata-server` with the same config except `max_context`,
`parallel`, `cache_mib` and the port.  The pool (`--kv-unified`, `--kv-pool`), the vision encoder and the MCP servers
are left out.  The weights are the same mapped file, so it costs its context and compute buffers (Qwen3.6 at 6K: under
1 GB).  The child server skips the firewall history, voice and scheduled prompts (the main server runs those).  It
stops with the main server, and on its own when the main server is gone (it checks its parent every 2 s).

**The routing** (http.rs): a POST to one of the entry's paths whose User-Agent matches goes through to that engine,
headers and body as sent, and its answer streams back.  The first entry that matches wins.  The engine still loading,
or gone: HTTP 503 with `Retry-After`, so the client's limit is never lifted by falling back to the main engine.

**The Clients tab:** the routed requests count for their client as usual: tokens from the answer's `usage` (and
Strata's `timings`), plus an Engine column (`main`, or the entry's name).

## Limits

- The two engines do not share a batch: each reads the weights for its own tokens.  A busy small engine slows the main
  one, as with strata-router (docs/CONTEXT_POOL.md: three engines at once read prompts ~3x slower).
- Blue-green (`make app-run`): the new main server starts its own child with `--handoff` on the same port, so the old
  child drains like the old main server.
- The match is a User-Agent substring: a client that hides its User-Agent cannot be routed.  The research found no
  better fixed signal; LiteLLM and Envoy's AI gateway route by configured header matches the same way.
