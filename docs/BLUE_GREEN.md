# Blue-green: a new Strata in place of the running one, agent by agent

Status: implemented 2026-11-03.  `make app-run` runs `Strata --deploy` (app/src/deploy.rs).

## Problem

Restarting Strata to run a new build cut off every request in flight and lost every agent's conversation state: each
agent's next request read its whole prompt again (10-30 s at 10-30K tokens).  With 4 agents the model is rarely idle.

## Design

1. **The port** (strata-server, `--handoff`): every server binds with SO_REUSEPORT; macOS hands each new connection
   to the newest listener (measured).  The new server (green) loads its model first, then binds beside the old one
   (blue), so from that moment new requests go to green; it then sends blue SIGUSR2.
2. **Draining** (blue, SIGUSR2): no new connections; the requests in flight finish (axum's graceful shutdown), then
   blue stops the usual way.
3. **The agents' state, one by one** (strata-metal `EXPORT` / `IMPORT`): blue writes each conversation it holds -
   the parked ones and every slot not in a request - to `$TMPDIR/strata-handoff-<port>/<content hash>.sseq` at once,
   and each slot again as its request ends; green loads each file into its conversation cache and deletes it.  An
   agent's next request then finds its state on green (the same content keys that decide any cache hit) instead of
   reading its prompt again.  Same model file, same build of the state format: a state llama.cpp refuses is dropped and
   that agent reads its prompt as before (slower, never wrong).
4. **The decision** (`Strata --deploy`): idle (or nothing runs) - restart as before; busy, blue shares its port and
   a second engine fits (its contexts; the weights are the same mapped file) - blue-green; otherwise wait for idle.
   Another Strata engine running beside it stops the deploy before anything quits (the new server would refuse to
   start).  Done when the new server answers.

## Limits

- Two engines run during the handoff: the contexts twice (Qwen3.6 at 64K x 4: about 9 GB more).  GLM-5.3 at 128K
  does not fit on 128 GB: it waits for idle.
- The files are written on the SSD (a 30K-token Qwen3.6 conversation: ~200 MB); EXPORT pauses blue's decoding while
  it writes.
- A client whose next request comes before its slot's file is imported reads its prompt once more.

## Verification

Unit tests: the deploy plan, the memory check, port sharing (real sockets), the file format round trip.  Live: an
answer in flight completes on blue while green answers new requests; an idle agent's next request on green reuses
its prompt (`cached_tokens` > 0).
