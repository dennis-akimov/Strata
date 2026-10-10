# Security

## Reporting a vulnerability

This is the Apple Silicon fork of Strata. Please report anything that should not be public until it is fixed
**privately**, through GitHub's private vulnerability reporting on this repository: its **Security** tab, **Report a
vulnerability** ([direct link](https://github.com/dennis-akimov/Strata/security/advisories/new)). Say what you found,
how to reproduce it (the request, the config keys involved, the version) and what an attacker gains.

A problem that is also in the original Strata (the server, the Windows or Linux engines) should go to the original as
well: [github.com/Niko1221/Strata/security](https://github.com/Niko1221/Strata/security/advisories/new).

Ordinary hardening ideas and findings that are safe to discuss in public are welcome as an
[issue](https://github.com/dennis-akimov/Strata/issues) or a pull request.

Supported: the latest commit on `main`. Fixes go there; there are no separate releases to patch.

## What the server exposes

Strata runs a model on your Mac and serves it over HTTP (`serve/server.py`). Out of the box it is reachable from
this computer only. The details and every setting are in [docs/DETAILS.md](docs/DETAILS.md) ("From other devices",
"Host names", "Web pages without an API key", "Tools from MCP servers").

- **Where it listens.** `127.0.0.1` by default. `--host 0.0.0.0` (or `"host"` in `strata-<model>.json`) opens it to
  your network, and the server then warns when no API key is set.
- **API key.** When `"api_key"` is set in the run config (or `STRATA_API_KEY`), it is required on `/v1/*` and on every endpoint
  that shows the model's state, requests or answers (`/status`, `/metrics`, `/settings`, `/mcp`, `/props`, `/slots`,
  `/api/requests`, `/config`) and on every `POST`. It is compared in constant time. Set one before you open the
  server to your network or put a tunnel in front of it.
- **Host check (DNS rebinding, 0.1.38).** Without an API key the server answers only requests whose `Host` is a name
  it knows (`localhost`, an IP address, the address it listens on and, when it listens beyond this computer, its
  name). Others get 403. `"allowed_hosts"` adds names; with an API key the key decides.
- **Origin check (0.1.38).** Without an API key, a browser `POST` to `/v1/*` from another web site's page (it carries
  an `Origin` header) gets 403 unless that origin is Strata's own page, `localhost`, an allowed host, or listed in
  `"trusted_origins"` / `"cors_origins"`. Changing settings (`/settings`, and `/config`: the few run config keys the
  page's Model settings may change - never the network, key, MCP or program keys), `/load` and `/unload` and the
  MCP tools are accepted only as JSON from Strata's own page (or `"trusted_origins"`), so a page elsewhere cannot
  change settings or run tools.
- **CORS** is off unless `"cors_origins"` lists origins, and then only for `/v1/*`.
- **MCP tools** are opt-in: only the servers you put in the run config, only for requests from Strata's own page
  that ask for them. They run with your user's rights, and the model decides when to call them.
- **The request monitor** (`/api-monitor`, which keeps the last prompts and answers in memory) is off unless
  `"api_monitor": true` is set.

Strata has not had an outside security audit yet.
