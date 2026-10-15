"""serve/mcp.py - tools from MCP (Model Context Protocol) servers for the web app's chats.

The user lists servers in the run config (`"mcp_servers"`, or Claude Desktop's `"mcpServers"` block pasted as is) or
in a separate file (`--mcp-config`).  Each one is either a program Strata starts (`command`, `args`, `env`, `cwd`:
the stdio transport, newline-delimited JSON-RPC 2.0 on its stdin/stdout) or an address (`url`, `headers`: the
Streamable HTTP transport, JSON-RPC POSTed, answered as JSON or as an event stream).  Both are implemented here with
the standard library only; no MCP SDK is needed.

`McpHub` owns every configured server: it starts them in the background when the server starts, offers their tools to
the model as OpenAI function tools named `<server>__<tool>` (two servers may both have a `search`), and routes a call
back to the server that has it.  A server that does not start is left out with one log line; the chat works without
it.  A server that stops later (a crash) is started again on its next call.

Tool results reach the model as text: the MCP content blocks joined, capped at `max_result_chars` (a directory tree or
a web page can be megabytes, far more than a context holds) with a note saying so.  A failed call - the tool's own
error, a crash, a timeout - becomes a result that starts with "error:", so the model can react instead of the chat
breaking.
"""
from __future__ import annotations

import collections
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

from serve.winjob import contain

PROTOCOL = "2025-06-18"               # the MCP revision Strata asks for; the server's answer is used as given
DEFAULTS = {"timeout_s": 60.0, "max_result_chars": 20000, "max_rounds": 8, "start_timeout_s": 120.0}


class McpError(RuntimeError):
    """A server answered with an error, stopped, or could not be reached."""


class McpTimeout(McpError):
    pass


class McpCancelled(McpError):
    """The chat was stopped while the call ran."""


class _Slot:
    """One request waiting for its response."""

    def __init__(self):
        self.done = threading.Event()
        self.msg = None
        self.error = None


def _wait(slot: _Slot, timeout: float, cancel: threading.Event | None, what: str):
    """Wait for a response, checking the chat's stop button and the deadline a few times a second."""
    deadline = time.monotonic() + timeout
    while not slot.done.wait(0.05):
        if cancel is not None and cancel.is_set():
            raise McpCancelled("stopped")
        if time.monotonic() > deadline:
            raise McpTimeout(f"{what} timed out after {timeout:g} s")
    if slot.error:
        raise slot.error
    msg = slot.msg
    if "error" in msg:
        err = msg["error"] if isinstance(msg["error"], dict) else {"message": str(msg["error"])}
        raise McpError(f"{err.get('message') or 'error'} (code {err.get('code')})")
    return msg.get("result") or {}


# ------------------------------------------------------------------------------------------------ transports
class StdioTransport:
    """A server Strata starts: JSON-RPC messages one per line on its stdin and stdout; its stderr is its log."""
    kind = "stdio"

    def __init__(self, name: str, cfg: dict):
        self.name, self.cfg = name, cfg
        self.proc = None
        self.lock = threading.Lock()                    # the pending requests
        self.write_lock = threading.Lock()              # stdin; separate, so a full pipe can't block the reader
        self.pending: dict[int, _Slot] = {}
        self.next_id = 0
        self.stderr = collections.deque(maxlen=40)       # its last log lines, for the error message when it fails
        self.protocol = PROTOCOL

    def start(self):
        env = dict(os.environ)
        env.update({str(k): str(v) for k, v in (self.cfg.get("env") or {}).items()})
        command = str(self.cfg["command"])
        # On Windows `npx`, `uvx` and friends are .cmd files that CreateProcess finds only with their extension
        exe = shutil.which(command, path=env.get("PATH")) or command
        extra = {}
        if os.name == "nt":
            extra["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        else:
            extra["start_new_session"] = True           # its own process group, so close() ends its children too
        try:
            self.proc = subprocess.Popen([exe, *[str(a) for a in self.cfg.get("args") or []]], stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=self.cfg.get("cwd") or None,
                                         env=env, **extra)
        except OSError as e:
            raise McpError(f"could not start {command!r}: {e}") from None
        contain(self.proc)                              # ends with the server, however it ends (Windows)
        self._err_reader = threading.Thread(target=self._read_stderr, daemon=True)
        self._err_reader.start()
        threading.Thread(target=self._read, daemon=True).start()

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None and not getattr(self, "ended", False)

    @staticmethod
    def _lines(f):
        """A pipe's lines until it closes (or close() closed it under us)."""
        try:
            yield from f
        except (OSError, ValueError):
            return

    def _read_stderr(self):
        for raw in self._lines(self.proc.stderr):
            line = raw.decode("utf-8", "replace").rstrip()
            if line:
                self.stderr.append(line)
                if os.environ.get("STRATA_DEBUG"):
                    print(f"[mcp {self.name}] {line}", flush=True)

    def _read(self):
        for raw in self._lines(self.proc.stdout):
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:                           # a server that prints to stdout; the spec forbids it
                self.stderr.append(line)
                continue
            for m in msg if isinstance(msg, list) else [msg]:
                if isinstance(m, dict):
                    self._dispatch(m)
        self.ended = True
        # the server's last log line says why it stopped: let the stderr reader take it first (the pipe closes with
        # the process), or the error raced it and said only "the server stopped"
        self._err_reader.join(1.0)
        code = self.proc.poll()
        err = McpError(f"the server stopped{f' (exit code {code})' if code is not None else ''}{self._tail()}")
        with self.lock:
            slots, self.pending = list(self.pending.values()), {}
        for s in slots:
            s.error = err
            s.done.set()

    def _tail(self) -> str:
        return f": {self.stderr[-1][:300]}" if self.stderr else ""

    def _dispatch(self, m: dict):
        if "method" in m:
            if "id" in m:                                # a request from the server: we offer no client features
                try:
                    if m["method"] == "ping":
                        self._send({"jsonrpc": "2.0", "id": m["id"], "result": {}})
                    else:
                        self._send({"jsonrpc": "2.0", "id": m["id"],
                                    "error": {"code": -32601, "message": f"method not supported: {m['method']}"}})
                except McpError:
                    pass
            return                                       # notifications (log messages, list changes): not used
        with self.lock:
            slot = self.pending.pop(m.get("id"), None)
        if slot is not None:                             # an answer to a call that timed out is simply dropped
            slot.msg = m
            slot.done.set()

    def _send(self, msg: dict):
        data = json.dumps(msg, ensure_ascii=False).encode("utf-8") + b"\n"
        try:
            with self.write_lock:
                self.proc.stdin.write(data)
                self.proc.stdin.flush()
        except (OSError, ValueError):
            raise McpError(f"the server stopped{self._tail()}") from None

    def request(self, method: str, params: dict, timeout: float, cancel=None):
        if not self.alive():
            raise McpError(f"the server is not running{self._tail()}")
        slot = _Slot()
        with self.lock:
            self.next_id += 1
            rid = self.next_id
            self.pending[rid] = slot
        try:
            self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
            return _wait(slot, timeout, cancel, method)
        except (McpTimeout, McpCancelled) as e:
            with self.lock:
                self.pending.pop(rid, None)
            try:                                         # tell the server to stop working on it
                self.notify("notifications/cancelled", {"requestId": rid, "reason": str(e)})
            except McpError:
                pass
            raise

    def notify(self, method: str, params: dict | None = None):
        self._send({"jsonrpc": "2.0", "method": method, **({"params": params} if params is not None else {})})

    def close(self):
        """As the spec says: close its stdin, give it a moment to exit, then end it (and what it started)."""
        p = self.proc
        if p is None:
            return
        try:
            p.stdin.close()
        except OSError:
            pass
        try:
            p.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._kill_tree(p)
        for f in (p.stdout, p.stderr):
            try:
                f.close()
            except OSError:
                pass

    @staticmethod
    def _kill_tree(p):
        try:
            if os.name == "nt":                          # npx.cmd -> node: end the whole tree
                subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=10)
            else:
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
        except (OSError, subprocess.SubprocessError):
            pass
        try:
            p.kill()
        except OSError:
            pass


class HttpTransport:
    """A server at an address (Streamable HTTP): every message is a POST; the answer is JSON or an event stream that
    carries it.  The session id the server hands out at `initialize` rides on every later request."""
    kind = "http"

    def __init__(self, name: str, cfg: dict):
        self.name, self.url = name, str(cfg["url"])
        self.headers = {str(k): str(v) for k, v in (cfg.get("headers") or {}).items()}
        self.session = None
        self.protocol = None                             # sent as MCP-Protocol-Version once negotiated
        self.next_id = 0
        self.lock = threading.Lock()
        self.broken = None

    def start(self):
        pass

    def alive(self) -> bool:
        return self.broken is None

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
             "User-Agent": "strata", **self.headers}
        if self.session:
            h["Mcp-Session-Id"] = self.session
        if self.protocol:
            h["MCP-Protocol-Version"] = self.protocol
        return h

    def _post(self, msg: dict, rid, timeout: float, slot: _Slot | None):
        try:
            req = urllib.request.Request(self.url, data=json.dumps(msg, ensure_ascii=False).encode("utf-8"),
                                         headers=self._headers(), method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as r:
                if r.headers.get("Mcp-Session-Id"):
                    self.session = r.headers["Mcp-Session-Id"]
                if slot is None:                         # a notification: 202, nothing to read
                    return
                ctype = r.headers.get("Content-Type", "")
                if "text/event-stream" in ctype:
                    found = self._from_events(r, rid)
                else:
                    body = r.read()
                    found = None
                    parsed = json.loads(body) if body.strip() else None
                    for m in parsed if isinstance(parsed, list) else [parsed]:
                        if isinstance(m, dict) and m.get("id") == rid and "method" not in m:
                            found = m
                if found is None:
                    raise McpError("the server's answer did not contain a response")
                slot.msg = found
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", "replace")[:300]
            except OSError:
                pass
            if e.code == 404 and self.session:           # the session expired: a restart starts a new one
                self.broken = McpError("the server ended the session")
            err = McpError(f"HTTP {e.code} {e.reason}{': ' + detail if detail else ''}")
            if slot is None:
                raise err from None
            slot.error = err
        except (OSError, ValueError, McpError) as e:
            err = e if isinstance(e, McpError) else McpError(f"could not reach {self.url}: {e}")
            if slot is None:
                raise err from None
            slot.error = err
        finally:
            if slot is not None:
                slot.done.set()

    @staticmethod
    def _from_events(r, rid):
        """Read the event stream until the event that answers `rid` (it may carry notifications first)."""
        data = []
        for raw in r:
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            if line.startswith("data:"):
                data.append(line[5:].lstrip(" "))
            elif not line and data:
                try:
                    m = json.loads("\n".join(data))
                except ValueError:
                    m = None
                data = []
                for x in m if isinstance(m, list) else [m]:
                    if isinstance(x, dict) and x.get("id") == rid and "method" not in x:
                        return x
        return None

    def request(self, method: str, params: dict, timeout: float, cancel=None):
        if self.broken:
            raise self.broken
        with self.lock:
            self.next_id += 1
            rid = self.next_id
        slot = _Slot()
        # the POST runs on its own thread so the stop button and the deadline are noticed while it waits
        threading.Thread(target=self._post, args=({"jsonrpc": "2.0", "id": rid, "method": method, "params": params},
                                                  rid, timeout + 5, slot), daemon=True).start()
        try:
            return _wait(slot, timeout, cancel, method)
        except (McpTimeout, McpCancelled) as e:
            threading.Thread(target=self._quiet_notify, args=("notifications/cancelled",
                                                              {"requestId": rid, "reason": str(e)}), daemon=True).start()
            raise

    def _quiet_notify(self, method, params):
        try:
            self.notify(method, params)
        except McpError:
            pass

    def notify(self, method: str, params: dict | None = None):
        self._post({"jsonrpc": "2.0", "method": method, **({"params": params} if params is not None else {})},
                   None, 10, None)

    def close(self):
        if not self.session:
            return
        try:                                             # end the session on the server (it may not allow that)
            req = urllib.request.Request(self.url, headers=self._headers(), method="DELETE")
            urllib.request.urlopen(req, timeout=3).close()
        except (OSError, ValueError):
            pass


# ------------------------------------------------------------------------------------------------ one server
class McpServer:
    """One configured server: started once (and again after it stopped), with the tools it listed."""

    def __init__(self, name: str, cfg: dict, settings: dict, editable: bool = False):
        self.name, self.cfg, self.settings = name, cfg, settings
        self.kind = "http" if cfg.get("url") else "stdio"
        self.editable = editable and self.kind == "http"  # the page may change it: a URL in the config's mcp_servers
        self.status = "idle"             # idle -> starting -> ready | failed; ready -> stopped when it ends
        self.error = None
        self.tools: list[dict] = []
        self.info: dict = {}
        self.transport = None
        self.lock = threading.Lock()
        self.last_start = 0.0

    def start(self) -> bool:
        with self.lock:
            return self._start()

    def _start(self) -> bool:
        if self.transport is not None:
            self.transport.close()
        self.status, self.error, self.last_start = "starting", None, time.monotonic()
        t = HttpTransport(self.name, self.cfg) if self.kind == "http" else StdioTransport(self.name, self.cfg)
        self.transport = t
        timeout = float(self.settings["start_timeout_s"])
        try:
            t.start()
            res = t.request("initialize", {"protocolVersion": PROTOCOL, "capabilities": {},
                                           "clientInfo": {"name": "strata", "title": "Strata", "version": "1"}},
                            timeout)
            t.protocol = str(res.get("protocolVersion") or PROTOCOL)
            self.info = {k: v for k, v in (res.get("serverInfo") or {}).items() if k in ("name", "title", "version")}
            self.info["protocol"] = t.protocol
            t.notify("notifications/initialized")
            tools, cursor = [], None
            for _ in range(100):                         # pages; a server that never stops paging is cut off
                try:
                    page = t.request("tools/list", {"cursor": cursor} if cursor else {}, timeout)
                except McpError as e:
                    if "code -32601" in str(e) and not tools:   # a server without tools (prompts/resources only)
                        break
                    raise
                tools += [x for x in page.get("tools") or [] if isinstance(x, dict) and x.get("name")]
                cursor = page.get("nextCursor")
                if not cursor:
                    break
            if self.cfg.get("tools") is not None:       # an allowlist: the model neither sees nor calls the others
                tools = [x for x in tools if x["name"] in self.cfg["tools"]]
            self.tools, self.status = tools, "ready"
            names = ", ".join(x["name"] for x in tools[:8]) + (", ..." if len(tools) > 8 else "")
            print(f"[strata] MCP server {self.name!r}: {len(tools)} tool{'s' * (len(tools) != 1)}"
                  f"{' (' + names + ')' if tools else ''}", flush=True)
            return True
        except McpError as e:
            self.status, self.error, self.tools = "failed", str(e), []
            url = str(self.cfg.get("url") or "")
            if self.kind == "http" and str(e).startswith("HTTP 404") and \
                    re.fullmatch(r"https?://[^/?#]+/?", url.split("?", 1)[0]):
                # the address without a path: most servers answer MCP at /mcp, their root is a 404
                self.error += f" - the URL has no path; most MCP servers are at {url.split('?', 1)[0].rstrip('/')}/mcp"
            t.close()
            print(f"[strata] MCP server {self.name!r} did not start: {e}. Its tools are left out; the chat works "
                  "without them.", flush=True)
            return False

    def call(self, tool: str, arguments: dict, timeout: float, cancel=None) -> dict:
        with self.lock:
            if self.transport is None or not self.transport.alive():
                # it stopped since its last call (a crash, an expired session): start it again; one that failed to
                # start is tried again at most every 10 s, so a broken command does not run on every call
                if self.status == "failed" and time.monotonic() - self.last_start < 10:
                    raise McpError(f"the server is not running ({self.error or 'it stopped'})")
                print(f"[strata] MCP server {self.name!r} had stopped; starting it again", flush=True)
                if not self._start():
                    raise McpError(f"the server could not be started again: {self.error}")
            t = self.transport
        try:
            return t.request("tools/call", {"name": tool, "arguments": arguments}, timeout, cancel)
        except McpError as e:
            if not t.alive():
                self.status, self.error = "stopped", str(e)
            raise

    def close(self):
        with self.lock:
            if self.transport is not None:
                self.transport.close()
            if self.status in ("ready", "starting"):
                self.status = "stopped"


# ------------------------------------------------------------------------------------------------ the hub
def result_text(result: dict) -> str:
    """An MCP tool result as the text the model reads: the text blocks as they are, other blocks named."""
    parts = []
    for block in result.get("content") or []:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            parts.append(str(block.get("text", "")))
        elif kind == "resource":
            res = block.get("resource") or {}
            parts.append(str(res["text"]) if "text" in res else f"[resource {res.get('uri', '')} "
                                                                  f"({res.get('mimeType', 'binary')}), not shown]")
        elif kind == "resource_link":
            parts.append(f"[resource link: {block.get('name') or ''} {block.get('uri', '')}]".replace("  ", " "))
        elif kind in ("image", "audio"):
            size = len(block.get("data") or "") * 3 // 4
            parts.append(f"[{kind} ({block.get('mimeType', '?')}, {size:,} bytes): not shown to the model]")
        else:
            parts.append(json.dumps(block, ensure_ascii=False))
    if not parts and result.get("structuredContent") is not None:
        parts.append(json.dumps(result["structuredContent"], ensure_ascii=False))
    return "\n".join(parts)


def _clean(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(name)) or "x"


class McpHub:
    """Every configured server, and the merged, namespaced tool list the model sees."""

    def __init__(self, servers: dict[str, dict], settings: dict | None = None, editable=()):
        self.settings = {**DEFAULTS, **(settings or {})}
        self.servers = {name: McpServer(name, cfg, self.settings, name in editable) for name, cfg in servers.items()}
        self.threads: list[threading.Thread] = []
        self._routes: dict[str, tuple[McpServer, str]] = {}
        self.lock = threading.Lock()                    # self.servers, changed at run time by put() and remove()

    def put(self, name: str, cfg: dict) -> McpServer:
        """Add a URL server, or replace one the page may change, and start it in the background (the Settings
        view's connections: POST /mcp/servers)."""
        server = McpServer(name, cfg, self.settings, editable=True)
        with self.lock:
            old = self.servers.get(name)
            self.servers = {**self.servers, name: server}
        if old is not None:
            threading.Thread(target=old.close, daemon=True).start()
        t = threading.Thread(target=server.start, daemon=True, name=f"mcp-{name}")
        self.threads.append(t)
        t.start()
        return server

    def remove(self, name: str) -> bool:
        with self.lock:
            old = self.servers.get(name)
            if old is None:
                return False
            self.servers = {k: v for k, v in self.servers.items() if k != name}
        threading.Thread(target=old.close, daemon=True).start()
        return True

    def start(self, wait: bool = False):
        """Start every server on its own thread (npx may download a package first: that must not delay the chat)."""
        threads = [threading.Thread(target=s.start, daemon=True, name=f"mcp-{s.name}")
                   for s in self.servers.values()]
        self.threads += threads
        for t in threads:
            t.start()
        if wait:
            self.wait(None)

    def wait(self, timeout: float | None) -> None:
        """Wait (up to `timeout` s in all) for servers that are still starting."""
        end = None if timeout is None else time.monotonic() + timeout
        for t in list(self.threads):
            t.join(None if end is None else max(0.0, end - time.monotonic()))

    def routes(self) -> dict[str, tuple[McpServer, str]]:
        """tool name the model sees -> (server, the tool's own name).  A server that stopped after it was ready keeps
        its tools here: a call starts it again."""
        out = {}
        for s in list(self.servers.values()):           # put()/remove() swap the dict, never change it in place
            if s.status not in ("ready", "stopped"):
                continue
            for t in s.tools:
                name = f"{_clean(s.name)}__{_clean(t['name'])}"[:64]
                n = 2
                while name in out:                       # two names that differ only in cleaned characters
                    name = f"{name[:60]}_{n}"
                    n += 1
                out[name] = (s, t["name"])
        self._routes = out
        return out

    def template_tools(self, exclude=()) -> list[dict]:
        """The tools in the chat template's form ({name, description, parameters}); `exclude`: names the request
        brought itself (those win)."""
        out = []
        for name, (s, tool) in self.routes().items():
            if name in exclude:
                continue
            t = next(x for x in s.tools if x["name"] == tool)
            schema = t.get("inputSchema") if isinstance(t.get("inputSchema"), dict) else {}
            out.append({"name": name, "description": str(t.get("description") or t.get("title") or ""),
                        "parameters": schema or {"type": "object", "properties": {}}})
        return out

    def openai_tools(self) -> list[dict]:
        return [{"type": "function", "function": t} for t in self.template_tools()]

    def call(self, name: str, arguments: dict, cancel: threading.Event | None = None) -> dict:
        """Run one tool -> {"ok", "text" (what the model reads, capped), "chars" (its full length), "truncated",
        "ms", "server", "tool"}.  Errors become text starting with "error:"; only McpCancelled is raised."""
        t0 = time.monotonic()
        server, tool = self._routes.get(name) or self.routes().get(name, (None, name))
        out = {"server": server.name if server else None, "tool": tool}
        if server is None:
            text, ok = f"error: there is no tool named {name!r}", False
        else:
            try:
                res = server.call(tool, arguments if isinstance(arguments, dict) else {},
                                  float(self.settings["timeout_s"]), cancel)
                text, ok = result_text(res), not res.get("isError")
                if not ok:
                    text = "error: " + (text or "the tool reported an error")
            except McpCancelled:
                raise
            except McpError as e:
                text, ok = f"error: {e}", False
        cap = int(self.settings["max_result_chars"])
        full = len(text)
        if cap > 0 and full > cap:
            text = text[:cap] + (f"\n\n[... truncated: the tool returned {full:,} characters; only the first "
                                 f"{cap:,} are shown]")
        out.update(ok=ok, text=text, chars=full, truncated=cap > 0 and full > cap,
                   ms=int((time.monotonic() - t0) * 1000))
        return out

    def status(self) -> dict:
        """GET /mcp: every server, its state and its tools (for the web app)."""
        routes = self.routes()
        servers = []
        for s in list(self.servers.values()):
            names = {tool: n for n, (srv, tool) in routes.items() if srv is s}
            where = {}
            if s.kind == "http":                         # the address without its query (it may hold a token) and
                url = str(s.cfg.get("url"))              # the names of the headers sent: never their values
                where = {"url": url.split("?", 1)[0] + ("?…" if "?" in url else ""),
                         "headers": sorted(str(k) for k in (s.cfg.get("headers") or {}))}
            servers.append({"name": s.name, "transport": s.kind, "status": s.status, "error": s.error,
                            "info": s.info, "editable": s.editable, **where,
                            "tools": [{"name": names.get(t["name"], t["name"]), "tool": t["name"],
                                       "description": str(t.get("description") or "")[:300]} for t in s.tools]})
        return {"servers": servers, "tools": len(routes),
                "settings": {k: self.settings[k] for k in ("timeout_s", "max_result_chars", "max_rounds")}}

    def close(self):
        for s in list(self.servers.values()):
            try:
                s.close()
            except Exception:  # noqa: BLE001 - closing is best effort
                pass


# ------------------------------------------------------------------------------------------------ config
def _check_server(name, cfg, where) -> dict:
    if not isinstance(cfg, dict):
        raise SystemExit(f"[strata] {where}: MCP server {name!r} must be an object with \"command\" or \"url\"")
    if "tools" in cfg and not (isinstance(cfg["tools"], list) and all(isinstance(t, str) for t in cfg["tools"])):
        raise SystemExit(f"[strata] {where}: MCP server {name!r}: \"tools\" must be a list of tool names")
    if cfg.get("url"):
        if cfg.get("type") == "sse":
            raise SystemExit(f"[strata] {where}: MCP server {name!r} uses the old SSE transport (\"type\": \"sse\"); "
                             "Strata speaks Streamable HTTP - most servers offer it at /mcp")
        if not isinstance(cfg.get("headers") or {}, dict):
            raise SystemExit(f"[strata] {where}: MCP server {name!r}: \"headers\" must be an object")
        return cfg
    if not isinstance(cfg.get("command"), str) or not cfg["command"].strip():
        raise SystemExit(f"[strata] {where}: MCP server {name!r} needs \"command\" (a program to start) or \"url\"")
    if not isinstance(cfg.get("args") or [], list):
        raise SystemExit(f"[strata] {where}: MCP server {name!r}: \"args\" must be a list")
    if not isinstance(cfg.get("env") or {}, dict):
        raise SystemExit(f"[strata] {where}: MCP server {name!r}: \"env\" must be an object")
    return cfg


def servers_from(block, where: str) -> dict[str, dict]:
    """{"name": {...}} with the entries checked; `"disabled": true` leaves one out."""
    if block is None:
        return {}
    if not isinstance(block, dict):
        raise SystemExit(f"[strata] {where}: the MCP servers must be an object {{\"name\": {{...}}}}")
    return {str(n): _check_server(n, c, where) for n, c in block.items()
            if not (isinstance(c, dict) and c.get("disabled") is True)}


def settings_from(cfg: dict) -> dict:
    """The run config's optional `"mcp"` block: timeout_s, max_result_chars, max_rounds, start_timeout_s."""
    out = {}
    for key, value in (cfg.get("mcp") or {}).items():
        number = isinstance(value, (int, float)) and not isinstance(value, bool)
        if key in ("timeout_s", "start_timeout_s"):
            if not number or value <= 0:
                raise SystemExit(f"[strata] config mcp.{key}={value!r}: expected a number of seconds > 0")
            out[key] = float(value)
        elif key in ("max_result_chars", "max_rounds"):
            if not number or value != int(value) or value < 1:
                raise SystemExit(f"[strata] config mcp.{key}={value!r}: expected a positive integer")
            out[key] = int(value)
        else:
            print(f"[strata] config mcp.{key}={value!r}: unknown key, ignored", flush=True)
    return out


# headers the transport sets itself (or HTTP owns): a key can't be sent in one of them
_RESERVED_HEADERS = {"accept", "content-type", "content-length", "host", "connection", "transfer-encoding",
                     "mcp-session-id", "mcp-protocol-version", "user-agent", "cookie"}


def page_entry(req, old: dict | None = None) -> tuple[str, dict]:
    """POST /mcp/servers from the Settings view -> (name, the "mcp_servers" entry): a URL server only (a program
    server runs a command, so it stays in the config file).  `key` goes in the `header` the server reads
    ("Authorization" sends "Bearer <key>"); no `key` keeps the old one (the page never reads it back), "" drops it.
    ValueError with what is wrong."""
    if not isinstance(req, dict):
        raise ValueError("send a JSON object")
    name = req.get("name")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,39}", name):
        raise ValueError("name: 1-40 letters, digits, '_', '.', '-' (starting with a letter or digit)")
    url = req.get("url")
    if not isinstance(url, str) or len(url) > 2000 or not re.fullmatch(r"https?://[^\s/?#]+[^\s]*", url.strip()):
        raise ValueError("url: an http:// or https:// address (Streamable HTTP; most servers offer it at /mcp)")
    header = req.get("header", "Authorization")
    if not isinstance(header, str) or not re.fullmatch(r"[A-Za-z0-9-]{1,64}", header) or \
            header.lower() in _RESERVED_HEADERS:
        raise ValueError("header: a header name such as Authorization or X-API-Key")
    key = req.get("key")
    if key is not None and (not isinstance(key, str) or len(key) > 4096 or any(c in key for c in "\r\n\0")):
        raise ValueError("key: one line of text, at most 4096 characters")
    entry = {"url": url.strip()}
    if key is None:                                     # unchanged: the old entry's key, moved to the header asked for
        old_headers = list(((old or {}).get("headers") or {}).items())
        if old_headers:
            old_name, value = str(old_headers[0][0]), str(old_headers[0][1])
            if old_name.lower() == "authorization" and header.lower() != "authorization":
                value = value.split(" ", 1)[-1]          # "Bearer abc" -> "abc"
            elif header.lower() == "authorization" and old_name.lower() != "authorization" and " " not in value:
                value = f"Bearer {value}"
            entry["headers"] = {header: value}
    elif key.strip():
        key = key.strip()
        # a key is one token; only Authorization may carry a scheme first ("Bearer abc"): anything else with spaces
        # is pasted text (the status panel copied by mistake), refused here instead of failing later with a 401
        if (len(key.split()) > 2 or header.lower() != "authorization") and any(c.isspace() for c in key):
            raise ValueError("key: paste only the key itself (it has spaces; for Authorization "
                             "\"<scheme> <token>\" is fine)")
        value = key if header.lower() != "authorization" or " " in key else f"Bearer {key}"
        entry["headers"] = {header: value}
    return name, entry


def hub_from_config(cfg: dict, mcp_config_path: str | None = None) -> McpHub | None:
    """The run config's `"mcp_servers"` (or `"mcpServers"`) plus the servers in the --mcp-config file (Claude
    Desktop's format: {"mcpServers": {...}}); a name in both takes the file's entry.  None when there are none."""
    servers = {}
    servers.update(servers_from(cfg.get("mcp_servers"), "config mcp_servers"))
    servers.update(servers_from(cfg.get("mcpServers"), "config mcpServers"))
    if mcp_config_path:
        try:
            with open(mcp_config_path, encoding="utf-8-sig") as f:
                data = json.load(f)
        except (OSError, ValueError) as e:
            raise SystemExit(f"[strata] --mcp-config {mcp_config_path}: {e}") from None
        block = data.get("mcpServers", data.get("mcp_servers")) if isinstance(data, dict) else None
        if block is None:
            raise SystemExit(f"[strata] --mcp-config {mcp_config_path}: expected {{\"mcpServers\": {{...}}}}")
        servers.update(servers_from(block, f"--mcp-config {mcp_config_path}"))
    if not servers:
        return None
    # the page may change a URL server only when it lives in "mcp_servers", the block it writes (not one the
    # --mcp-config file or a pasted "mcpServers" block overrides)
    editable = {n for n, c in servers_from(cfg.get("mcp_servers"), "config mcp_servers").items()
                if c.get("url") and servers.get(n) is c}
    return McpHub(servers, settings_from(cfg), editable)


if __name__ == "__main__":                               # python -m serve.mcp config.json: list what a config offers
    hub = hub_from_config(json.load(open(sys.argv[1], encoding="utf-8-sig")) if len(sys.argv) > 1 else {},
                          sys.argv[2] if len(sys.argv) > 2 else None)
    if hub is None:
        sys.exit("no MCP servers configured")
    hub.start(wait=True)
    print(json.dumps(hub.status(), indent=1))
    hub.close()
