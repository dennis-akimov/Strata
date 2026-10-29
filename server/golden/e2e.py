"""End-to-end parity: the Python server and the Rust server, both `--engine mock` with the same tokenizer and scripts,
get the same requests; their answers are compared after ids and clocks are normalized.

    python golden/e2e.py <rust binary> <tokenizer dir> [--engine-strata | --mcp] [--record NAME | --replay NAME]

--record NAME: the Python server's answers into tests/golden/e2e-NAME.json.gz (it ran against serve/server.py before
that was removed); --replay NAME: the Rust server against that recording (no Python server needed).
"""
import json, os, re, subprocess, sys, time, urllib.request, urllib.error, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
RUST, TOK = sys.argv[1], sys.argv[2]
RECORD = sys.argv[sys.argv.index("--record") + 1] if "--record" in sys.argv else None
REPLAY = sys.argv[sys.argv.index("--replay") + 1] if "--replay" in sys.argv else None
GOLDEN = pathlib.Path(__file__).resolve().parents[1] / "tests" / "golden"
ENGINE = "--engine-strata" in sys.argv
MCP = "--mcp" in sys.argv                       # both servers with the fake MCP server (stdio and HTTP)          # both servers on golden/fake_engine.py, through a run config
QWEN = "<|im_start|>" in (pathlib.Path(TOK) / "vocab.json").read_text(encoding="utf-8")[:10**9] if TOK != "-" else True
HARMONY = (pathlib.Path(TOK) / "vocab.json").exists() and "<|channel|>" in (pathlib.Path(TOK) / "token_type.json").read_text() or "gpt-oss" in TOK
CALL = ("<|channel|>analysis<|message|>need a file<|end|><|start|>assistant<|channel|>commentary to=functions.read_file <|constrain|>json<|message|>{\"path\": \"a.txt\"}<|call|>"
        if "gpt-oss" in TOK else
        "I should read it.</think>\n\n<tool_call>\n<function=read_file>\n<parameter=path>\na.txt\n</parameter>\n</function>\n</tool_call>")
ANSWER = ("<|channel|>analysis<|message|>Thinking about it.<|end|><|start|>assistant<|channel|>final<|message|>Hello from the mock engine. STOPHERE and more {\"a\": 1}<|return|>"
          if "gpt-oss" in TOK else "Thinking about it.</think>\n\nHello from the mock engine. STOPHERE and more {\"a\": 1}")
JSONA = ("<|channel|>final<|message|>{\"a\": 1, \"b\": \"é\"}<|return|>" if "gpt-oss" in TOK else "</think>\n\n```json\n{\"a\": 1, \"b\": \"é\"}\n```")
SCRIPTS = [ANSWER, CALL, ANSWER, JSONA] + [ANSWER] * 2
T2 = [{"type": "function", "function": {"name": "read_file", "description": "Read a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]

class _None:
    def terminate(self):
        pass

def start(cmd, port, env=None):
    if REPLAY and cmd[0].endswith("python") and str(ROOT / "serve/server.py") in cmd:
        return _None()
    if RECORD and cmd[0] == RUST:
        return _None()
    env = {**(env or os.environ), "STRATA_FIREWALL_LOG": ""}   # test traffic stays out of the user's firewall history
    p = subprocess.Popen(cmd + ["--port", str(port)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
    for _ in range(200):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1).read()
            return p
        except Exception:
            time.sleep(0.1)
    raise SystemExit(f"{cmd[0]} did not start: {p.stdout.read(4000)}")

def call(port, method, path, body=None, headers=None):
    data = None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
    h = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.headers.get("Content-Type", ""), r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type", ""), e.read().decode("utf-8", "replace")

IDS = re.compile(r'^(chatcmpl-|msg_|resp_|rs_|fc_|ctc_|call_|video_)[0-9a-f]{6,}$')
CLOCKS = {"ms", "duration_s", "load_s", "queue_s", "wallclock_s", "first_token_s", "path", "file", "log", "chat_template", "model_path", "created", "created_at", "completed_at", "time", "since", "started", "uptime_s", "at", "started_at", "finished_at", "last_request_at", "elapsed_s", "web", "encrypted_content", "machine"}  # machine: live hardware readings

def norm(v):
    """ids and clocks out of a parsed answer; the order of keys is kept (it is compared too)."""
    if isinstance(v, dict):
        return [[k, "X" if k in CLOCKS else norm(x)] for k, x in v.items()]
    if isinstance(v, list):
        return [norm(x) for x in v]
    if isinstance(v, str) and IDS.match(v):
        return IDS.match(v).group(1) + "X"
    return v

def parse(ctype, text):
    if "event-stream" in ctype:
        out = []
        for block in text.split("\n\n"):
            if not block or block.startswith(": keep-alive"):
                continue
            lines = block.split("\n")
            ev = [l[7:] for l in lines if l.startswith("event: ")]
            data = "\n".join(l[6:] for l in lines if l.startswith("data: "))
            try:
                data = norm(json.loads(data, object_pairs_hook=dict))
            except ValueError:
                pass
            out.append([ev, data])
        return out
    try:
        return norm(json.loads(text))
    except ValueError:
        return text

CASES = [
    ("GET", "/v1/models", None), ("GET", "/props", None), ("GET", "/slots", None), ("GET", "/settings", None), ("GET", "/mcp", None),
    ("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}),
    ("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "read a.txt"}], "tools": T2}),
    ("POST", "/v1/chat/completions", {"stream": True, "messages": [{"role": "user", "content": "hi é"}], "stop": ["STOPHERE"]}),
    ("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "json"}], "response_format": {"type": "json_object"}}),
    ("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 7}),
    ("POST", "/v1/chat/completions", {"stream": True, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 9, "reasoning_effort": "low"}),
    ("POST", "/v1/chat/completions", {"messages": "nope"}),
    ("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}], "tools": "auto"}),
    ("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}], "temperature": "hot"}),
    ("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}], "reasoning_effort": "turbo"}),
    ("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}], "max_tokens": 10**9}),
    ("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}], "logit_bias": {"5": 3}}),
    ("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}], "stop": ["a", "b", "c", "d", "e"]}),
    ("POST", "/v1/messages", {"max_tokens": 200, "messages": [{"role": "user", "content": "hi"}]}),
    ("POST", "/v1/messages", {"max_tokens": 200, "stream": True, "messages": [{"role": "user", "content": "read a.txt"}], "tools": [{"name": "read_file", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}]}),
    ("POST", "/v1/messages", {"max_tokens": 200, "stop_sequences": ["STOPHERE"], "messages": [{"role": "user", "content": "hi"}]}),
    ("POST", "/v1/messages/count_tokens", {"messages": [{"role": "user", "content": "count me é"}]}),
    ("POST", "/v1/responses", {"input": "hi", "max_output_tokens": 300}),
    ("POST", "/v1/responses", {"input": [{"role": "user", "content": "hi"}], "stream": True, "include": ["reasoning.encrypted_content"]}),
    ("POST", "/v1/responses", {"input": "x", "previous_response_id": "r1"}),
    ("POST", "/context", {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "draft"}], "strata_draft": True}),
    ("POST", "/v1/chat/completions", b"{bad json", "status-only"),
    ("POST", "/nope", {}), ("GET", "/nope", None), ("DELETE", "/v1/videos/x", None),
    ("POST", "/settings", {"defaults": {"temperature": 0.5, "max_tokens": 50}}), ("GET", "/settings", None),
    ("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "uses shared"}]}),
    ("POST", "/settings", {"defaults": None}),
    ("GET", "/status", None), ("GET", "/health", None),
    ("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}], "tools": T2, "tool_choice": "required"}),
    ("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}], "tools": T2, "tool_choice": {"type": "function", "function": {"name": "read_file"}}, "reasoning_effort": "none"}),
    ("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}], "tools": T2, "tool_choice": "none", "stream": True}),
    ("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}], "reasoning_budget_tokens": 3}),
    ("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}], "reasoning_budget_tokens": "lots"}),
    ("POST", "/v1/chat/completions", {"messages": [{"role": "system", "content": "a long shared document " * 40}, {"role": "user", "content": "q"}], "strata_prefix": {"messages": 1}}),
    ("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}], "strata_prefix": {"bogus": 1}}),
    ("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "json"}], "response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}}}}),
    ("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}], "response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "array"}}}}),
    ("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}], "top_k": "5", "seed": "7", "temperature": "0.5"}),
    ("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}], "top_k": 5.5}),
    ("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "go"}, {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\": \"a\"}"}}]}, {"role": "tool", "tool_call_id": "c1", "content": "body <think> and </think>"}], "tools": T2}),
    ("POST", "/v1/messages", {"max_tokens": 50, "messages": [{"role": "user", "content": "q"}], "tools": [{"name": "read_file", "input_schema": {"type": "object"}}], "tool_choice": {"type": "any"}}),
    ("POST", "/v1/messages", {"max_tokens": 50, "thinking": {"type": "disabled"}, "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "read_file", "input": {"path": "a"}}]}, {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "ok"}]}], "tools": [{"name": "read_file", "input_schema": {"type": "object"}}]}),
    ("POST", "/v1/messages", {"max_tokens": 50, "messages": [{"role": "user", "content": "q"}], "stream": True, "output_config": {"effort": "low"}}),
    ("POST", "/v1/responses", {"input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "read"}]}, {"type": "function_call", "call_id": "c1", "name": "read_file", "arguments": "{\"path\": \"a\"}"}, {"type": "function_call_output", "call_id": "c1", "output": "body"}], "tools": [{"type": "function", "name": "read_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}, {"type": "web_search"}], "reasoning": {"effort": "low"}}),
    ("POST", "/v1/responses", {"input": "json please", "text": {"format": {"type": "json_schema", "name": "x", "schema": {"type": "object"}}}}),
    ("POST", "/v1/responses", {"input": [{"type": "item_reference", "id": "x"}]}),
    ("POST", "/v1/responses", {"input": "x", "tools": [{"type": "namespace", "name": "mcp__srv", "tools": [{"name": "t1", "parameters": {"type": "object"}}]}], "stream": True}),
    ("POST", "/v1/messages/count_tokens", {"messages": [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}]}]}, "status-only"),
    ("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": [{"type": "text", "text": "see"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]}, "status-only"),
]

def run(port):
    out = []
    for case in CASES:
        method, path, body = case[:3]
        status, ctype, text = call(port, method, path, body)
        out.append({"req": [method, path], "status": status, "body": None if case[3:] == ("status-only",) else parse(ctype, text)})
    return out

py_port, rs_port = 18601, 18602
tok_args = ["--tokenizer", TOK] if TOK != "-" else []
scripts = sum([["--script", s] for s in SCRIPTS], [])
if ENGINE:
    import tempfile
    work = pathlib.Path(tempfile.mkdtemp(prefix="strata-e2e-"))
    replies = work / "replies.json"
    replies.write_text(json.dumps(SCRIPTS))
    def cfg(name):
        d = work / name
        d.mkdir()
        c = {"exe": str(pathlib.Path(__file__).with_name("fake_engine.py")), "args": ["--tokenizer", TOK, "--max-context", "8192", "--replies", str(replies)],
             "tokenizer": TOK, "model_name": "fake-model", "log": str(d / "engine.log"), "cwd": str(d), "backend": "metal",
             "slot_save_path": str(d / "slots")}
        (d / "strata-fake.json").write_text(json.dumps(c))
        return ["--engine", "strata", "--config", str(d / "strata-fake.json")]
    py = start([str(ROOT / ".venv/bin/python"), str(ROOT / "serve/server.py"), *cfg("py")], py_port)
    rs = start([RUST, *cfg("rs")], rs_port)
    CASES += [
        ("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "first turn"}]}),
        ("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "first turn"}, {"role": "assistant", "content": "ok"}, {"role": "user", "content": "second turn"}]}),
        ("POST", "/slots/0?action=save", {"filename": "conv1"}), ("POST", "/slots/0?action=restore", {"filename": "conv1"}),
        ("POST", "/slots/0?action=restore", {"filename": "missing"}), ("POST", "/slots/0?action=save", {"filename": "../x"}),
        ("POST", "/v1/vram", {"reserve_mib": 1024}),
        ("POST", "/unload", {}), ("GET", "/v1/models", None), ("POST", "/load", {}),
        ("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "after reload"}], "temperature": 0.7, "top_k": 100, "seed": 3}),
        ("GET", "/v1/status", None), ("GET", "/props", None), ("GET", "/slots", None),
    ]
elif MCP:
    import tempfile
    work = pathlib.Path(tempfile.mkdtemp(prefix="strata-mcp-"))
    fake = str(pathlib.Path(__file__).with_name("mcp_fake_server.py"))
    http = subprocess.Popen([str(ROOT / ".venv/bin/python"), fake, "--http", "18765"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1)
    call_echo = "I will use the tool.</think>\n\n<tool_call>\n<function=fake__echo>\n<parameter=text>\nhello mcp\n</parameter>\n</function>\n</tool_call>"
    call_add = "Adding.</think>\n\n<tool_call>\n<function=web__add>\n<parameter=a>\n2\n</parameter>\n<parameter=b>\n40\n</parameter>\n</function>\n</tool_call>"
    call_fail = "Trying.</think>\n\n<tool_call>\n<function=fake__fail>\n</function>\n</tool_call>"
    final = "Done.</think>\n\nThe tool said what it said."
    mscripts = sum([["--script", x] for x in [call_echo, final, call_echo, final, call_add, final, call_fail, final]], [])
    def mcfg(name):
        c = {"mcp_servers": {"fake": {"command": str(ROOT / ".venv/bin/python"), "args": [fake]}, "web": {"url": "http://127.0.0.1:18765/mcp"}}}
        (work / f"{name}.json").write_text(json.dumps(c))
        return ["--config", str(work / f"{name}.json")]
    py = start([str(ROOT / ".venv/bin/python"), str(ROOT / "serve/server.py"), "--engine", "mock", *tok_args, *mscripts, *mcfg("py")], py_port)
    rs = start([RUST, "--engine", "mock", *tok_args, *mscripts, *mcfg("rs")], rs_port)
    time.sleep(3)
    m = {"messages": [{"role": "user", "content": "use the tool"}], "strata_mcp": True}
    CASES[:] = [("GET", "/mcp", None), ("POST", "/v1/chat/completions", m), ("POST", "/v1/chat/completions", {**m, "stream": True}),
                ("POST", "/v1/chat/completions", m), ("POST", "/v1/chat/completions", m),
                ("POST", "/context", {**m, "strata_draft": True})]
else:
    py = start([str(ROOT / ".venv/bin/python"), str(ROOT / "serve/server.py"), "--engine", "mock", *tok_args, *scripts], py_port)
    rs = start([RUST, "--engine", "mock", *tok_args, *scripts], rs_port)
import gzip
try:
    if RECORD:
        a = run(py_port)
        (GOLDEN / f"e2e-{RECORD}.json.gz").write_bytes(gzip.compress(json.dumps({"answers": a}).encode(), mtime=0))
        print(f"recorded {len(a)} answers into e2e-{RECORD}.json.gz")
        b = a
    elif REPLAY:
        a = json.loads(gzip.decompress((GOLDEN / f"e2e-{REPLAY}.json.gz").read_bytes()))["answers"]
        b = json.loads(json.dumps(run(rs_port)))
    else:
        a, b = run(py_port), run(rs_port)
finally:
    py.terminate(); rs.terminate()
    if MCP:
        http.terminate()
# the fake engine's decode time counts the tokens it wrote before a STOP reached it: a race of the fake, not the server
RACY = {"predicted_ms", "decode_ms", "engine_generated", "decode_tok_s", "predicted_per_second", "predicted_per_token_ms", "generated"}

def calm(v):
    if isinstance(v, list) and all(isinstance(p, list) and len(p) == 2 and isinstance(p[0], str) for p in v) and v:
        return [[k, "X" if k in RACY else calm(x)] for k, x in v]
    if isinstance(v, list):
        return [calm(x) for x in v]
    return v

a, b = calm(a), calm(b)
bad = 0
for x, y in zip(a, b):
    if x != y:
        bad += 1
        print("=== DIFF", x["req"], "status", x["status"], y["status"])
        xs, ys = json.dumps(x["body"], ensure_ascii=False), json.dumps(y["body"], ensure_ascii=False)
        i = next((k for k in range(min(len(xs), len(ys))) if xs[k] != ys[k]), min(len(xs), len(ys)))
        print("  py:", xs[max(0, i - 150):i + 250])
        print("  rs:", ys[max(0, i - 150):i + 250])
if "--show" in sys.argv:
    for x in b:
        print(x["req"], x["status"], json.dumps(x["body"], ensure_ascii=False)[:700])
print(f"{len(a) - bad} of {len(a)} identical")
sys.exit(1 if bad else 0)
