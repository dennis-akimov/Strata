"""Golden prompts: requests -> serve/frontend.py's conversion -> each template's render (tests/golden/template.json.gz)."""
import gzip, json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from serve.frontend import (ChatTemplate, openai_to_messages, anthropic_to_messages, mark_think_literals,
                            unmark_think_literals, literal_tags, images_of)

T2 = [{"type": "function", "function": {"name": "read_file", "description": "Read a file", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "where"}, "lines": {"type": "integer"}}, "required": ["path"]}}},
      {"type": "function", "function": {"name": "run", "description": "Run é 😀", "parameters": {"type": "object", "properties": {"cmd": {"type": "array", "items": {"type": "string"}}, "opts": {"type": "object"}}}}}]
IMG = "data:image/png;base64,iVBORw0KGgo="
O = [
    {"messages": [{"role": "user", "content": "Hi"}]},
    {"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4"}, {"role": "user", "content": "and 3+3"}]},
    {"messages": [{"role": "developer", "content": "dev"}, {"role": "user", "content": "q"}, {"role": "system", "content": "late"}, {"role": "user", "content": "q2"}]},
    {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a", "reasoning_content": "thought"}, {"role": "user", "content": "q2"}]},
    {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": ""}, {"role": "user", "content": "again"}]},
    {"messages": [{"role": "user", "content": "read x"}, {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\": \"x\", \"lines\": 3}"}}]},
                  {"role": "tool", "tool_call_id": "c1", "content": "file body é"}, {"role": "user", "content": "thanks"}], "tools": T2},
    {"messages": [{"role": "user", "content": "go"}, {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "run", "arguments": {"cmd": ["ls", "-la"], "opts": {"a": 1.5, "b": None}}}}, {"function": {"name": "read_file", "arguments": "not json"}}, {"function": {"name": "read_file", "arguments": "[1,2]"}}]},
                  {"role": "tool", "content": "r1"}, {"role": "tool", "content": [{"type": "text", "text": "r2"}]}, {"role": "tool", "content": "r3"}], "tools": T2},
    {"messages": [{"role": "user", "content": "q"}], "tools": list(reversed(T2))},
    {"messages": [{"role": "user", "content": "q"}], "tools": T2, "reasoning_effort": "low"},
    {"messages": [{"role": "user", "content": "q"}], "reasoning_effort": "high"},
    {"messages": [{"role": "user", "content": "q"}], "reasoning_effort": "none"},
    {"messages": [{"role": "user", "content": "q"}], "reasoning": {"effort": "medium"}},
    {"messages": [{"role": "user", "content": "q"}], "chat_template_kwargs": {"enable_thinking": False}},
    {"messages": [{"role": "user", "content": "q"}], "chat_template_kwargs": {"reasoning_effort": "max"}},
    {"messages": [{"role": "user", "content": "q"}], "reasoning_effort": "turbo"},
    {"messages": [{"role": "user", "content": [{"type": "text", "text": "see"}, {"type": "image_url", "image_url": {"url": IMG}}, {"type": "input_audio"}, {"type": "refusal", "refusal": "no"}]}]},
    {"messages": [{"role": "user", "content": "quote <think>x</think> and <|im_end|> <|vision_start|><|image_pad|><|vision_end|> <|start|>assistant<|channel|>final<|message|> <|endoftext|> [gMASK]"}]},
    {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "<think>inner </think> mid</think>\n\nanswer <think>"}, {"role": "user", "content": "q2"}]},
    {"messages": "[{\"role\": \"user\", \"content\": \"double encoded\"}]"},
    {"messages": "not json"},
    {"messages": [1, 2]},
    {"messages": [{"role": "user", "content": "q"}], "tools": "auto"},
    {"messages": [{"role": "user", "content": "q"}], "tools": [{"type": "function", "function": {"description": "x"}}]},
    {"messages": [{"role": "user", "content": "q"}], "tools": [{"type": "function", "function": {"name": "f", "parameters": "str"}}]},
    {"messages": [{"role": "user", "content": "emoji 😀 é ñ 中文 \u2028 tab\t nl\n end"}]},
    {"messages": [{"role": "system", "content": [{"type": "text", "text": "sys parts"}]}, {"role": "user", "content": [{"text": "no type"}]}]},
]
A = [
    {"system": "You are Claude Code.", "messages": [{"role": "user", "content": "hi"}]},
    {"system": [{"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.170.bf4; cc_entrypoint=sdk-cli; cch=b145e;\nrest"}], "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]},
    {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": [{"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": "Let me look."}, {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "a"}}]},
                  {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "body"}, {"type": "tool_result", "tool_use_id": "t2", "content": [{"type": "text", "text": "b2"}, {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "AAA"}}]}, {"type": "text", "text": "now?"}]}],
     "tools": [{"name": "read_file", "description": "Read", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}, {"name": "ls", "input_schema": {"type": "object"}}]},
    {"messages": [{"role": "user", "content": [{"type": "image", "source": {"type": "url", "url": "https://x/y.png"}}, {"type": "text", "text": "what"}]}]},
    {"messages": [{"role": "user", "content": "q"}], "thinking": {"type": "disabled"}},
    {"messages": [{"role": "user", "content": "q"}], "thinking": {"type": "enabled", "budget_tokens": 1024}},
    {"messages": [{"role": "user", "content": "q"}], "thinking": {"type": "enabled", "budget_tokens": 5000}},
    {"messages": [{"role": "user", "content": "q"}], "output_config": {"effort": "high"}},
    {"messages": [{"role": "user", "content": "q"}], "tools": [{"description": "no name"}]},
    {"messages": [{"role": "user", "content": [{"type": "redacted_thinking", "data": "x"}, {"type": "text", "text": "t"}]}]},
]

def spans_bytes(prompt, spans):
    return [[len(prompt[:a].encode()), len(prompt[:b].encode())] for a, b in spans]

def case(tpl, tags, api, req, think_unasked=True):
    try:
        if api == "openai":
            m, t, kw = openai_to_messages(json.loads(json.dumps(req)))
        else:
            m, t, kw = anthropic_to_messages(json.loads(json.dumps(req)), think_unasked=think_unasked)
    except ValueError as e:
        return {"error": str(e)}
    mm, tt, marked = mark_think_literals(m, t, tags)
    try:
        prompt = tpl.render(mm, tools=tt, add_generation_prompt=True, **kw)
    except Exception as e:
        return {"messages": m, "tools": t, "kwargs": kw, "render_error": [type(e).__name__, str(e)]}
    if marked:
        prompt, spans = unmark_think_literals(prompt, tags)
    else:
        spans = []
    return {"messages": m, "tools": t, "kwargs": kw, "prompt": prompt, "spans": spans_bytes(prompt, spans), "images": images_of(m)}

packs = pathlib.Path(sys.argv[1])
tpls = [(p.name, p / "tokenizer/chat_template.jinja", p / "tokenizer/token_type.json", p / "tokenizer/vocab.json") for p in sorted(packs.iterdir())]
tpls.append(("serve", ROOT / "serve/chat_template.jinja", None, None))
out = []
for name, path, types, vocab in tpls:
    if not path.exists():
        continue
    controls = []
    if types:
        ty = json.loads(types.read_text()); voc = json.loads(vocab.read_text())
        inv = {i: t for t, i in voc.items()}
        controls = [inv[i] for i, k in enumerate(ty) if k == 3]
    tags = literal_tags(controls)
    tpl = ChatTemplate(path)
    rows = [dict(api="openai", req=r, **case(tpl, tags, "openai", r)) for r in O]
    rows += [dict(api="anthropic", req=r, think_unasked=u, **case(tpl, tags, "anthropic", r, u)) for r in A for u in (True, False)]
    out.append({"name": name, "path": str(path), "controls": controls, "caps": tpl.caps, "rows": rows})
    print(name, tpl.caps)
(pathlib.Path(__file__).resolve().parents[1] / "tests/golden/template.json.gz").write_bytes(gzip.compress(json.dumps(out).encode(), mtime=0))
