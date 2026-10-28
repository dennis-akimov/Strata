"""Golden events from serve/frontend.py's parsers: tests/golden/parser.json (inputs as the exact chunks fed)."""
import json, pathlib, random, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from serve.frontend import OutputParser, HarmonyParser, GlmParser

TOOLS = [{"name": "read", "description": "r", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "n": {"type": "integer"}}}},
         {"name": "write", "description": "w", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "body": {"type": "string"}, "opts": {"type": "object"}}}}]
CALL = "<tool_call>\n<function=read>\n<parameter=path>\n/tmp/a b\n</parameter>\n<parameter=n>\n5\n</parameter>\n</function>\n</tool_call>"
CALL2 = "<tool_call>\n<function=write>\n<parameter=path>\nx.md\n</parameter>\n<parameter=body>\nline1\n</parameter> not end\n\"q\" é 😀\n</parameter>\n<parameter=opts>\n{\"a\": [1, 2.5]}\n</parameter>\n</function>\n</tool_call>"
QWEN = [
    "Let me think.</think>\n\nThe answer is 4.",
    "Thinking `</think>` quoted and \"</think>\" too</think>\n\nDone",
    "plan</think>\n\n" + CALL,
    "plan</think>\n\nI will read it.\n\n" + CALL + "\n\n" + CALL2,
    "reasoning\n" + CALL + "\n</think>\n\nafter",
    "reasoning text " + CALL + " more prose</think>answer",
    "```\n" + CALL + "\n```\n</think>\n\nok",
    "</think>\n\nUse `<tool_call>` like this. And ```\n<tool_call>\n```",
    "</think>\n\nI'll use a <tool_call> block here.",
    "</think>\n\n<function=read>\n<parameter=path>\nz\n</parameter>\n</function>\nthen",
    "</think>\n\n<tool_call>\n{\"name\": \"read\", \"arguments\": {\"path\": \"q\"}}\n</tool_call>\nand",
    "</think>\n\n<tool_call>\n{\"name\": \"nope\", \"arguments\": {}}\n</tool_call>",
    "</think>\n\n<tool_call>\n<parameter=read>\n<parameter=path>\nv\n</parameter>\n</function>\n</tool_call>",
    "</think>\n\n<tool_call>\n<function=read>\n<parameter=path>\na\n</parameter>\n</function>\n<function=write>\n<parameter=path>\nb\n</parameter>\n</function>\n</tool_call>",
    "</think>\n\n<tool_call>\n<function=read>\n<parameter=path>\ncut off here",
    "</think>\n\n<tool_call>\n<function=read>\n<parameter=n>\n12",
    "</think>\n\n<tool_call>\n<function=read>\n<parameter=path>\nq\n</parameter>\n</function>",
    "</think>\n\n<tool_call>garbage</tool_call>",
    "no think end, just thinking forever <tool",
    "ends on tag</think>",
    "x" * 300 + "</think>" + "y" * 50,
    "a\n~~~\n" + CALL + "\n~~~\nb " + CALL + "</think>",
]
H = [
    "<|channel|>analysis<|message|>Let me think<|end|><|start|>assistant<|channel|>final<|message|>Hi there<|return|>",
    "<|channel|>analysis<|message|>need tool<|end|><|start|>assistant<|channel|>commentary to=functions.read <|constrain|>json<|message|>{\"path\": \"a\"}<|call|>",
    "<|channel|>commentary<|message|>preamble<|end|><|start|>assistant to=functions.write<|channel|>commentary json<|message|>{bad json<|call|>",
    "<|channel|>final<|message|>cut <|en",
    "<|channel|>analysis<|message|>never ends",
    "<|channel|>commentary to=functions.read<|message|>{\"n\": 1",
    "<|start|>assistant<|channel|>final<|message|>multi é 😀 <|end|>",
]
G = [
    "think</think>\n\n<tool_call>read<arg_key>path</arg_key><arg_value>/a</arg_value><arg_key>n</arg_key>\n<arg_value>3</arg_value></tool_call>",
    "think</think>\n\nText <tool_call>write<arg_key>body</arg_key><arg_value>multi\nline</arg_value></tool_call> after",
    "think</think>\n\n<tool_call>read<arg_key>path</arg_key><arg_value>cut",
    "</think>\n\n<tool_call>bad name with < </tool_call>",
]
FRAG = ["</think>", "<tool_call>", "</tool_call>", "<function=read>", "<function=write>", "<parameter=path>", "<parameter=n>",
        "</parameter>", "</function>", "\n", "\n\n", " ", "`", "```", "~~~", "\"", "'", "text", "é😀", "{\"name\": \"read\"}", "<", "</", "12",
        "<parameter=read>", "<function=", "<tool_", "</thi"]

def run(parser, chunks, reason):
    ev = []
    try:
        for c in chunks:
            ev += parser.feed(c)
        ev += parser.finish(reason)
    except Exception as e:
        return [["error", type(e).__name__, str(e)]]
    ids = {}
    out = []
    for e in ev:
        cid = ids.setdefault(e.call.id, len(ids)) if e.call else None
        out.append([e.kind, e.text, e.call.name if e.call else None, e.call.arguments if e.call else None, cid])
    return out

def splits(text, rnd):
    yield [text]
    if len(text) < 400:                 # one character at a time: the short texts (keeps the file small)
        yield list(text)
    for _ in range(2):
        cs, i = [], 0
        while i < len(text):
            n = rnd.randrange(1, 12)
            cs.append(text[i:i + n]); i += n
        yield cs

rnd = random.Random(11)
rows = []
fuzz = ["".join(rnd.choice(FRAG) for _ in range(rnd.randrange(3, 40))) for _ in range(120)]
for text in QWEN + fuzz:
    for thinking in (True, False):
        for tools in (TOOLS, None):
            for stream in (False, True):
                for recover in (False, True):
                    if tools is None and (stream or recover):
                        continue
                    for chunks in splits(text, rnd):
                        for reason in (None, "length"):
                            p = OutputParser(thinking=thinking, tools=tools, stream_tools=stream, recover=recover)
                            rows.append({"p": "qwen", "thinking": thinking, "tools": tools is not None, "stream": stream,
                                         "recover": recover, "chunks": chunks, "reason": reason, "events": run(p, chunks, reason)})
for text in H + ["".join(rnd.choice(["<|channel|>", "analysis", "final", "<|message|>", "<|end|>", "<|start|>", "assistant",
                                     " to=functions.read", "<|call|>", "<|return|>", "{\"a\": 1}", "x", " ", "<|", "<|en"])
                     for _ in range(rnd.randrange(3, 30))) for _ in range(150)]:
    for chunks in splits(text, rnd):
        for reason in (None, "length"):
            rows.append({"p": "harmony", "chunks": chunks, "reason": reason, "events": run(HarmonyParser(), chunks, reason)})
for text in G + ["".join(rnd.choice(["<tool_call>", "</tool_call>", "read", "<arg_key>", "path", "</arg_key>", "<arg_value>",
                                     "v", "</arg_value>", "</think>", "\n", " x "]) for _ in range(rnd.randrange(3, 30)))
                 for _ in range(150)]:
    for thinking in (True, False):
        for recover in (False, True):
            for chunks in splits(text, rnd):
                for reason in (None, "length"):
                    rows.append({"p": "glm", "thinking": thinking, "recover": recover, "chunks": chunks, "reason": reason,
                                 "events": run(GlmParser(thinking=thinking, tools=TOOLS, recover=recover), chunks, reason)})
import gzip
out = pathlib.Path(__file__).resolve().parents[1] / "tests/golden/parser.json.gz"
out.write_bytes(gzip.compress(json.dumps({"tools": TOOLS, "rows": rows}).encode(), mtime=0))
print(len(rows), "rows", out.stat().st_size // 1024, "KB")
