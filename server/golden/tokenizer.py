"""Golden ids from the reference tokenizer (tools/strata_tokenizer.py) for every pack: tests/golden/tokenizer-<pack>.json."""
import json, pathlib, random, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import strata_tokenizer as ST

def load(tpath):
    vocab = json.loads((tpath / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    merges = (tpath / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((tpath / "token_type.json").read_text())
    meta = json.loads((tpath / "tokenizer.json").read_text())
    return ST.Tokenizer(tokens, merges, types, meta.get("pre", "qwen35"))

def corpus(tok):
    texts = ["", "Hello, world!", "  leading and trailing  ", "a\n\n\nb", "def f(x):\n\treturn x  # comment\n",
             "你好，世界", "مرحبا", "\U0001f600\U0001f680\U0001f1fa\U0001f1f8",
             "é́ combining", "\x00\x01\x7f control \x1c\x1d\x1e\x1f\x85", " non-breaking space",
             "1234567890 12.5e-3", "MixedCASE_and-dashes I'M HE'S they'll WE'VE ſs", "\r\n\r\n", "\u2028\u2029",
             "x" * 5000, "中" * 3000, "   \n   \n\t\t  end", "a/b/c\n/d", "<think>x</think> <|im_end|> <|endoftext|>",
             "<|start|>assistant<|channel|>final<|message|>hi<|end|>", "<tool_call>\n<function=f>\n</tool_call>"]
    texts += list(tok.special_tokens)[:300]
    rnd = random.Random(7)
    for f in sorted(ROOT.glob("README*.md")) + [ROOT / "serve/frontend.py", ROOT / "server/web/app.js", ROOT / "docs/DETAILS.md"]:
        if not f.exists():
            continue
        t = f.read_text(encoding="utf-8")
        for _ in range(25):
            a = rnd.randrange(0, max(1, len(t) - 1))
            texts.append(t[a:a + rnd.choice([10, 100, 1000, 3000])])
    # random characters across the planes, specials glued in
    pool = [chr(c) for c in list(range(0x20, 0x250)) + list(range(0x370, 0x3ff)) + list(range(0x3040, 0x30ff)) +
            list(range(0x4e00, 0x4e80)) + list(range(0x1f300, 0x1f340)) + [0x300, 0x301, 0x302, 0x200d, 0xfe0f, 0xa, 0xd, 0x9]]
    for _ in range(300):
        texts.append("".join(rnd.choice(pool + [" ", "  ", "\n", "'s", "123"]) for _ in range(rnd.randrange(1, 80))))
    return texts

out_dir = pathlib.Path(__file__).resolve().parents[1] / "tests/golden"
for pack in sorted(pathlib.Path(sys.argv[1]).iterdir()):
    tpath = pack / "tokenizer"
    if not (tpath / "vocab.json").exists():
        continue
    tok = load(tpath)
    rows = [{"text": t, "special": sp, "ids": tok.encode(t, parse_special=sp)} for t in corpus(tok) for sp in (False, True)]
    import gzip
    (out_dir / f"tokenizer-{pack.name}.json.gz").write_bytes(gzip.compress(json.dumps({"dir": str(tpath), "rows": rows}).encode(), mtime=0))   # ASCII: no raw U+2028/U+2029 (editors treat them as line breaks)
    print(pack.name, tok.pre, len(rows))
