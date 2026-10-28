#!/usr/bin/env python3
"""A deterministic stand-in for strata-metal's line protocol (golden/e2e_engine.py runs both servers on it):
INFO/READY at start; GEN/GENI -> RESUME, PP lines, T lines, DONE (STOP honoured between tokens); BGEN -> T, BADM, then
BT ... BDONE (BSTOP honoured); SAVE/RESTORE -> SAVED/RESTORED; VRAM -> ERR; QUIT or EOF ends it.

    fake_engine.py --serve --tokenizer DIR [--max-context N] [--batch N] [--delay S] [--replies FILE]
"""
import json, os, pathlib, queue, sys, threading, time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
try:
    import strata_tokenizer as ST
except ImportError:                              # the tokenizer needs `regex`: Strata's own .venv has it
    venv = ROOT / ".venv" / "bin" / "python"
    if not venv.exists() or pathlib.Path(sys.executable).resolve() == venv.resolve():
        raise
    os.execv(str(venv), [str(venv), __file__, *sys.argv[1:]])

args = sys.argv[1:]
opt = lambda k, d=None: args[args.index(k) + 1] if k in args else d
tdir = pathlib.Path(opt("--tokenizer"))
vocab = json.loads((tdir / "vocab.json").read_text(encoding="utf-8"))
tokens = [None] * len(vocab)
for t, i in vocab.items():
    tokens[i] = t
tok = ST.Tokenizer(tokens, (tdir / "merges.txt").read_text(encoding="utf-8").split("\n"),
                   json.loads((tdir / "token_type.json").read_text()), json.loads((tdir / "tokenizer.json").read_text()).get("pre", "qwen35"))
harmony = "<|channel|>" in tok.special_tokens
end = tok.encode("<|return|>" if harmony else "<|im_end|>", parse_special=True)
REPLIES = json.loads(pathlib.Path(opt("--replies")).read_text()) if opt("--replies") else [
    ("<|channel|>analysis<|message|>Let me think about it carefully.<|end|><|start|>assistant<|channel|>final<|message|>The answer is 42, as always.<|return|>"
     if harmony else "Let me think about it carefully.</think>\n\nThe answer is 42, as always.")]
ctx = int(opt("--max-context", "8192"))
batch = int(opt("--batch", "0"))
delay = float(opt("--delay", "0.002"))
pp_delay = float(opt("--pp-delay", "0"))       # seconds per PP chunk: a slow prompt read a STOP can cut short
out_lock = threading.Lock()

def out(s):
    with out_lock:
        sys.stdout.write(s + "\n")
        sys.stdout.flush()

stop = threading.Event()
bstop = set()
lines = queue.Queue()

def reader():
    for raw in sys.stdin:
        line = raw.rstrip("\r\n")
        if line == "STOP":
            stop.set()
            continue
        if line.startswith("BSTOP "):
            bstop.add(int(line.split()[1]))
            continue
        if line.startswith("BYIELD"):
            continue
        if line.startswith(("GEN", "BGEN")):
            stop.clear()
        lines.put(line)
    lines.put("QUIT")

threading.Thread(target=reader, daemon=True).start()
count = 0

def reply_ids():
    global count
    text = REPLIES[min(count, len(REPLIES) - 1)]
    count += 1
    return tok.encode(text, parse_special=True)

def parse_gen(line, slot=False):
    f = line.split()
    rest = f[2:] if slot else f[1:]
    max_new = int(rest[0])
    ids = [int(x) for x in rest[-1].split(",") if x]
    keys = " ".join(x for x in rest[1:-1] if "=" in x)
    return max_new, ids, keys

held = []
out(f"INFO engine=0.9.9-fake backend=metal context={ctx} batch_slots={batch} kv=f16")
out(f"READY {ctx} stop")
slots = {}
while True:
    line = lines.get()
    if line == "QUIT":
        break
    if line.startswith("GEN"):
        max_new, ids, keys = parse_gen(line)
        reused = 0
        for a, b in zip(held, ids):
            if a != b:
                break
            reused += 1
        out(f"RESUME {reused}")
        step = 512
        read = len(ids) - reused
        cut = False
        for at in range(reused + step, len(ids) - 64, step):
            time.sleep(pp_delay)
            if stop.is_set():
                read, cut = at - reused, True
                break
            out(f"PP {at} {len(ids)} {at * 0.5:.0f} {1000.0:.1f}")
        produced, finish = 0, "length"
        for t in ([] if cut else reply_ids()):
            if produced >= max_new:
                break
            if stop.is_set():
                finish = "cancel"
                break
            time.sleep(delay)
            out(f"T {t}")
            produced += 1
            if t in end:
                finish = "stop"
                break
        held = ids + []
        if cut:
            finish = "cancel"
        out(f"DONE {produced} {len(ids)} {len(ids) * 0.5:.1f} {produced * 30.0:.1f} {finish} 0 0 {reused} 10 20 0 0 0.0 {read} 0")
    elif line.startswith("BGEN"):
        slot = int(line.split()[1])
        max_new, ids, keys = parse_gen(line, slot=True)
        rid = reply_ids()
        out(f"T {rid[0]}")
        out(f"DONE 1 {len(ids)} {len(ids) * 0.5:.1f} 1.0 length 0 0 0 0 0 0 0 0.0 {len(ids)} 0")
        out(f"BADM {slot} 1")
        bstop.discard(slot)

        def run(slot=slot, rid=rid, max_new=max_new):
            produced, finish = 1, "length"
            for t in rid[1:]:
                if produced >= max_new:
                    break
                if slot in bstop:
                    finish = "cancel"
                    break
                time.sleep(delay)
                out(f"BT {slot} {t}")
                produced += 1
                if t in end:
                    finish = "stop"
                    break
            out(f"BDONE {slot} {produced} {finish} {produced * 30.0:.1f}")
        threading.Thread(target=run, daemon=True).start()
    elif line.startswith("SAVE "):
        out(f"SAVED {len(held)} {len(held) * 100} 1.5")
    elif line.startswith("RESTORE "):
        out(f"RESTORED {len(held)} {len(held) * 100} 1.2")
    elif line.startswith("VRAM"):
        out("ERR the Metal engine has no expert cache to resize")
    elif line:
        out(f"ERR unknown command: {line[:40]}")
