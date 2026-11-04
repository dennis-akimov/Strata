"""strata-metal against the engine line protocol, on a real model (macOS).

    STRATA_METAL_EXE=build-metal/metal/strata-metal STRATA_METAL_GGUF=<a .gguf> python -m unittest metal.test_strata_metal

Any GGUF with <|im_start|> works (LiquidAI/LFM2-350M-GGUF Q8_0 is a 360 MB hybrid model whose recurrent layers need
the checkpoints, as Qwen3.8-Flash-Next's do).  Skipped without the two variables.  Greedy throughout, so the answers
of two engines given the same ids MUST be the same tokens.
"""
from __future__ import annotations

import os
import struct
import subprocess
import tempfile
import unittest

EXE, GGUF = os.environ.get("STRATA_METAL_EXE"), os.environ.get("STRATA_METAL_GGUF")
PAD = 5000                       # the image cells' token id here (an ordinary id of any test model's vocabulary)


def embeddings(path, images, n_embd):
    """strata-vision's file: per image int32 {'SVE1', n, nx, ny, n_embd} then n x n_embd float32 (a fill value)."""
    with open(path, "wb") as f:
        for nx, ny, fill in images:
            f.write(struct.pack("<5i", 0x31455653, nx * ny, nx, ny, n_embd))
            f.write(struct.pack(f"<{nx * ny * n_embd}f", *([fill] * (nx * ny * n_embd))))


class Engine:
    def __init__(self, *extra):
        # STRATA_METAL_EXTRA: more engine options for the whole suite (--kv-unified: docs/CONTEXT_POOL.md)
        self.p = subprocess.Popen([EXE, "--serve", "--gguf", GGUF, "--max-context", "4096", "--image-pad-id", str(PAD), *extra,
                                   *os.environ.get("STRATA_METAL_EXTRA", "").split()],
                                  stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.info = {}
        for line in self.p.stdout:
            if line.startswith("INFO "):
                self.info.update(kv.split("=", 1) for kv in line.split()[1:])
            if line.startswith("READY"):
                self.ready = line.split()
                break
        self.im_start = int(self.info["im_start"])
        self.later = []                  # BT / BDONE lines that came while an admission was read
        self.n_embd = int(self.info["n_embd"])

    def send(self, line: str):
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()

    def line(self) -> str:
        return self.p.stdout.readline().strip()

    def gen(self, ids, max_new=16, stop_after=None, image=None):
        """-> (tokens, resume, DONE fields); `image`: an embeddings file (GENI)"""
        head = f"GENI {max_new} {image}" if image else f"GEN {max_new}"
        self.send(head + " " + ",".join(map(str, ids)))
        toks, resume = [], None
        while True:
            line = self.line()
            if line.startswith("RESUME "):
                resume = int(line.split()[1])
            elif line.startswith("T "):
                toks.append(int(line[2:]))
                if stop_after is not None and len(toks) == stop_after:
                    self.send("STOP")
            elif line.startswith("DONE "):
                return toks, resume, line.split()
            elif line.startswith("ERR"):
                raise AssertionError(line)

    def admit(self, slot, ids, max_new):
        """BGEN -> (first token, resume, continues)"""
        self.send(f"BGEN {slot} {max_new} " + ",".join(map(str, ids)))
        toks, resume = [], None
        while True:
            line = self.line()
            if line.startswith("RESUME "):
                resume = int(line.split()[1])
            elif line.startswith("T "):
                toks.append(int(line[2:]))
            elif line.startswith(("BT ", "BDONE ")):              # other slots decode between the commands
                self.later.append(line)
            elif line.startswith("BADM "):
                f = line.split()
                return toks, resume, f[1] == str(slot) and f[2] == "1"
            elif line.startswith("ERR"):
                raise AssertionError(line)

    def slots(self, stop_after=None):
        """BT / BDONE until every admitted slot is done -> {slot: (tokens, finish)}"""
        got, done = {}, {}
        while True:
            f = (self.later.pop(0) if self.later else self.line()).split()
            if f[0] == "BT":
                got.setdefault(int(f[1]), []).append(int(f[2]))
                if stop_after and len(got[int(f[1])]) == stop_after[1] and int(f[1]) == stop_after[0]:
                    self.send(f"BSTOP {stop_after[0]}")
            elif f[0] == "BDONE":
                done[int(f[1])] = (got.get(int(f[1]), []), f[3])
                if len(done) == self.admitted:
                    return done

    def close(self):
        self.send("QUIT")
        self.p.wait(timeout=30)


def message(im_start, body):
    return [im_start] + body


@unittest.skipUnless(EXE and GGUF, "set STRATA_METAL_EXE and STRATA_METAL_GGUF")
class TestStrataMetal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.e = Engine()
        s = cls.e.im_start
        cls.m1 = message(s, list(range(300, 340)))
        cls.m2 = message(s, list(range(500, 560)))
        cls.m3 = message(s, list(range(700, 720)))
        cls.mx = message(s, list(range(900, 930)))

    @classmethod
    def tearDownClass(cls):
        cls.e.close()

    def test_ready(self):
        self.assertEqual(self.e.ready, ["READY", "4096", "stop"])
        self.assertEqual(self.e.info["backend"], "metal")

    def test_divergence_goes_back_to_the_boundary_and_matches_a_fresh_engine(self):
        a = self.m1 + self.m2 + self.m3
        b = self.m1 + self.m2 + self.mx
        self.e.gen(a)
        out_b, resume, _ = self.e.gen(b)
        self.assertEqual(resume, len(self.m1) + len(self.m2))     # the checkpoint before m3's <|im_start|>
        fresh = Engine()
        try:
            out_fresh, resume0, _ = fresh.gen(b)
        finally:
            fresh.close()
        self.assertEqual(resume0, 0)
        self.assertEqual(out_b, out_fresh)

    def test_a_new_question_after_one_long_message_reads_only_the_question(self):
        # skill-quiz's shape: one user message "Document: ... Question: q" - the document is no message of its own
        e = Engine("--prefill-batch", "64")
        try:
            doc = self.m1 + message(e.im_start, list(range(1000, 1300)))
            e.gen(doc + list(range(700, 720)))
            out_b, resume, _ = e.gen(doc + list(range(900, 930)))
        finally:
            e.close()
        self.assertGreater(resume, len(doc) - 64)                 # a checkpoint inside the message, not its start
        self.assertLessEqual(resume, len(doc))
        fresh = Engine()
        try:
            out_fresh, _, _ = fresh.gen(doc + list(range(900, 930)))
        finally:
            fresh.close()
        self.assertEqual(out_b, out_fresh)

    def test_checkpoints_inside_a_long_message_keep_the_boundaries(self):
        # the long message's own checkpoints (many more than --checkpoints) never push out the one before it
        e = Engine("--prefill-batch", "32", "--checkpoints", "2")
        try:
            head = self.m1 + self.m2
            e.gen(head + message(e.im_start, list(range(1000, 1400))))
            _, resume, _ = e.gen(head + self.mx)
        finally:
            e.close()
        self.assertEqual(resume, len(head))

    def test_a_json_schema_holds_the_reply_to_it(self):
        # response_format by grammar: json_schema=<hex> on the GEN line, converted by llama.cpp; the reply can only be
        # JSON the schema accepts, and a schema that does not parse is refused with the engine still in step
        import json
        from tools.strata_tokenizer import Tokenizer
        self.assertEqual(self.e.info.get("json_schema"), "1")
        schema = {"type": "object", "properties": {"answer": {"enum": ["yes", "no"]}}, "required": ["answer"],
                  "additionalProperties": False}
        hexs = json.dumps(schema).encode().hex()
        self.e.send(f"GEN 40 json_schema={hexs} " + ",".join(map(str, self.m1)))
        toks = []
        while True:
            line = self.e.line()
            if line.startswith("T "):
                toks.append(int(line[2:]))
            elif line.startswith(("DONE", "ERR")):
                self.assertTrue(line.startswith("DONE"), line)
                break
        text = Tokenizer.from_gguf(GGUF).decode([t for t in toks if t != self.e.im_start]).strip()
        self.assertIn(json.loads(text.split("<|im_end|>")[0].strip())["answer"], ("yes", "no"), text)
        self.e.send(f"GEN 8 json_schema={b'{not json'.hex()} " + ",".join(map(str, self.m1)))
        while not (line := self.e.line()).startswith(("ERR", "DONE")):
            pass
        self.assertTrue(line.startswith("ERR") and "json_schema" in line, line)
        self.assertEqual(len(self.e.gen(self.m1, max_new=3)[0]), 3)

    def test_a_lazy_json_schema_starts_after_its_trigger_token(self):
        # the server's thinking models: the schema holds from the thinking's end token on (grammar_trigger).  llama.cpp
        # feeds a lazy grammar its trigger token too: a schema grammar that did not begin with it ended the engine
        # ("empty grammar stack after accepting </think>").  The first token this model writes stands in for </think>.
        import json
        from tools.strata_tokenizer import Tokenizer
        tok = Tokenizer.from_gguf(GGUF)
        first = self.e.gen(self.m1, max_new=1)[0][0]
        schema = {"type": "object", "properties": {"answer": {"enum": ["yes", "no"]}}, "required": ["answer"],
                  "additionalProperties": False}
        keys = f"json_schema={json.dumps(schema).encode().hex()} grammar_trigger={first}"
        for slot in (None, 0):
            e = Engine("--batch", "1")
            try:
                if slot is None:
                    e.send(f"GEN 40 {keys} " + ",".join(map(str, self.m1)))
                else:
                    e.send(f"BGEN 0 40 {keys} " + ",".join(map(str, self.m1)))
                toks, done = [], False
                while not done:
                    line = e.line()
                    self.assertFalse(line.startswith("ERR") or line == "", f"slot {slot}: {line or 'the engine ended'}")
                    f = line.split()
                    if f[0] == "T" or (f[0] == "BT" and f[1] == "0"):
                        toks.append(int(f[-1]))
                    # a slot's admission ends with its own DONE (the first token); the slot goes on until BDONE
                    done = (slot is None and f[0] == "DONE") or (f[0] == "BDONE" and f[1] == "0")
                self.assertEqual(toks[0], first)                       # the trigger, freely
                text = tok.decode(toks[1:]).split("<|im_end|>")[0].strip()
                try:
                    answer = json.loads(text)["answer"]
                except ValueError:
                    self.fail(f"slot {slot}: not JSON: {text!r} from {toks[:12]}")
                self.assertIn(answer, ("yes", "no"), f"slot {slot}: {text!r}")
                self.assertEqual(len(e.gen(self.m1, max_new=2)[0]), 2)  # still in step
            finally:
                e.close()

    def test_a_grammar_error_ends_the_request_not_the_engine(self):
        # a lazy grammar that does not begin with its trigger token makes llama.cpp throw when the trigger comes ("empty
        # grammar stack after accepting ..."): the request (or the slot) ends, the engine goes on for everyone else
        first = self.e.gen(self.m1, max_new=1)[0][0]
        keys = f"grammar={b'root ::= [a-z]+'.hex()} grammar_trigger={first}"
        e = Engine("--batch", "1")
        try:
            e.send(f"GEN 8 {keys} " + ",".join(map(str, self.m1)))
            while not (line := e.line()).startswith(("ERR", "DONE")):
                self.assertNotEqual(line, "", "the engine ended")
            self.assertTrue(line.startswith("ERR") and "sampling" in line, line)
            self.assertEqual(len(e.gen(self.m1, max_new=2)[0]), 2)
            e.send(f"BGEN 0 8 {keys} " + ",".join(map(str, self.m1)))
            while not (line := e.line()).startswith(("BDONE", "ERR")):
                self.assertNotEqual(line, "", "the engine ended")
            self.assertTrue(line.startswith(("BDONE 0", "ERR")), line)
            self.assertEqual(len(e.gen(self.m1, max_new=2)[0]), 2)
        finally:
            e.close()

    def test_a_displaced_conversation_comes_back_from_the_ram_cache(self):
        # document A, then document B in the same sequence (A's cells are gone), then a new question on A: the RAM
        # cache gives A back instead of reading it again, and the answer is the one a fresh engine gives
        s = self.e.im_start
        doc_a = self.m1 + message(s, list(range(1000, 1300)))
        doc_b = self.m1 + message(s, list(range(2000, 2300)))
        resumes = {}
        for extra in ((), ("--conversation-cache-mib", "512")):
            e = Engine("--prefill-batch", "64", *extra)
            try:
                if extra:
                    self.assertEqual(e.info.get("conversation_cache_mib"), "512")
                e.gen(doc_a + list(range(700, 720)))
                e.gen(doc_b + list(range(700, 720)))
                out, resumes[extra], _ = e.gen(doc_a + list(range(900, 930)))
            finally:
                e.close()
        self.assertEqual(resumes[()], len(self.m1))                       # without it: back to the shared start
        self.assertGreater(resumes[("--conversation-cache-mib", "512")], len(doc_a) - 64)
        fresh = Engine()
        try:
            self.assertEqual(out, fresh.gen(doc_a + list(range(900, 930)))[0])
        finally:
            fresh.close()

    def test_the_ram_cache_evicts_the_oldest_conversation(self):
        # two cache slots, four documents: A and B are parked, then C evicts A (the oldest); B comes back, A does not
        s = self.e.im_start
        docs = [self.m1 + message(s, list(range(b, b + 300))) for b in (1000, 2000, 3000, 4000)]
        e = Engine("--prefill-batch", "64", "--conversation-cache-mib", "512", "--conversation-cache-slots", "2")
        try:
            for d in docs:
                e.gen(d + list(range(700, 720)))
            _, back_b, _ = e.gen(docs[1] + list(range(900, 930)))
            _, back_a, _ = e.gen(docs[0] + list(range(900, 930)))
        finally:
            e.close()
        self.assertGreater(back_b, len(docs[1]) - 64)
        self.assertEqual(back_a, len(self.m1))

    def test_checkpoint_tail_reads_the_end_in_small_chunks(self):
        # the default batch (2048) reads this whole prompt in one chunk: no checkpoint inside the long message, unless
        # --checkpoint-tail cuts its last tokens into chunks of 256
        doc = self.m1 + message(self.e.im_start, list(range(1000, 1700)))
        resumes = {}
        for extra in ((), ("--checkpoint-tail", "512")):
            e = Engine(*extra)
            try:
                e.gen(doc + list(range(700, 720)))
                out, resumes[extra], _ = e.gen(doc + list(range(900, 930)))
            finally:
                e.close()
        self.assertEqual(resumes[()], len(self.m1))                              # back to the message's start
        self.assertGreater(resumes[("--checkpoint-tail", "512")], len(doc) - 256)  # within one small chunk
        fresh = Engine()
        try:
            self.assertEqual(out, fresh.gen(doc + list(range(900, 930)))[0])
        finally:
            fresh.close()

    def test_the_latest_inside_checkpoints_are_kept(self):
        # --checkpoints 2: the long message's last two chunk ends; one early in it went, so the message's start serves
        e = Engine("--prefill-batch", "32", "--checkpoints", "2")
        try:
            a = self.m1 + message(e.im_start, list(range(1000, 1400)))
            e.gen(a)
            _, late, _ = e.gen(a[:-5] + self.mx)
            e.gen(a)
            _, early, _ = e.gen(a[:len(self.m1) + 100] + self.mx)
        finally:
            e.close()
        self.assertGreater(late, len(a) - 5 - 32)
        self.assertLessEqual(late, len(a) - 5)
        self.assertEqual(early, len(self.m1))

    def test_a_boundary_at_a_chunk_end_is_a_boundary(self):
        # the first message is exactly one chunk: its end is the next message's <|im_start|>, a boundary, not pushed
        # out by the long message's inside checkpoints
        e = Engine("--prefill-batch", "32", "--checkpoints", "2")
        try:
            head = message(e.im_start, list(range(300, 331)))
            self.assertEqual(len(head), 32)
            e.gen(head + message(e.im_start, list(range(1000, 1400))))
            _, resume, _ = e.gen(head + self.mx)
        finally:
            e.close()
        self.assertEqual(resume, 32)

    def test_a_long_image_across_chunks_comes_back_whole(self):
        # an image of 128 rows read in chunks of 32, then a new question after it: what is held matches a fresh engine
        e = Engine("--prefill-batch", "32")
        try:
            with tempfile.TemporaryDirectory() as d:
                one = os.path.join(d, "one.bin")
                embeddings(one, [(16, 8, 0.01)], e.n_embd)
                ids = self.m1 + [PAD] * 128
                e.gen(ids + self.m3, image=one)
                out, resume, _ = e.gen(ids + self.mx, image=one)
                fresh = Engine()
                try:
                    self.assertEqual(out, fresh.gen(ids + self.mx, image=one)[0])
                finally:
                    fresh.close()
        finally:
            e.close()
        self.assertGreater(resume, len(self.m1))
        self.assertLessEqual(resume, len(ids))

    def test_no_checkpoints_reads_it_all_again(self):
        e = Engine("--prefill-batch", "64", "--checkpoints", "0")
        try:
            doc = self.m1 + message(e.im_start, list(range(1000, 1300)))
            e.gen(doc + list(range(700, 720)))
            out, resume, _ = e.gen(doc + list(range(900, 930)))
        finally:
            e.close()
        self.assertEqual(resume, 0)
        fresh = Engine()
        try:
            self.assertEqual(out, fresh.gen(doc + list(range(900, 930)))[0])
        finally:
            fresh.close()

    def test_a_batch_slot_reads_only_the_new_question_too(self):
        b = Engine("--batch", "1", "--prefill-batch", "64")
        try:
            doc = self.m1 + message(b.im_start, list(range(1000, 1300)))
            b.admit(0, doc + list(range(700, 720)), 4)
            b.admitted = 1
            b.slots()
            first, resume, _ = b.admit(0, doc + list(range(900, 930)), 8)
            b.admitted = 1
            got = b.slots()
        finally:
            b.close()
        self.assertGreater(resume, len(doc) - 64)
        self.assertLessEqual(resume, len(doc))
        fresh = Engine()
        try:
            self.assertEqual(first + got[0][0], fresh.gen(doc + list(range(900, 930)), max_new=8)[0])
        finally:
            fresh.close()

    def test_an_extended_prompt_reads_only_the_new_part(self):
        a = self.m1 + self.m3
        out, _, done = self.e.gen(a)
        nxt = a + out + self.mx
        _, resume, done2 = self.e.gen(nxt)
        self.assertEqual(resume, len(a) + len(out) - 1)            # all held: the prompt and every token but the last
        self.assertEqual(int(done2[14]), len(self.mx) + 1)         # prompt tokens read

    def test_stop_cancels(self):
        toks, _, done = self.e.gen(self.m2 + self.mx, max_new=2000, stop_after=2)
        self.assertEqual(done[5], "cancel")
        self.assertLess(len(toks), 50)

    def test_save_restore(self):
        a = self.m1 + self.m2
        self.e.gen(a, max_new=4)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.bin")
            self.e.send(f"SAVE {path}")
            saved = self.e.line().split()
            self.assertEqual(saved[0], "SAVED")
            held = int(saved[1])
            other = Engine()
            try:
                other.send(f"RESTORE {path}")
                restored = other.line().split()
                self.assertEqual(restored[:2], ["RESTORED", str(held)])
                other.send(f"RESTORE {path}.missing")
                self.assertTrue(other.line().startswith("SERR invalid 0 "))
            finally:
                other.close()

    def test_images_are_compared_by_their_content(self):
        with tempfile.TemporaryDirectory() as d:
            one, two = os.path.join(d, "one.bin"), os.path.join(d, "two.bin")
            embeddings(one, [(4, 2, 0.01)], self.e.n_embd)
            embeddings(two, [(4, 2, -0.02)], self.e.n_embd)
            ids = self.m1 + [PAD] * 8 + self.m3
            out1, _, _ = self.e.gen(ids, image=one)
            _, resume, _ = self.e.gen(ids, image=one)                   # the same picture: read again only the end
            self.assertGreaterEqual(resume, len(self.m1) + 8)
            _, resume, _ = self.e.gen(ids, image=two)                   # same ids, another picture: not past m1
            self.assertLessEqual(resume, len(self.m1))
            fresh = Engine()
            try:
                self.assertEqual(fresh.gen(ids, image=one)[0], out1)      # what a fresh engine says for picture one
            finally:
                fresh.close()

    def test_bad_image_requests_are_refused_in_step(self):
        with tempfile.TemporaryDirectory() as d:
            one = os.path.join(d, "one.bin")
            embeddings(one, [(4, 2, 0.01)], self.e.n_embd)
            for ids, image in ((self.m1 + [PAD] * 7 + self.m3, one),        # fewer pads than rows
                               (self.m1 + [PAD] * 8, one),                   # ends in the image
                               (self.m1 + self.m3, one),                     # an image the prompt does not have
                               (self.m1 + [PAD] * 8 + self.m3, os.path.join(d, "missing.bin"))):
                with self.assertRaises(AssertionError):
                    self.e.gen(ids, image=image)
            bad = os.path.join(d, "bad.bin")
            embeddings(bad, [(4, 2, 0.01)], self.e.n_embd + 1)            # the wrong width
            with self.assertRaises(AssertionError):
                self.e.gen(self.m1 + [PAD] * 8 + self.m3, image=bad)
        self.assertEqual(len(self.e.gen(self.m1, max_new=2)[0]), 2)       # still in step

    def test_batch_slots(self):
        b = Engine("--batch", "2")
        try:
            self.assertEqual(b.info["batch_slots"], "2")
            a = self.m1 + self.m2
            solo, _, _ = b.gen(a, max_new=12)                              # the solo path's answer (seq 0)
            first, resume, cont = b.admit(0, a + [self.m3[1]], 12)         # a different prompt in slot 0
            self.assertTrue(cont)
            b.admitted = 1
            got = b.slots()
            self.assertEqual(len(got[0][0]) + 1, 12)                       # the admission's token + 11 windows
            self.assertEqual(got[0][1], "length")
            part, _, _ = b.gen(a, max_new=4)                               # a solo request the server then promotes:
            self.assertEqual(part, solo[:4])
            first, resume, cont = b.admit(1, a + part, 8)                  # BGEN with the prompt + its tokens so far
            self.assertEqual(resume, len(a) + 3)                           # seq 0 copied over: nothing read again
            b.admitted = 1
            got = b.slots()
            self.assertEqual(solo[:4] + first + got[1][0], solo)          # and it goes on as the solo path would have
            b.admit(0, a, 400)
            with self.assertRaises(AssertionError):                       # a busy slot is refused
                b.admit(0, a, 4)
            b.send("BSTOP 0")
            b.admitted = 1
            self.assertEqual(b.slots()[0][1], "cancel")
            # two slots at once: each says what it says alone (a slot sampled another slot's row once - a C++
            # overload took the row index for a draft token - and only this comparison sees that)
            x, y = self.m1 + self.m3, self.m2 + self.mx
            alone_x, _, _ = b.gen(x, max_new=10)
            alone_y, _, _ = b.gen(y, max_new=10)
            fx, _, _ = b.admit(0, x, 10)
            fy, _, _ = b.admit(1, y, 10)
            b.admitted = 2
            got = b.slots()
            # one K/V pool (--kv-unified): attention over the shared pool sums in another order, and on these made-up
            # prompts (near-tied top tokens) that alone changes a greedy answer after a few tokens; a slot reading
            # another's row shows from the first token, which is what this guards
            n = 4 if "--kv-unified" in os.environ.get("STRATA_METAL_EXTRA", "") else 10
            self.assertEqual((fx + got[0][0])[:n], alone_x[:n])
            self.assertEqual((fy + got[1][0])[:n], alone_y[:n])
            b.admit(0, self.m1 + self.m2 + self.m3, 400)                   # two slots at once, one stopped
            b.admit(1, self.m3 + self.m1, 400)
            b.admitted = 2
            got = b.slots(stop_after=(1, 3))
            self.assertEqual(got[1][1], "cancel")
            self.assertLess(len(got[1][0]), 40)
            self.assertIn(got[0][1], ("length", "stop"))
            self.assertEqual(len(b.gen(self.m1, max_new=2)[0]), 2)         # the solo path still in step
        finally:
            b.close()

    def test_a_prompt_read_lets_decoding_slots_go_on(self):
        # a slot decoding while another slot's long prompt is read gets tokens during the read (a window between
        # --prefill-share pieces), not only after it
        b = Engine("--batch", "2", "--prefill-share", "64")
        try:
            b.admit(0, self.m1 + self.m2, 400)
            b.send("BGEN 1 4 " + ",".join(str(300 + i % 40) for i in range(1500)))
            during, reading = 0, False
            while True:
                line = b.line()
                if line.startswith("RESUME "):
                    reading = True
                elif line.startswith("BT 0 ") and reading:
                    during += 1
                elif line.startswith("BADM ") or line.startswith("ERR"):
                    break
            self.assertTrue(line.startswith("BADM 1 1"), line)
            self.assertGreaterEqual(during, 1500 // 64 - 1)              # a token a piece
            b.send("BSTOP 0")
            b.send("BSTOP 1")
            b.admitted = 2
            b.slots()
        finally:
            b.close()

    def test_vip_holds_the_engine_until_released(self):
        # VIP 1 (a client engine's request runs): this engine gives way between passes until VIP 0
        import select
        self.e.send("VIP 1")
        self.e.send("GEN 8 " + ",".join(map(str, self.m1)))
        ready, _, _ = select.select([self.e.p.stdout], [], [], 1.0)
        held = []
        while ready:                                  # RESUME may come; no token while held
            line = self.e.line()
            held.append(line)
            self.assertFalse(line.startswith(("T ", "DONE")), held)
            ready, _, _ = select.select([self.e.p.stdout], [], [], 0.5)
        self.e.send("VIP 0")
        while True:
            line = self.e.line()
            if line.startswith("DONE"):
                break
        self.assertEqual(len(self.e.gen(self.m1, max_new=2)[0]), 2)   # still in step

    def test_refusals_stay_in_step(self):
        for cmd in ("GEN x", "GENI 4 1,2", "VRAM 100", "HELLO"):
            self.e.send(cmd)
            self.assertTrue(self.e.line().startswith("ERR"), cmd)
        self.assertEqual(len(self.e.gen(self.m1, max_new=2)[0]), 2)


@unittest.skipUnless(EXE, "set STRATA_METAL_EXE")
class TestArguments(unittest.TestCase):
    def test_one_draft_kind_at_a_time(self):
        for pair in (["--mtp", "a.gguf", "--dflash", "b.gguf"], ["--eagle3", "a.gguf", "--dflash", "b.gguf"]):
            r = subprocess.run([EXE, "--serve", "--gguf", "x.gguf"] + pair, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(r.returncode, 0, pair)
            self.assertIn("only one of them", r.stderr, pair)

if __name__ == "__main__":
    unittest.main()
