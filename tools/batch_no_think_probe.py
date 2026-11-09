#!/usr/bin/env python3
"""Two batched decodes garble each other when the prompt ends in an empty thought.

Seen 2026-11-07 on a 16 GB M1 (LFM2.5-8B-A1B, parallel 4, server 2f6a1193): with
reasoning_effort "none" the server ends the prompt with "<think>\\n\\n</think>\\n\\n" (templates with
no switch of their own, see template.rs EMPTY_THINK). One request at a time answered clean JSON;
two or more at once lost words ("Your has received will reviewed our team"), so response_format
answers came back 502 structured_output_failed. Thinking on (no reasoning_effort) under the same
concurrency was clean in the one run tried.

    STRATA_KEY=... python3 tools/batch_no_think_probe.py [http://127.0.0.1:8090] [N]

The prompts are tools/batch_no_think_probe.json: four made-up job-application emails (~600 tokens
each, the shape of a mail-classifying app's requests), or the short ones below without that file.

INTERMITTENT. Garbled in three runs 12:20-12:45 PST at concurrency 2 and 4 (with the bible quiz
also on the server), clean at 12:55 with the same prompts. Suspect the conversation cache (a
parked sequence restored into a running batch) or a third request joining the batch, not
batching alone. Run it a few times, and alongside other traffic, before calling it fixed.

Prints one line per answer and exits 1 if any answer at concurrency > 1 is not valid JSON.
"""
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import os

URL = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8090").rstrip("/")
N = int(sys.argv[2]) if len(sys.argv) > 2 else 2
MAILS = [
    "Thank you for your interest in working at Northwind Labs. Your application has been received and will be reviewed.",
    "We don't think our Senior Platform Engineer role is the best match for you right now.",
    "Unfortunately, we won't be moving forward with your application at this time.",
    "I'd love to set up a quick 30-minute intro call. Book a time here: example.com/book",
]


PROMPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "batch_no_think_probe.json")
if os.path.exists(PROMPTS):
    with open(PROMPTS) as f:
        MAILS = json.load(f)


def ask(mail) -> str:
    messages = mail if isinstance(mail, list) else [
        {"role": "system", "content": 'Reply with JSON only: {"evidence": "<copied sentence>", "status": "ack|rejected|interview"}'},
        {"role": "user", "content": mail},
    ]
    body = {"model": "local", "temperature": 0, "max_tokens": 120, "reasoning_effort": "none", "messages": messages}
    req = urllib.request.Request(
        f"{URL}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {os.environ.get('STRATA_KEY', '')}"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def valid(text: str) -> bool:
    try:
        json.loads(text[text.index("{"): text.rindex("}") + 1])
        return True
    except ValueError:
        return False


bad = 0
for n in (1, N):
    with ThreadPoolExecutor(n) as pool:
        for text in pool.map(ask, MAILS):
            ok = valid(text)
            bad += (not ok) and n > 1
            print(f"concurrency {n} {'ok ' if ok else 'BAD'} {text[:100]!r}")
sys.exit(1 if bad else 0)
