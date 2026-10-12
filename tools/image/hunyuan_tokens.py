"""The token sequence HunyuanImage-3.0-Instruct-Distil generates a picture from, as its own tokenizer and image processor
make it (sequence_template "instruct", the "en_unified" system prompt, bot_task "image": no think, no recaption):

    <|startoftext|> SYSTEM "\\n\\n" "User: " PROMPT "\\n\\n" "Assistant: " <answer> <boi> <img_size_1024> <img_ratio_R>
    <timestep> <guidance> <timestep_r> <img> x (h * w) <eoi> </answer> "\\n\\n"

Each text piece is tokenized alone (the template's sections).  The prompt is text: a prompt that spells `<img>` does not
become the token (the reference would make it one).  The tokens after the image are never attended to by the image and
are left out.  RoPE positions [y, x]: a text token's are its index; the image token (r, c)'s are
L + (h*w - h) // 2 + r and L + (h*w - w) // 2 + c, with L the first image token's index (build_2d_rope).

    python tools/image/hunyuan_tokens.py tokenizer.json "prompt" HEIGHTxWIDTH > tokens.txt

prints what strata-image --hunyuan reads: "HEIGHT WIDTH" and the token ids before <timestep>, whitespace-separated.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from strata_tokenizer import Tokenizer  # noqa: E402

SYSTEM = (pathlib.Path(__file__).with_name("hunyuan_system_prompt.txt")).read_text(encoding="utf-8")
BOS, ANSWER, BOI, SIZE_1024, TIMESTEP, GUIDANCE, TIMESTEP_R, IMG = 127958, 128025, 128000, 128037, 128017, 130082, \
    130107, 128006
# image_processor.py's vae_reso_group (base 1024, align 16) and its 4 extra sizes: index -> (height, width)
RESO = [(512, 2048 - 64 * i) for i in range(9)] + [(576 + 64 * i, 1472 - 64 * i) for i in range(7)] + [(1024, 1024)] + \
       [(1088 + 64 * i, 960 - 64 * i) for i in range(7)] + [(1536 + 64 * i, 512) for i in range(9)] + \
       [(1024, 768), (1280, 720), (768, 1024), (720, 1280)]
assert len(RESO) == 37 and RESO[16] == (1024, 1024) and RESO[24] == (1536, 512)


def ratio_token(i):
    return 128044 + i if i <= 32 else 130103 + i - 33


def resolution(height, width):
    """The nearest size the model was trained for: (index, height, width), the first one of the closest ratio."""
    r = height / width
    i = min(range(len(RESO)), key=lambda k: abs(RESO[k][0] / RESO[k][1] - r))
    return (i, *RESO[i])


def build(tok: Tokenizer, prompt: str, height: int, width: int) -> dict:
    i, H, W = resolution(height, width)
    h, w = H // 16, W // 16
    ids = [BOS] + tok.encode(SYSTEM, parse_special=True) + tok.encode("\n\n") + tok.encode("User: ") + \
        tok.encode(prompt) + tok.encode("\n\n") + tok.encode("Assistant: ") + \
        [ANSWER, BOI, SIZE_1024, ratio_token(i), TIMESTEP, GUIDANCE, TIMESTEP_R]
    L = len(ids)
    ids += [IMG] * (h * w)
    pos = [[p, p] for p in range(L)] + [[L + (h * w - h) // 2 + r, L + (h * w - w) // 2 + c]
                                        for r in range(h) for c in range(w)]
    return {"ids": ids, "image_start": L, "grid": [h, w], "height": H, "width": W, "pos": pos}


if __name__ == "__main__":
    t = Tokenizer.from_hf_json(sys.argv[1], "hunyuan")
    hh, ww = map(int, sys.argv[3].split("x"))
    b = build(t, sys.argv[2], hh, ww)
    print(b["height"], b["width"])
    print(" ".join(map(str, b["ids"][:b["image_start"] - 3])))
