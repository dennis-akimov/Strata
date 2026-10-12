"""strata-image's DiT against the port: relative error of the host conditioning, every block and the output.

    python tools/image/check_dit.py <dump>      (after: strata-image --dit <gguf> --check <dump> [--each])
"""
import sys
from pathlib import Path

import numpy as np


def rel(a, b):
    return float(np.linalg.norm(a.astype(np.float64) - b) / max(np.linalg.norm(b.astype(np.float64)), 1e-30))


d = Path(sys.argv[1])
each = sorted(d.glob("each*_cpp.npy"))
for f in each:                    # every block alone (strata-image --each): its own error, nothing carried in
    i = int(f.stem[4:6])
    print(f"block {i:2d} alone: rel. error {rel(np.load(f), np.load(d / f'block{i:02d}.npy')):.2e}")
if each:
    sys.exit(0)
worst = 0.0
for i in range(64):
    ref, got = d / f"block{i:02d}.npy", d / f"block{i:02d}_cpp.npy"
    if not ref.exists() or not got.exists():
        break
    r = rel(np.load(got), np.load(ref))
    worst = max(worst, r)
    print(f"block {i:2d}: rel. error {r:.2e}")
r = rel(np.load(d / "out_cpp.npy"), np.load(d / "out.npy"))
print(f"output  : rel. error {r:.2e}")
sys.exit(0 if r < 1e-2 else 1)
