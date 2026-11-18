"""Step 2c: compact the candidate tables.

The raw tables are float64 pickles and reach hundreds of MB on the denser
graphs. This script
  * subsamples negatives down to NEG_KEEP times the positive count (a second
    uniform thinning of already uniform negative samples, so the case-control
    structure is preserved and the new rate is written back),
  * casts to float32,
  * rewrites the per-window sampling rate accordingly.
"""
import glob
import os

import numpy as np
import pandas as pd

TAB = os.path.join(os.path.dirname(__file__), "..", "results", "tables")
NEG_KEEP = 10


def compact(path):
    with open(path, "rb") as fh:
        head = fh.read(2)
    if head == b"\x1f\x8b":
        return None            # already compacted
    df = pd.read_pickle(path)
    pos = df.label == 1
    n_pos = int(pos.sum())
    neg = df[~pos]
    keep = min(1.0, (NEG_KEEP * n_pos) / max(1, len(neg)))
    rng = np.random.default_rng(7)
    neg_keep = neg[rng.random(len(neg)) <= keep]
    out = pd.concat([df[pos], neg_keep]).sort_values(["window", "label"],
                                                     ascending=[True, False])
    out["rho"] = out["rho"] * keep       # effective sampling rate (float64)
    for c in out.columns:
        if c not in ("window", "label"):
            try:
                out[c] = out[c].astype(np.float32)
            except (ValueError, TypeError):
                pass
    out["window"] = out["window"].astype(np.int16)
    out["label"] = out["label"].astype(np.int8)
    out.to_pickle(path, compression="gzip")
    return df.shape, out.shape, keep


if __name__ == "__main__":
    for p in sorted(glob.glob(os.path.join(TAB, "*.pkl"))):
        if p.endswith(".pkl.gz"):
            continue
        res = compact(p)
        if res is None:
            print(f"{os.path.basename(p)}: already compacted, skipped")
            continue
        before, after, keep = res
        print(f"{os.path.basename(p)}: {before} -> {after} (neg keep {keep:.3f})")
