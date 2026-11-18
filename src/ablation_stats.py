"""Pairwise ablation statistics over all datasets.

For each (dataset, k) pair we compare three successive feature-family additions
on the same windows:
    struct        -> struct+temp        (effect of temporal features)
    struct+temp   -> struct+temp+ctx    (effect of context features)
    struct+temp+ctx -> all(+search)     (effect of search-state features)

The last comparison is the project's core hypothesis. Intervals are bootstrap
percentile intervals (2000 resamples, seed 0); the p-value is a two-sided
Wilcoxon signed-rank test when >= 6 paired windows exist, otherwise an exact
two-sided sign test. Both are reported honestly: with 4 windows nothing can be
called significant.
"""
import glob
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

warnings.filterwarnings("ignore")
OUT = os.path.join(os.path.dirname(__file__), "..", "results")
RNG = np.random.default_rng(0)

PAIRS = [("struct", "struct+temp", "temporal over static"),
         ("struct+temp", "struct+temp+ctx", "context over temporal"),
         ("struct+temp+ctx", "all(+search)", "search-state over best")]


def boot_ci(d, n=2000):
    idx = RNG.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compare(d):
    n = len(d)
    out = {"n": n, "delta": float(np.mean(d)), "delta_median": float(np.median(d)),
           "wins": int((d > 0).sum())}
    out["ci_low"], out["ci_high"] = boot_ci(d) if n >= 2 else (float("nan"), float("nan"))
    if n >= 6:
        try:
            out["p_value"] = float(wilcoxon(d).pvalue)
            out["test"] = "wilcoxon"
        except ValueError:
            out["p_value"], out["test"] = float("nan"), "wilcoxon-failed"
    else:
        k = int((d > 0).sum())
        out["p_value"] = float(binomtest(k, n, 0.5).pvalue)
        out["test"] = "exact-sign"
    return out


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(OUT, "T1b_walkforward_*.csv"))):
        res = pd.read_csv(f)
        ds, k = res.dataset.iloc[0], int(res.k.iloc[0])
        piv = res.pivot_table(index="window", columns="config", values="auc")
        for base, target, label in PAIRS:
            if base not in piv.columns or target not in piv.columns:
                continue
            d = piv[target] - piv[base]
            d = d.dropna().values
            rows.append({"dataset": ds, "k": k, "comparison": label,
                         "base": base, "target": target, **compare(d)})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "T1b_ablation_stats.csv"), index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
