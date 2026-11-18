"""Step 5: walk-forward evaluation and feature-family ablation.

Why this exists: the 70/15/15 holdout leaves only 4 test transitions on
CollegeMsg, which cannot support a claim that feature family X helps. This
script runs an expanding-window walk-forward over the last 40% of transitions
(refit for every test window) and compares feature families window by
window, so a Wilcoxon signed-rank test can be reported instead of eyeballing
a handful of numbers.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import (F_CTX, F_SEARCH, F_STRUCT, F_TEMP, load)

OUT = os.path.join(os.path.dirname(__file__), "..", "results")

CONFIGS = {
    "struct": F_STRUCT,
    "struct+temp": F_STRUCT + F_TEMP,
    "struct+temp+ctx": F_STRUCT + F_TEMP + F_CTX,
    "all(+search)": F_STRUCT + F_TEMP + F_CTX + F_SEARCH,
}


def fit(tr, feats):
    m = lgb.LGBMClassifier(objective="binary", n_estimators=200, learning_rate=0.05,
                          num_leaves=15, min_child_samples=100, subsample=0.9,
                          subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
                          random_state=0, n_jobs=2, verbose=-1)
    m.fit(tr[feats], tr.label.values)
    return m


def walk_forward(dataset, k, test_frac=0.40, min_train_windows=8):
    df = load(dataset, k)
    ws = sorted(df.window.unique())
    start = int(np.floor((1 - test_frac) * len(ws)))
    rows = []
    for w in ws[start:]:
        tr = df[df.window < w]
        te = df[df.window == w]
        if tr.window.nunique() < min_train_windows or te.label.nunique() < 2:
            continue
        y = te.label.values
        for name, feats in CONFIGS.items():
            m = fit(tr, feats)
            p = m.predict_proba(te[feats])[:, 1]
            rows.append({"dataset": dataset, "k": k, "window": w,
                         "config": name, "auc": roc_auc_score(y, p)})
    return pd.DataFrame(rows)


def statistical_compare(res, base="struct+temp+ctx", target="all(+search)"):
    a = res[res.config == base].set_index("window").auc
    b = res[res.config == target].set_index("window").auc
    idx = a.index.intersection(b.index)
    if len(idx) < 6:
        return {"n": len(idx), "delta": float("nan"), "p_value": float("nan")}
    d = (b.loc[idx] - a.loc[idx]).values
    try:
        p = float(wilcoxon(d).pvalue)
    except ValueError:
        p = float("nan")
    return {"n": len(idx), "delta": float(np.mean(d)),
            "delta_median": float(np.median(d)),
            "ci_low": float(np.percentile(d, 2.5)),
            "ci_high": float(np.percentile(d, 97.5)),
            "p_value": p, "wins": int((d > 0).sum())}


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    ds, kk = sys.argv[1], int(sys.argv[2])
    res = walk_forward(ds, kk)
    res.to_csv(os.path.join(OUT, f"T1b_walkforward_{ds}_k{kk}.csv"), index=False)
    tab = res.pivot_table(index="window", columns="config", values="auc")
    print(tab.round(4).to_string())
    print("\nmean AUC per family:")
    print(res.groupby("config").auc.agg(["mean", "std", "count"]).round(4).to_string())
    cmp = statistical_compare(res)
    print("\nsearch-state features vs same model without them:")
    print(cmp)
    pd.DataFrame([{"dataset": ds, "k": kk, **cmp}]).to_csv(
        os.path.join(OUT, f"T1b_test_{ds}_k{kk}.csv"), index=False)
