"""Step 4: T2 -- forecasting the number of newly emerging k-cliques per window.

Inputs at window t (all observable at the end of window t):
  recent emergence counts, clique-stock size and growth, candidate-pool size,
  pool coverage, and mean values of a few candidate features.
Target: |C_{t+1} \\ C_t|, the number of k-cliques that emerge in the next window.

Evaluation is expanding-window walk-forward on the last 40% of windows: for each
test window the model is refit on everything before it. Baselines are the
last-value carryover and a rolling mean, which are hard to beat on such a short
series.
"""
import os
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import TAB, load

OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def window_table(dataset, k):
    ev = pd.read_csv(os.path.join(TAB, f"{dataset}_k{k}_events.csv"))
    df = load(dataset, k)
    agg = df.groupby("window").agg(
        mean_deg_w=("deg_w", "mean"), mean_aa=("aa_wx", "mean"),
        mean_cn=("cn_wx", "mean"), mean_xdecay=("x_decay_0", "mean"),
        mean_search=("search_cand_size", "mean"),
        frac_active=("x_decay_0", lambda s: float((s > 0).mean()))).reset_index()
    t = ev.merge(agg, on="window", how="left").sort_values("window").reset_index(drop=True)
    t["target"] = t.n_pos_total.shift(-1)
    t["prev1"] = t.n_pos_total
    t["prev2"] = t.n_pos_total.shift(1)
    t["prev3"] = t.n_pos_total.shift(2)
    t["roll3"] = t.prev1.rolling(3).mean()
    t["stock_prev"] = t.n_cliques_t
    t["stock_growth"] = t.n_cliques_t - t.n_cliques_t.shift(1)
    return t


FEATS = ["prev1", "prev2", "prev3", "roll3", "stock_prev", "stock_growth",
         "n_pool", "coverage", "event_rate", "mean_deg_w", "mean_aa", "mean_cn",
         "mean_xdecay", "mean_search", "frac_active"]


def evaluate(name, dataset, k, test_frac=0.40):
    t = window_table(dataset, k).dropna(subset=["target"]).reset_index(drop=True)
    n = len(t)
    start = int(np.floor((1 - test_frac) * n))
    rows = []
    for i in range(start, n):
        train = t.iloc[:i].dropna()
        test = t.iloc[[i]]
        if len(train) < 8:
            continue
        preds = {"naive_last": test.prev1.values[0],
                 "roll3": test.roll3.values[0]}
        # AR(1) on the log scale: a one-parameter baseline that respects
        # the multiplicative decay of emergence as the network saturates
        x = np.log1p(train.prev1.values)
        ytr = np.log1p(train.target.values)
        slope, intercept = np.polyfit(x, ytr, 1)
        preds["log_ar1"] = float(np.expm1(slope * np.log1p(test.prev1.values[0]) + intercept))
        m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=7,
                              min_child_samples=10, random_state=0, n_jobs=2,
                              verbose=-1)
        m.fit(train[FEATS], train.target.values)
        preds["gbdt"] = float(m.predict(test[FEATS].values)[0])
        y = float(test.target.values[0])
        for kk, v in preds.items():
            rows.append({"model": kk, "window": int(test.window.values[0]),
                         "y": y, "pred": v})
    r = pd.DataFrame(rows)
    out = []
    for kk, g in r.groupby("model"):
        err = g.pred - g.y
        denom = (np.abs(g.y) + np.abs(g.pred)) / 2
        out.append({"dataset": dataset, "k": k, "model": kk,
                    "mae": float(np.abs(err).mean()),
                    "rmse": float(np.sqrt((err ** 2).mean())),
                    "smape": float((np.abs(err) / np.maximum(1, denom)).mean()),
                    "r2": float(1 - (err ** 2).sum() / max(1e-9, ((g.y - g.y.mean()) ** 2).sum())),
                    "n_test_windows": len(g)})
    return pd.DataFrame(out), r


if __name__ == "__main__":
    ds, kk = sys.argv[1], int(sys.argv[2])
    res, raw = evaluate("t2", ds, kk)
    print(res.to_string(index=False))
    os.makedirs(OUT, exist_ok=True)
    res.to_csv(os.path.join(OUT, f"T2_metrics_{ds}_k{kk}.csv"), index=False)
    raw.to_csv(os.path.join(OUT, f"T2_raw_{ds}_k{kk}.csv"), index=False)
