"""Step 9: coupling T1 back into T2 -- forecasting the emergence count from the
calibrated candidate probabilities.

Under case-control sampling at rate rho, a classifier trained on the sampled
pool has shifted odds: sampled_odds = true_odds / rho. So the probability has to
be recalibrated before it can be summed as an expectation:

    p_true = odds_true / (1 + odds_true),  odds_true = (p / (1 - p)) * rho

Expected number of emerging cliques in the window is then
    E = n_positive_rows_kept + sum_over_kept_negatives (p_true / rho)

This is the natural way to reuse the T1 model for the T2 target, and it is
compared against the naive carryover / AR(1) / plain GBDT baselines.
"""
import os
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import ALL_FEATURES, TAB, load

OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def calibrate(p, rho):
    odds = (p / np.clip(1 - p, 1e-9, 1)) * rho
    return odds / (1 + odds)


def run(dataset, k, test_frac=0.40):
    df = load(dataset, k)
    ws = sorted(df.window.unique())
    start = int(np.floor((1 - test_frac) * len(ws)))
    rows = []
    for w in ws[start:]:
        tr = df[df.window < w]
        te = df[df.window == w]
        if tr.window.nunique() < 8 or te.label.nunique() < 2:
            continue
        m = lgb.LGBMClassifier(objective="binary", n_estimators=200,
                              learning_rate=0.05, num_leaves=15,
                              min_child_samples=100, subsample=0.9,
                              subsample_freq=1, colsample_bytree=0.7,
                              reg_lambda=5.0, random_state=0, n_jobs=2, verbose=-1)
        m.fit(tr[ALL_FEATURES], tr.label.values)
        p = m.predict_proba(te[ALL_FEATURES])[:, 1]
        rho = float(te.rho.iloc[0])
        p_true = calibrate(p, rho)
        pos = te.label.values == 1
        est = pos.sum() + (p_true[~pos] / rho).sum()
        rows.append({"window": int(w), "y": int(pos.sum()), "pred_sum_prob": est,
                     "rho": rho, "n_kept": len(te)})
    r = pd.DataFrame(rows)
    if r.empty:
        return None
    err = r.pred_sum_prob - r.y
    denom = (np.abs(r.y) + np.abs(r.pred_sum_prob)) / 2
    metrics = {"dataset": dataset, "k": k, "model": "T1_sum_prob_calibrated",
               "mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean())),
               "smape": float((np.abs(err) / np.maximum(1, denom)).mean()),
               "r2": float(1 - (err ** 2).sum() / max(1e-9, ((r.y - r.y.mean()) ** 2).sum())),
               "n_test_windows": len(r)}
    return metrics, r


if __name__ == "__main__":
    ds, kk = sys.argv[1], int(sys.argv[2])
    metrics, r = run(ds, kk)
    print(metrics)
    r.to_csv(os.path.join(OUT, f"T2b_sumprob_raw_{ds}_k{kk}.csv"), index=False)
    pd.DataFrame([metrics]).to_csv(os.path.join(OUT, f"T2b_sumprob_{ds}_k{kk}.csv"), index=False)
