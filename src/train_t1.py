"""Step 3: models, baselines and evaluation for candidate-closure prediction (T1).

Protocol
--------
* Split by window index, strictly forward in time: the first 70% of transitions
  train, the next 15% are used for early stopping, the last 15% are test.
* Base rates differ by more than an order of magnitude between windows (a
  saturated network produces almost no new cliques), so metrics are computed
  per window and then averaged (macro), instead of pooling rows across windows.
* The pool is case-control (all positives kept, negatives sampled at rate rho).
  Recall is reported at "top f% of the candidate pool": the sampled equivalent
  is the top f% of the sampled rows, which is the same quantile because the
  negative sampling is uniform.
"""
import os
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import (ALL_FEATURES, F_CTX, F_SEARCH, F_STRUCT, F_TEMP,
                      lgb_classifier, load, split_windows)

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
FRACS = [0.005, 0.01, 0.05, 0.10, 0.25]


def per_window(df, score):
    rows = []
    for w, g in df.groupby("window"):
        y = g.label.values
        if y.min() == y.max():
            continue
        rows.append({"window": w, "auc": roc_auc_score(y, score[g.index]),
                     "ap": average_precision_score(y, score[g.index]),
                     "n_pos": int(y.sum()), "n": len(g)})
    return pd.DataFrame(rows)


def recall_at_frac(df, score, frac):
    s = pd.Series(score, index=df.index)
    cap, tot = 0, 0
    for w, g in df.groupby("window"):
        n_take = max(1, int(round(frac * len(g))))
        top = g.loc[s.loc[g.index].sort_values(ascending=False).index[:n_take]]
        cap += top.label.sum()
        tot += g.label.sum()
    return cap / max(1, tot)


def summarise(df, score):
    pw = per_window(df, score)
    out = {"auc": pw.auc.mean(), "auc_std": pw.auc.std(ddof=1),
           "ap": pw.ap.mean(), "ap_std": pw.ap.std(ddof=1), "n_windows_eval": len(pw)}
    for f in FRACS:
        out[f"recall@{f:.3f}"] = recall_at_frac(df, score, f)
    return out


def fit_gbdt(train, valid, feats, **kw):
    """GBDT with early stopping on the held-out transition block."""
    m = lgb.LGBMClassifier(objective="binary", n_estimators=1000, learning_rate=0.03,
                           num_leaves=15, min_child_samples=100, subsample=0.9,
                           subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
                           random_state=0, n_jobs=2, verbose=-1, **kw)
    m.fit(train[feats], train.label.values,
          eval_set=[(valid[feats], valid.label.values)],
          eval_metric="average_precision",
          callbacks=[lgb.early_stopping(60, verbose=False)])
    return m


def run_one(dataset, k, save=True):
    df = load(dataset, k)
    tr_ws, va_ws, te_ws = split_windows(df)
    tr = df[df.window.isin(tr_ws)].reset_index(drop=True)
    va = df[df.window.isin(va_ws)].reset_index(drop=True)
    te = df[df.window.isin(te_ws)].reset_index(drop=True)
    res = []

    score = (te.aa_wx + te.cn_wx + te.x_decay_0 + te.w_decay_0).values.astype(float)
    res.append({"model": "B1_heuristic", **summarise(te, score)})

    cols = F_STRUCT + F_TEMP
    sc = StandardScaler().fit(tr[cols].values)
    lr = LogisticRegression(max_iter=3000, class_weight="balanced")
    lr.fit(sc.transform(tr[cols].values), tr.label.values)
    score = lr.predict_proba(sc.transform(te[cols].values))[:, 1]
    res.append({"model": "B2_logit_S+T", **summarise(te, score)})

    m3 = fit_gbdt(tr, va, cols)
    res.append({"model": "B3_gbdt_S+T", **summarise(te, m3.predict_proba(te[cols])[:, 1])})

    cols4 = F_STRUCT + F_TEMP + F_CTX
    m4 = fit_gbdt(tr, va, cols4)
    res.append({"model": "B4_gbdt_S+T+ctx", **summarise(te, m4.predict_proba(te[cols4])[:, 1])})

    m1 = fit_gbdt(tr, va, ALL_FEATURES)
    res.append({"model": "M1_gbdt_all", **summarise(te, m1.predict_proba(te[ALL_FEATURES])[:, 1])})

    trs = (tr.groupby("window", group_keys=False)
           .apply(lambda g: g if len(g) <= 8000 else g.sample(n=8000, random_state=1)))
    trs = trs.sort_values("window").reset_index(drop=True)
    ranker = lgb.LGBMRanker(objective="lambdarank", n_estimators=500,
                            learning_rate=0.03, num_leaves=15, min_child_samples=100,
                            subsample=0.9, subsample_freq=1, colsample_bytree=0.7,
                            random_state=0, n_jobs=2, verbose=-1)
    ranker.fit(trs[ALL_FEATURES], trs.label.values,
               group=trs.groupby("window").size().values)
    res.append({"model": "M2_lambdarank",
                **summarise(te, ranker.predict(te[ALL_FEATURES]).astype(float))})

    out = pd.DataFrame(res)
    out.insert(0, "dataset", dataset)
    out.insert(1, "k", k)
    out["n_train"] = len(tr)
    out["pos_train"] = int(tr.label.sum())
    out["n_test"] = len(te)
    out["pos_test"] = int(te.label.sum())
    out["n_windows_test"] = te.window.nunique()

    imp = pd.DataFrame({"feature": ALL_FEATURES,
                        "gain": m1.booster_.feature_importance(importance_type="gain")})
    imp = imp.sort_values("gain", ascending=False)
    if save:
        os.makedirs(OUT, exist_ok=True)
        out.to_csv(os.path.join(OUT, f"T1_metrics_{dataset}_k{k}.csv"), index=False)
        imp.to_csv(os.path.join(OUT, f"T1_importance_{dataset}_k{k}.csv"), index=False)
    return out, imp


if __name__ == "__main__":
    ds, kk = sys.argv[1], int(sys.argv[2])
    out, imp = run_one(ds, kk)
    cols = ["model", "auc", "ap", "recall@0.005", "recall@0.010",
            "recall@0.050", "n_test", "pos_test"]
    print(out[cols].to_string(index=False))
    print("\ntop features by gain:")
    print(imp.head(10).to_string(index=False))
