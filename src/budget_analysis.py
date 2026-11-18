"""Step 6: budget-constrained verification (T3) plus an honest cost accounting.

Question A (benefit): if a practitioner is willing to verify the top f% of the
candidate pool (verification = check whether the single missing edge appeared),
how many of the truly emerging k-cliques are recovered, compared with random
selection and a learning-free heuristic?

Question B (cost): can ranking actually save wall-clock time? We measure
microseconds per candidate for feature construction and compare "score the
whole pool" against "one full clique enumeration of the same snapshot". If
scoring costs more than enumerating, there is no speed-up to claim, only a
forecasting benefit, and the report says so.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from kclique import (TemporalGraph, degree_rank, enumerate_cliques, feature_row,
                     load_edges)
from pipeline import (ALL_FEATURES, iter_candidates, lgb_classifier, load,
                      split_windows)

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
OUT = os.path.join(os.path.dirname(__file__), "..", "results")
FRACS = [0.005, 0.01, 0.05, 0.10, 0.25]

CONFIG = {
    "CollegeMsg": ("CollegeMsg.txt.gz", 7),
    "email-Eu-core-temporal": ("email-Eu-core-temporal.txt.gz", 14),
    "sx-askubuntu-a2q": ("sx-askubuntu-a2q.txt.gz", 30),
    "sx-superuser-a2q": ("sx-superuser-a2q.txt.gz", 30),
}


def recall_curve(df, score):
    s = pd.Series(score, index=df.index)
    out = []
    for f in FRACS:
        cap, tot = 0, 0
        for w, g in df.groupby("window"):
            n_take = max(1, int(round(f * len(g))))
            top = g.loc[s.loc[g.index].sort_values(ascending=False).index[:n_take]]
            cap += top.label.sum()
            tot += g.label.sum()
        out.append({"frac": f, "recall": cap / max(1, tot)})
    return out


def budget(dataset, k):
    df = load(dataset, k)
    tr_ws, va_ws, te_ws = split_windows(df)
    tr = df[df.window.isin(tr_ws)].reset_index(drop=True)
    te = df[df.window.isin(te_ws)].reset_index(drop=True)
    m = lgb_classifier().fit(tr[ALL_FEATURES], tr.label.values)
    p_m1 = m.predict_proba(te[ALL_FEATURES])[:, 1]
    p_heur = (te.aa_wx + te.cn_wx + te.x_decay_0 + te.w_decay_0).values
    rng = np.random.default_rng(0)
    recs = []
    for name, sc in [("RANDOM", rng.random(len(te))), ("HEURISTIC", p_heur),
                     ("M1", p_m1)]:
        for r in recall_curve(te, sc):
            recs.append({"dataset": dataset, "k": k, "strategy": name, **r})
    out = pd.DataFrame(recs)
    out.to_csv(os.path.join(OUT, f"T3_budget_{dataset}_k{k}.csv"), index=False)
    print(out.pivot_table(index="frac", columns="strategy", values="recall").round(3))
    return out


def cost_probe(dataset, k, n_samp=20000):
    fn, wd = CONFIG[dataset]
    arr, _ = load_edges(os.path.join(DATA, fn))
    t0 = int(arr[:, 2].min())
    delta = wd * 86400
    win = ((arr[:, 2] - t0) // delta).astype(int)
    nwin = int(win.max()) + 1
    mid = int(nwin * 0.6)
    tg = TemporalGraph()
    for w in range(mid + 1):
        sub = arr[win == w]
        tg.add_batch(sub[:, 0], sub[:, 1], sub[:, 2])
    _, stats = enumerate_cliques(tg.adj, k, collect_stats=True)
    cores = [c for c in stats if c != "__calls__"]
    t0s = time.time()
    n = 0
    for core, x, w in iter_candidates(tg.adj, k, cores):
        n += 1
    gen_s = time.time() - t0s
    taus = [t * delta for t in (3.0, 9.0, 27.0)]
    dr = degree_rank(tg.adj)
    t1 = time.time()
    cnt = 0
    for core, x, w in iter_candidates(tg.adj, k, cores):
        feature_row(tg, core, x, w, int(mid * delta + t0), taus, 1, 0.0,
                    stats.get(core), dr)
        cnt += 1
        if cnt >= n_samp:
            break
    us = (time.time() - t1) / max(1, cnt) * 1e6
    t2 = time.time()
    enumerate_cliques(tg.adj, k, collect_stats=False)
    enum_s = time.time() - t2
    res = {"dataset": dataset, "k": k, "probe_window": mid, "n_pool": n,
           "pool_gen_seconds": round(gen_s, 2),
           "us_per_candidate_feature": round(us, 2),
           "score_whole_pool_seconds": round(us * n / 1e6, 2),
           "full_enumeration_seconds": round(enum_s, 3),
           "ratio_score_vs_enum": round((us * n / 1e6) / max(1e-9, enum_s), 1)}
    print(res)
    return res


if __name__ == "__main__":
    if sys.argv[1] == "cost":
        rows = [cost_probe(sys.argv[2], int(sys.argv[3]))]
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "T3_cost.csv"), index=False)
    else:
        budget(sys.argv[1], int(sys.argv[2]))
