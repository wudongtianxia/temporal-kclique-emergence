"""Shared definitions for the modelling scripts."""
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

TAB = os.path.join(os.path.dirname(__file__), "..", "results", "tables")

F_STRUCT = ["cn_wx", "deg_w", "deg_x", "deg_min", "deg_max", "deg_prod_log",
            "deg_ratio", "aa_wx", "jaccard_wx", "core_edge_cnt_min",
            "core_edge_cnt_mean", "core_edge_age_mean", "core_edge_age_max",
            "core_decay"]
F_TEMP = ["x_edge_last_min_age", "x_edge_cnt_sum", "x_decay_0", "x_decay_1",
          "x_decay_2", "w_edge_cnt_sum", "w_decay_0", "w_decay_1",
          "w_decay_2", "co_activity_mean", "co_activity_max"]
F_SEARCH = ["search_cand_size", "search_skipped", "search_parent_cand",
            "search_scan_pos", "search_cand_per_deg"]
F_CTX = ["deg_rank_w", "deg_rank_x", "window_idx", "hist_emerge_rate"]

ALL_FEATURES = F_STRUCT + F_TEMP + F_SEARCH + F_CTX


def load(dataset, k):
    path = os.path.join(TAB, f"{dataset}_k{k}.pkl")
    if not os.path.exists(path):
        path = path + ".gz"
    with open(path, "rb") as fh:
        head = fh.read(2)
    # gzip magic; pandas only infers compression from the extension, and the
    # tables keep a .pkl name, so this has to be explicit.
    comp = "gzip" if head == b"\x1f\x8b" else None
    return pd.read_pickle(path, compression=comp)


def split_windows(df, train_frac=0.70, valid_frac=0.15):
    ws = sorted(df.window.unique())
    n = len(ws)
    i_tr = max(1, int(np.floor(train_frac * n)))
    i_va = max(i_tr + 1, int(np.floor((train_frac + valid_frac) * n)))
    return ws[:i_tr], ws[i_tr:i_va], ws[i_va:]


def lgb_classifier(**kw):
    p = dict(objective="binary", n_estimators=300, learning_rate=0.05,
             num_leaves=31, min_child_samples=50, subsample=0.9,
             subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
             random_state=0, n_jobs=2, verbose=-1)
    p.update(kw)
    return lgb.LGBMClassifier(**p)


def iter_candidates(adj, k, cores_iter):
    """Yield (core, x, w) for every size-k set missing exactly one edge.

    A set missing one edge has TWO valid cores (drop either endpoint of the
    missing edge). Canonical rule: only emit when the dropped node x is larger
    than the non-adjacent core node w, so each candidate is produced once.
    """
    for core in cores_iter:
        cl = list(core)
        for wi in range(len(cl)):
            w = cl[wi]
            support = cl[:wi] + cl[wi + 1:]
            nbrs = set(adj.get(support[0], ()))
            for s in support[1:]:
                nbrs &= adj.get(s, set())
            nbrs -= adj.get(w, set())
            nbrs -= set(cl)
            for x in nbrs:
                if x < w:
                    continue
                yield core, x, w
