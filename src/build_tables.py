"""Step 2: build the candidate tables (features + label) for one dataset and k.

Protocol
--------
For each window transition t -> t+1 (cumulative snapshot semantics):
  1. S_t is the cumulative snapshot at the end of window t
  2. C_t = k-cliques of S_t, C_{t+1} = k-cliques of S_{t+1}
  3. emergence set E_t = C_{t+1} \\ C_t
  4. candidate pool U_t = all size-k node sets that miss exactly one edge in S_t,
     generated from (k-1)-clique cores. The enumeration pass over S_t also
     yields the search-state statistics attached to each core.
  5. label(candidate) = 1 iff candidate in E_t
  6. case-control sampling: keep every positive, keep each negative with
     probability rho_t = min(1, NEG_PER_POS * |E_t in pool| / |U_t \\ E_t|)

Leakage control: features are read from S_t only. C_{t+1} is obtained on a
throw-away adjacency copy, so no edge of window t+1 can enter a feature.
Pool size, positive count, coverage and rho are stored per window so a reader
can undo the sampling analytically.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from kclique import (TemporalGraph, degree_rank, enumerate_cliques,
                     feature_row, load_edges)

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "tables")

NEG_PER_POS = 20
TAUS = [3.0, 9.0, 27.0]   # decay scales, expressed in windows
MAX_KEEP_PER_WINDOW = 25000   # memory guard: rows are thinned after labelling
SHARD_EVERY = 15              # windows per on-disk shard


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


def build(dataset, fn, wdays, k, seed=0):
    arr, uniq = load_edges(os.path.join(DATA, fn))
    t0 = int(arr[:, 2].min())
    delta = wdays * 86400
    win = ((arr[:, 2] - t0) // delta).astype(int)
    nwin = int(win.max()) + 1
    rng = np.random.default_rng(seed)
    taus_sec = [t * delta for t in TAUS]

    tg = TemporalGraph()
    rows = []
    events = []
    t_start = time.time()
    per_win = {w: arr[win == w] for w in range(nwin)}

    def add_window(w):
        if w in per_win:
            sub = per_win[w]
            tg.add_batch(sub[:, 0], sub[:, 1], sub[:, 2])

    add_window(0)
    add_window(1)

    for t in range(1, nwin - 1):
        # ---- everything below until "add_window(t+1)" sees only S_t -------
        cur_cliques, cur_stats = enumerate_cliques(tg.adj, k, collect_stats=True)
        cur_set = set(cur_cliques)
        del cur_cliques

        adj_next = {v: set(nb) for v, nb in tg.adj.items()}
        if t + 1 in per_win:
            sub = per_win[t + 1]
            for u, v in zip(sub[:, 0], sub[:, 1]):
                adj_next.setdefault(int(u), set()).add(int(v))
                adj_next.setdefault(int(v), set()).add(int(u))
        nxt_set = set(enumerate_cliques(adj_next, k, collect_stats=False))
        del adj_next

        emerge = nxt_set - cur_set
        cores = [c for c in cur_stats if c != "__calls__"]
        n_pos = len(emerge)

        # pass 1: pool / negative counts (no allocation)
        n_pool = 0
        seen_emerge = 0
        for core, x, w in iter_candidates(tg.adj, k, cores):
            n_pool += 1
            if tuple(sorted(core + (x,))) in emerge:
                seen_emerge += 1
        neg_pool = n_pool - seen_emerge
        rho = 1.0
        if neg_pool > 0:
            rho = min(1.0, (NEG_PER_POS * max(1, seen_emerge)) / neg_pool)

        deg_rank = degree_rank(tg.adj)
        hist_rate = float(np.mean([e["event_rate"] for e in events[-3:]])) if events else 0.0

        # pass 2: keep positives + sampled negatives, build features
        kept = []
        for core, x, w in iter_candidates(tg.adj, k, cores):
            cand = tuple(sorted(core + (x,)))
            is_pos = cand in emerge
            if not is_pos and rng.random() > rho:
                continue
            f = feature_row(tg, core, x, w, int(t * delta + t0), taus_sec, t,
                            hist_rate, cur_stats.get(core), deg_rank)
            f["label"] = int(is_pos)
            f["window"] = t
            f["rho"] = rho
            kept.append(f)
        del cur_stats

        df = pd.DataFrame(kept)
        if len(df) > MAX_KEEP_PER_WINDOW:
            pos_mask = df.label.values == 1
            neg = df[~pos_mask]
            n_neg_keep = max(0, MAX_KEEP_PER_WINDOW - int(pos_mask.sum()))
            if len(neg) > n_neg_keep:
                idx = rng.choice(len(neg), size=n_neg_keep, replace=False)
                df = pd.concat([df[pos_mask], neg.iloc[idx]]).sort_values(
                    ["window", "label"], ascending=[True, False]).reset_index(drop=True)
                n_pool_scale = (len(neg) - n_neg_keep) / max(1, n_neg_keep)
                df["rho"] = df["rho"] / (1.0 + n_pool_scale) * 1.0
        if len(df):
            rows.append(df)
        if len(rows) >= SHARD_EVERY:
            shard_dir = os.path.join(OUT, "_shards")
            os.makedirs(shard_dir, exist_ok=True)
            pd.concat(rows, ignore_index=True).to_pickle(
                os.path.join(shard_dir, f"{dataset}_k{k}_{t}.pkl"))
            rows = []

        event_rate = len(emerge) / max(1, len(nxt_set))
        events.append({
            "dataset": dataset, "k": k, "window": t,
            "n_pool": n_pool, "n_pool_neg": neg_pool,
            "n_pos_total": n_pos, "n_pos_in_pool": seen_emerge,
            "coverage": seen_emerge / max(1, n_pos),
            "rho": rho, "n_kept": len(df),
            "n_cliques_t": len(cur_set), "n_cliques_t1": len(nxt_set),
            "event_rate": float(event_rate),
        })
        print(f"  w={t}/{nwin - 2} pool={n_pool} pos={n_pos} "
              f"in_pool={seen_emerge} rho={rho:.4f} kept={len(df)} "
              f"t={time.time() - t_start:.1f}s", flush=True)

        add_window(t + 1)

    shard_dir = os.path.join(OUT, "_shards")
    shard_files = sorted(f for f in os.listdir(shard_dir)
                         if f.startswith(f"{dataset}_k{k}_")) if os.path.isdir(shard_dir) else []
    parts = list(rows)
    del rows
    for f in shard_files:
        parts.append(pd.read_pickle(os.path.join(shard_dir, f)))
    if not parts:
        print("  no rows produced")
        return None, None
    full = pd.concat(parts, ignore_index=True)
    for f in shard_files:
        os.remove(os.path.join(shard_dir, f))
    ev = pd.DataFrame(events)
    os.makedirs(OUT, exist_ok=True)
    full.to_pickle(os.path.join(OUT, f"{dataset}_k{k}.pkl"))
    ev.to_csv(os.path.join(OUT, f"{dataset}_k{k}_events.csv"), index=False)
    return full, ev


CONFIG = {
    "CollegeMsg": ("CollegeMsg.txt.gz", 7),
    "email-Eu-core-temporal": ("email-Eu-core-temporal.txt.gz", 14),
    "sx-mathoverflow-a2q": ("sx-mathoverflow-a2q.txt.gz", 30),
    "sx-askubuntu-a2q": ("sx-askubuntu-a2q.txt.gz", 30),
    "sx-superuser-a2q": ("sx-superuser-a2q.txt.gz", 30),
}

if __name__ == "__main__":
    name = sys.argv[1]
    k = int(sys.argv[2])
    fn, wd = CONFIG[name]
    print(f"[build] {name} k={k} window={wd}d")
    df, ev = build(name, fn, wd, k)
    if df is not None:
        print(f"  rows={len(df)} pos={int(df.label.sum())} "
              f"windows={ev.shape[0]} mean_coverage={ev.coverage.mean():.3f}")
