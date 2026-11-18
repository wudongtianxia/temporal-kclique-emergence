"""Core library: temporal graph windowing, clique enumeration with search-state
statistics, candidate pool construction, and feature extraction.

Design notes
------------
* Edges carry a timestamp. A node pair may interact many times; we keep
  (count, first_time, last_time) per pair.
* The graph is *cumulative*: an edge, once observed, stays for all later
  windows ("contact established" semantics for interaction networks).
* A k-clique is a k-element node set, all of whose C(k,2) edges exist.
* An *emergence* event at window t+1 is a k-clique present in S_{t+1} but
  absent in S_t.
* The candidate pool at t consists of node sets of size k that miss exactly
  one edge in S_t. Such a set has a unique (k-1)-clique core, which is what
  the enumeration search tree visits, so we can attach the enumerator's own
  search-state statistics to every candidate.
"""
import gzip
import os

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# loading / windowing
# --------------------------------------------------------------------------
def load_edges(path):
    """Read a SNAP temporal edge list, drop self loops, return sorted-by-time."""
    rows = []
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            a, b, t = int(parts[0]), int(parts[1]), int(parts[2])
            if a != b:
                rows.append((a, b, t))
    arr = np.array(rows, dtype=np.int64)
    order = np.argsort(arr[:, 2], kind="stable")
    arr = arr[order]
    # compact node ids (np.unique flattens row-major, so inverse must be
    # reshaped back into (u, v) columns)
    uniq, inv = np.unique(arr[:, :2], return_inverse=True)
    inv = inv.reshape(-1, 2)
    arr[:, 0] = inv[:, 0]
    arr[:, 1] = inv[:, 1]
    lo = np.minimum(arr[:, 0], arr[:, 1])
    hi = np.maximum(arr[:, 0], arr[:, 1])
    arr[:, 0], arr[:, 1] = lo, hi
    return arr, uniq


def window_index(times, t0, delta):
    return (times - t0) // delta


# --------------------------------------------------------------------------
# temporal graph with edge statistics
# --------------------------------------------------------------------------
class TemporalGraph:
    def __init__(self):
        self.adj = {}
        self.cnt = {}      # (u,v) -> interaction count
        self.last = {}     # (u,v) -> last timestamp
        self.first = {}    # (u,v) -> first timestamp

    def n_edges(self):
        return len(self.cnt)

    def n_nodes(self):
        return len(self.adj)

    def add(self, u, v, t):
        self.adj.setdefault(u, set()).add(v)
        self.adj.setdefault(v, set()).add(u)
        k = (u, v) if u < v else (v, u)
        self.cnt[k] = self.cnt.get(k, 0) + 1
        self.last[k] = t
        self.first.setdefault(k, t)

    def add_batch(self, us, vs, ts):
        for u, v, t in zip(us, vs, ts):
            self.add(int(u), int(v), int(t))

    def deg(self, u):
        return len(self.adj.get(u, ()))


# --------------------------------------------------------------------------
# clique enumeration with search-state statistics
# --------------------------------------------------------------------------
def enumerate_cliques(adj, k, collect_stats=False):
    """Enumerate all k-cliques of an undirected graph.

    Vertices are ordered by their key; extensions only use neighbours with a
    larger order, which makes every clique appear exactly once.

    With collect_stats, also returns a dict keyed by the (k-1)-clique core:
        cand_size    : |P| at the moment the core node is entered, i.e. how
                       many completions the search could still try
        skipped      : number of already-explored siblings in the parent
        parent_cand  : |P| of the parent node (candidate frontier one level up)
        subtree_calls: recursive calls spent inside this core's subtree
    Every one of these is a by-product of the enumeration, not an extra cost.
    """
    order = {v: i for i, v in enumerate(sorted(adj))}
    out = []
    stats = {}

    def walk(R, P, depth, skipped, parent_cand):
        if depth == k:
            out.append(R)
            return 0
        if collect_stats and depth == k - 1:
            stats[R] = (len(P), skipped, parent_cand)
        calls = 0
        for i, v in enumerate(P):
            R2 = R + (v,)
            P2 = [u for u in P[i + 1:] if u in adj[v]]
            calls += 1 + walk(R2, P2, depth + 1, i, len(P))
        return calls

    calls = 0
    for u in sorted(adj):
        # P must be ordered by vertex order, otherwise the emitted tuples are
        # not canonical even though the counts still line up.
        P = sorted((v for v in adj[u] if order[v] > order[u]), key=order.get)
        calls += 1 + walk((u,), P, 1, 0, 0)
    if collect_stats:
        stats["__calls__"] = calls
        return out, stats
    return out


def clique_keys(adj, k):
    """Sorted tuple keys for all k-cliques."""
    out = []
    order = {v: i for i, v in enumerate(sorted(adj))}

    def walk(R, P, depth):
        if depth == k:
            out.append(R)
            return
        for i, v in enumerate(P):
            R2 = R + (v,)
            P2 = [u for u in P[i + 1:] if u in adj[v]]
            walk(R2, P2, depth + 1)

    for u in sorted(adj):
        P = sorted((v for v in adj[u] if order[v] > order[u]), key=order.get)
        walk((u,), P, 1)
    return out


# --------------------------------------------------------------------------
# candidate pool: sets of size k missing exactly one edge
# --------------------------------------------------------------------------
def candidates_from_cores(adj, k, cap=None, time_cap=None):
    """For every (k-1)-clique core, enumerate nodes adding exactly one missing edge.

    cap / time_cap exist so that an exploding pool (dense graph) aborts loudly
    instead of exhausting memory.
    """
    import time as _time
    t0 = _time.time()
    cores = set(clique_keys(adj, k - 1))
    cands = []
    for ci, core in enumerate(cores):
        if time_cap and _time.time() - t0 > time_cap:
            raise TimeoutError(f"candidate generation exceeded {time_cap}s at core {ci}")
        if cap and len(cands) > cap:
            raise TimeoutError(f"candidate pool exceeded cap {cap}")
        core_list = list(core)
        # choose the k-2 support nodes and the single excluded node w:
        # x must be adjacent to all support nodes and NOT adjacent to w.
        for wi in range(len(core_list)):
            w = core_list[wi]
            support = core_list[:wi] + core_list[wi + 1:]
            if len(support) == 0:
                # k == 2 is not used
                continue
            nbrs = set(adj.get(support[0], ()))
            for s in support[1:]:
                nbrs &= adj.get(s, set())
            nbrs -= adj.get(w, set())
            nbrs -= set(core_list)
            nbrs -= {w}
            for x in nbrs:
                cand = tuple(sorted(core + (x,)))
                cands.append((core, x, w, cand))
    return cands


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------
def _decay(t_now, t_edge, tau):
    if t_edge is None:
        return 0.0
    return float(np.exp(-(t_now - t_edge) / tau))


def feature_row(tg, core, x, w, t_now, tau_list, window_idx, hist_rate, stats_entry, deg_rank):
    """One candidate -> feature dict. Every input refers to information <= t_now."""
    feats = {}
    k = len(core) + 1
    feats["k"] = k

    # --- missing pair (w, x) is the edge that must appear -------------------
    key_wx = (w, x) if w < x else (x, w)
    deg_w = tg.deg(w)
    deg_x = tg.deg(x)
    cn = tg.adj.get(w, set()) & tg.adj.get(x, set())
    feats["cn_wx"] = len(cn)
    feats["deg_w"] = deg_w
    feats["deg_x"] = deg_x
    feats["deg_min"] = min(deg_w, deg_x)
    feats["deg_max"] = max(deg_w, deg_x)
    feats["deg_prod_log"] = float(np.log1p(deg_w) + np.log1p(deg_x))
    feats["deg_ratio"] = float(min(deg_w, deg_x) / max(1, max(deg_w, deg_x)))
    aa = 0.0
    for z in cn:
        dz = tg.deg(z)
        if dz > 1:
            aa += 1.0 / np.log(dz)
    feats["aa_wx"] = float(aa)
    uni = tg.adj.get(w, set()) | tg.adj.get(x, set())
    feats["jaccard_wx"] = float(len(cn) / max(1, len(uni)))

    # --- support edges: how strong is the existing half of the clique -------
    core_edges = [(core[i], core[j]) for i in range(len(core)) for j in range(i + 1, len(core))]
    counts, decays, ages = [], [], []
    for a, b in core_edges:
        key = (a, b) if a < b else (b, a)
        counts.append(tg.cnt.get(key, 0))
        ages.append(t_now - tg.last.get(key, t_now))
        decays.append(_decay(t_now, tg.last.get(key), tau_list[0]))
    feats["core_edge_cnt_min"] = int(min(counts)) if counts else 0
    feats["core_edge_cnt_mean"] = float(np.mean(counts)) if counts else 0.0
    feats["core_edge_age_mean"] = float(np.mean(ages)) if ages else 0.0
    feats["core_edge_age_max"] = float(np.max(ages)) if ages else 0.0
    feats["core_decay"] = float(np.mean(decays)) if decays else 0.0

    # --- edges from x into the core: these already exist -------------------
    x_edges = []
    for a in core:
        key = (a, x) if a < x else (x, a)
        rec = tg.last.get(key)
        x_edges.append((tg.cnt.get(key, 0), rec))
    n_present = sum(1 for c, _ in x_edges if c > 0)
    feats["n_x_edges"] = n_present
    last_x = [r for _, r in x_edges if r is not None]
    feats["x_edge_last_min_age"] = float(t_now - min(last_x)) if last_x else 0.0
    feats["x_edge_cnt_sum"] = int(sum(c for c, _ in x_edges))
    for i, tau in enumerate(tau_list):
        s = sum(_decay(t_now, r, tau) for _, r in x_edges)
        feats[f"x_decay_{i}"] = float(s)

    # --- edges from w into the core (all exist too) ------------------------
    w_edges = []
    for a in core:
        key = (a, w) if a < w else (w, a)
        rec = tg.last.get(key)
        w_edges.append((tg.cnt.get(key, 0), rec))
    last_w = [r for _, r in w_edges if r is not None]
    feats["w_edge_cnt_sum"] = int(sum(c for c, _ in w_edges))
    for i, tau in enumerate(tau_list):
        feats[f"w_decay_{i}"] = float(sum(_decay(t_now, r, tau) for _, r in w_edges))

    # temporal co-activity: same third parties recently touching w and x
    both = cn
    if both:
        t_w = [tg.last.get(((w, z) if w < z else (z, w))) for z in both]
        t_x = [tg.last.get(((x, z) if x < z else (z, x))) for z in both]
        gaps = [abs(a - b) / max(1.0, tau_list[-1]) for a, b in zip(t_w, t_x)
                if a is not None and b is not None]
        feats["co_activity_mean"] = float(np.mean(gaps)) if gaps else 1e3
        feats["co_activity_max"] = float(np.max(gaps)) if gaps else 1e3
    else:
        feats["co_activity_mean"] = 1e3
        feats["co_activity_max"] = 1e3

    # --- enumeration search state (the core innovation) --------------------
    if stats_entry is not None:
        cand_size, skipped, parent_cand = stats_entry
    else:
        cand_size, skipped, parent_cand = 0, 0, 0
    feats["search_cand_size"] = int(cand_size)
    feats["search_skipped"] = int(skipped)
    feats["search_parent_cand"] = int(parent_cand)
    feats["search_scan_pos"] = float(skipped / max(1, parent_cand))
    feats["search_cand_per_deg"] = float(cand_size / max(1, deg_w))

    # --- normalisation / context ------------------------------------------
    feats["deg_rank_w"] = float(deg_rank.get(w, 0.0))
    feats["deg_rank_x"] = float(deg_rank.get(x, 0.0))
    feats["window_idx"] = int(window_idx)
    feats["hist_emerge_rate"] = float(hist_rate)

    # --- core identity kept out of the model, used for bookkeeping ---------
    feats["core_0"] = core[0]
    return feats


def degree_rank(adj):
    """Percentile rank of each node's degree (uses only current snapshot)."""
    degs = {v: len(nb) for v, nb in adj.items()}
    arr = np.array(list(degs.values()), dtype=float)
    if arr.size == 0:
        return {}
    order = np.argsort(np.argsort(arr))
    rank = order / max(1, arr.size - 1)
    out = {}
    for v, r in zip(degs.keys(), rank):
        out[v] = float(r)
    return out
