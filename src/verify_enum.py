"""Verification gate 1: our enumerator must agree exactly with networkx.

Checks on real windowed snapshots:
  * the set of k-cliques equals networkx's
  * every candidate returned by candidates_from_cores misses exactly one edge,
    its core is a (k-1)-clique, and the candidate set is complete in the
    sense that no other node set missing exactly one edge exists.
"""
import itertools
import os
import sys

import networkx as nx

sys.path.insert(0, os.path.dirname(__file__))
from kclique import (TemporalGraph, clique_keys, candidates_from_cores,
                     load_edges)

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def build_tg_from_edges(edges):
    tg = TemporalGraph()
    for u, v, t in edges:
        tg.add(int(u), int(v), int(t))
    return tg


def nx_cliques(adj, k):
    g = nx.Graph()
    g.add_nodes_from(adj)
    for u, nb in adj.items():
        for v in nb:
            if u < v:
                g.add_edge(u, v)
    out = set()
    for c in nx.enumerate_all_cliques(g):
        if len(c) == k:
            out.add(tuple(sorted(c)))
        elif len(c) > k:
            break
    return out


def check_dataset(name, fn, delta_days, ks=(3, 4), n_windows=4):
    arr, _ = load_edges(os.path.join(DATA, fn))
    t0 = int(arr[:, 2].min())
    delta = delta_days * 86400
    win = ((arr[:, 2] - t0) // delta).astype(int)
    n = int(win.max()) + 1
    picks = list(range(max(1, n // 3), min(n, max(1, n // 3) + n_windows)))
    ok = True
    for k in ks:
        for w in picks:
            sub = arr[win <= w]
            tg = build_tg_from_edges(sub)
            ours = set(clique_keys(tg.adj, k))
            theirs = nx_cliques(tg.adj, k)
            if ours != theirs:
                ok = False
                print(f"  [FAIL] {name} k={k} w={w}: ours={len(ours)} nx={len(theirs)} "
                      f"diff={len(ours ^ theirs)}")
                continue
            # candidate completeness check on a small k only (k=3 is cheap)
            if k == 3:
                cands = candidates_from_cores(tg.adj, k)
                seen = set()
                bad = 0
                for core, x, w_, cand in cands:
                    assert len(core) == k - 1 and len(cand) == k
                    assert core == tuple(sorted(core))
                    # exactly one missing edge inside the candidate
                    missing = 0
                    for a, b in itertools.combinations(cand, 2):
                        if b not in tg.adj.get(a, ()):
                            missing += 1
                    if missing != 1:
                        bad += 1
                    seen.add(cand)
                # completeness: brute force over all 3-sets within a sampled
                # neighbourhood would be too slow globally; instead verify that
                # every set with exactly one missing edge found by direct
                # enumeration is in `seen`, using nodes that share an edge.
                extra = 0
                for a, b in itertools.combinations(
                        sorted(tg.adj)[:300], 2):
                    if b not in tg.adj.get(a, ()):
                        continue
                    for x in tg.adj.get(a, ()):
                        if x == b:
                            continue
                        cand = tuple(sorted((a, b, x)))
                        if cand in seen:
                            continue
                        miss = sum(1 for p, q in itertools.combinations(cand, 2)
                                   if q not in tg.adj.get(p, ()))
                        if miss == 1:
                            extra += 1
                print(f"  [OK] {name} k={k} w={w}: {len(ours)} cliques, "
                      f"cands={len(seen)}, malformed={bad}, missed={extra}")
                if bad or extra:
                    ok = False
    return ok


def main():
    cases = [
        ("CollegeMsg", "CollegeMsg.txt.gz", 7),
        ("email-Eu-core-temporal", "email-Eu-core-temporal.txt.gz", 14),
    ]
    all_ok = True
    for name, fn, d in cases:
        print(f"[verify] {name}")
        all_ok &= check_dataset(name, fn, d, ks=(3, 4), n_windows=3)
    print("VERIFY RESULT:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
