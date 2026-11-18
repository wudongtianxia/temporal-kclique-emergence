"""Measure candidate-pool size and generation cost, to decide which datasets
are tractable for the one-edge-missing candidate formulation.

A hard cap aborts any dataset whose pool explodes, so a dense graph cannot
silently eat the whole experiment budget.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from kclique import TemporalGraph, candidates_from_cores, clique_keys, load_edges

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
CONFIG = {
    "CollegeMsg": ("CollegeMsg.txt.gz", 7),
    "email-Eu-core-temporal": ("email-Eu-core-temporal.txt.gz", 14),
    "sx-mathoverflow-a2q": ("sx-mathoverflow-a2q.txt.gz", 30),
    "sx-askubuntu-a2q": ("sx-askubuntu-a2q.txt.gz", 30),
    "sx-superuser-a2q": ("sx-superuser-a2q.txt.gz", 30),
}
CAP_CANDS = 3_000_000
CAP_SECONDS = 120


def main():
    print(f"{'dataset':26s} {'k':>2s} {'nodes':>8s} {'edges':>8s} {'cores':>9s} "
          f"{'cands':>10s} {'sec':>7s} {'status':>8s}")
    for name, (fn, wd) in CONFIG.items():
        arr, _ = load_edges(os.path.join(DATA, fn))
        t0 = int(arr[:, 2].min())
        delta = wd * 86400
        win = ((arr[:, 2] - t0) // delta).astype(int)
        nwin = int(win.max()) + 1
        mid = max(1, nwin // 2)
        sub = arr[win <= mid]
        tg = TemporalGraph()
        for u, v, t in sub:
            tg.add(int(u), int(v), int(t))
        nedge = len(tg.cnt)
        for k in (3, 4):
            t1 = time.time()
            status = "ok"
            try:
                if k - 1 == 2:
                    cores = nedge  # edges are the 2-cliques
                else:
                    cores = len(clique_keys(tg.adj, k - 1))
                cands = candidates_from_cores(tg.adj, k, cap=CAP_CANDS,
                                             time_cap=CAP_SECONDS)
                ncand = len(cands)
            except TimeoutError:
                cores, ncand, status = cores, -1, "cap"
            elapsed = round(time.time() - t1, 1)
            print(f"{name:26s} {k:2d} {tg.n_nodes():8d} {nedge:8d} "
                  f"{cores:9d} {ncand:10d} {elapsed:7.1f} {status:>8s}")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
