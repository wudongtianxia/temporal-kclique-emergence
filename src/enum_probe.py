"""Step 0b: feasibility probe for clique enumeration under a windowed snapshot.

For each dataset, one representative mid-stream window is built and the cost of
counting triangles (k=3) and 4-cliques (k=4) is measured with a wall-clock cap,
so the plan can decide k per dataset from evidence instead of assumption.
"""
import gzip
import json
import os
import time

import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
OUT = os.path.join(os.path.dirname(__file__), "..", "results")

# (file, window length in days) chosen from the profile in step 0
CONFIG = {
    "CollegeMsg": ("CollegeMsg.txt.gz", 7),
    "email-Eu-core-temporal": ("email-Eu-core-temporal.txt.gz", 14),
    "sx-mathoverflow-a2q": ("sx-mathoverflow-a2q.txt.gz", 30),
    "sx-askubuntu-a2q": ("sx-askubuntu-a2q.txt.gz", 30),
    "sx-superuser-a2q": ("sx-superuser-a2q.txt.gz", 30),
}

CAP_SECONDS = 20.0


def load(path):
    rows = []
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            a, b, t = line.split()[:3]
            if a != b:
                rows.append((int(a), int(b), int(t)))
    return pd.DataFrame(rows, columns=["u", "v", "t"])


def build_adj(edges):
    adj = {}
    for u, v in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    return adj


def count_k3(adj):
    """Triangle count via ordered node triples (u<v<w)."""
    order = {n: i for i, n in enumerate(adj)}
    total = 0
    for u, nbrs in adj.items():
        iu = order[u]
        for v in nbrs:
            if order[v] <= iu:
                continue
            nu = adj[u]
            nv = adj[v]
            small, big = (nu, nv) if len(nu) < len(nv) else (nv, nu)
            total += sum(1 for w in small if w in big and order[w] > order[v])
    return total


def count_k4(adj, cap):
    """Count 4-cliques with Bron-Kerbosch style recursion; raises TimeoutError."""
    t0 = time.time()
    order = {n: i for i, n in enumerate(adj)}
    total = 0

    def ext(cand, excl, size):
        nonlocal total
        if time.time() - t0 > cap:
            raise TimeoutError
        if size == 4:
            total += 1
            return
        if len(cand) < 4 - size:
            return
        for i, v in enumerate(cand):
            new_cand = [u for u in cand[i + 1:] if u in adj[v]]
            new_excl = [u for u in excl if u in adj[v]]
            ext(new_cand, new_excl, size + 1)

    for v in sorted(adj, key=lambda x: order[x]):
        later = [u for u in adj[v] if order[u] > order[v]]
        ext(later, [], 1)
    return total


def main():
    os.makedirs(OUT, exist_ok=True)
    recs = []
    for name, (fn, wdays) in CONFIG.items():
        df = load(os.path.join(DATA, fn))
        t0 = int(df.t.min())
        nwin = int(((df.t.max() - t0) // (wdays * 86400)) + 1)
        mid = nwin // 2
        lo, hi = t0 + mid * wdays * 86400, t0 + (mid + 1) * wdays * 86400
        win = df[(df.t >= lo) & (df.t < hi)]
        # snapshot = edges accumulated up to window start (windowed graph)
        past = df[df.t < hi]
        adj = build_adj(list(zip(past.u, past.v)))
        edges_n = int(len(past))
        t1 = time.time()
        k3 = count_k3(adj)
        k3_cost = round(time.time() - t1, 2)
        try:
            k4 = count_k4(adj, CAP_SECONDS)
            k4_cost = round(time.time() - t1 - k3_cost, 2)
            k4_state = "ok"
        except TimeoutError:
            k4 = None
            k4_cost = CAP_SECONDS
            k4_state = "timeout"
        rec = {
            "dataset": name, "window_days": wdays, "n_windows": nwin,
            "probe_window": mid, "edges_in_window": int(len(win)),
            "cum_edges_at_probe": edges_n, "cum_nodes": len(adj),
            "k3_count": k3, "k3_seconds": k3_cost,
            "k4_count": k4, "k4_state": k4_state, "k4_seconds": k4_cost,
        }
        recs.append(rec)
        print(json.dumps(rec, ensure_ascii=False))
    pd.DataFrame(recs).to_csv(os.path.join(OUT, "enumeration_probe.csv"), index=False)


if __name__ == "__main__":
    main()
