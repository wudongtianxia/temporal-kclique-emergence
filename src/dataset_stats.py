"""Step 0: raw dataset profiling.

Reads the six SNAP temporal edge lists (u v timestamp) and reports basic scale
statistics, so that window granularity and k are chosen from measured data
rather than guessed.
"""
import gzip
import json
import os

import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
OUT = os.path.join(os.path.dirname(__file__), "..", "results")

DATASETS = {
    "CollegeMsg": "CollegeMsg.txt.gz",
    "email-Eu-core-temporal": "email-Eu-core-temporal.txt.gz",
    "sx-mathoverflow-a2q": "sx-mathoverflow-a2q.txt.gz",
    "sx-askubuntu-a2q": "sx-askubuntu-a2q.txt.gz",
    "sx-superuser-a2q": "sx-superuser-a2q.txt.gz",
}


def load(path):
    """SNAP edge lists are whitespace-separated: src dst timestamp."""
    rows = []
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            a, b, t = line.split()[:3]
            rows.append((int(a), int(b), int(t)))
    df = pd.DataFrame(rows, columns=["u", "v", "t"])
    return df


def main():
    os.makedirs(OUT, exist_ok=True)
    records = []
    for name, fn in DATASETS.items():
        df = load(os.path.join(DATA, fn))
        # undirected edge list without self loops, timestamp in seconds
        df = df[df.u != df.v].copy()
        nodes = pd.unique(pd.concat([df.u, df.v]))
        span_days = (df.t.max() - df.t.min()) / 86400.0
        daily = df.assign(day=((df.t - df.t.min()) // 86400).astype(int))
        per_day = daily.groupby("day").size()
        rec = {
            "dataset": name,
            "edges": int(len(df)),
            "nodes": int(len(nodes)),
            "t_min": int(df.t.min()),
            "t_max": int(df.t.max()),
            "span_days": round(float(span_days), 1),
            "days_with_edges": int(per_day.size),
            "edges_per_day_mean": round(float(per_day.mean()), 2),
            "edges_per_day_p50": float(per_day.median()),
            "edges_per_day_p95": float(per_day.quantile(0.95)),
            "max_edges_per_day": int(per_day.max()),
        }
        records.append(rec)
        print(rec)

    stats = pd.DataFrame(records)
    stats.to_csv(os.path.join(OUT, "dataset_stats.csv"), index=False)
    with open(os.path.join(OUT, "dataset_stats.json"), "w") as f:
        json.dump(records, f, indent=2)
    print("\nraw edge-list profile written to results/dataset_stats.csv")


if __name__ == "__main__":
    main()
