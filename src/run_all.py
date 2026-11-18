"""Run every downstream analysis for the tables that actually exist on disk."""
import glob
import os
import re
import sys
import traceback
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd

import importlib

mods = {}
for name, mod_name in (("03_train", "train_t1"), ("04_regress", "train_t2"),
                       ("05_walkforward", "walkforward"), ("06_budget", "budget_analysis")):
    mods[name] = importlib.import_module(mod_name)

TAB = os.path.join(os.path.dirname(__file__), "..", "results", "tables")
OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def available():
    pairs = []
    for p in sorted(glob.glob(os.path.join(TAB, "*.pkl"))):
        base = os.path.basename(p)
        m = re.match(r"(.+)_k(\d)\.pkl", base)
        if m:
            pairs.append((m.group(1), int(m.group(2))))
    return pairs


def main():
    pairs = available()
    print("tables:", pairs, flush=True)
    summary = []
    for ds, k in pairs:
        try:
            out, imp = mods["03_train"].run_one(ds, k)
            print(f"[T1] {ds} k={k}\n{out[['model', 'auc', 'ap', 'recall@0.010']].to_string(index=False)}",
                  flush=True)
            summary.append(out)
        except Exception:
            print(f"[T1] {ds} k={k} FAILED\n{traceback.format_exc()}", flush=True)
        try:
            mods["05_walkforward"].walk_forward(ds, k).to_csv(
                os.path.join(OUT, f"T1b_walkforward_{ds}_k{k}.csv"), index=False)
            print(f"[T1b] {ds} k={k} done", flush=True)
        except Exception:
            print(f"[T1b] {ds} k={k} FAILED\n{traceback.format_exc()}", flush=True)
        try:
            mods["06_budget"].budget(ds, k)
        except Exception:
            print(f"[T3] {ds} k={k} FAILED\n{traceback.format_exc()}", flush=True)
        try:
            res, raw = mods["04_regress"].evaluate("t2", ds, k)
            res.to_csv(os.path.join(OUT, f"T2_metrics_{ds}_k{k}.csv"), index=False)
            raw.to_csv(os.path.join(OUT, f"T2_raw_{ds}_k{k}.csv"), index=False)
            print(f"[T2] {ds} k={k}\n{res.to_string(index=False)}", flush=True)
        except Exception:
            print(f"[T2] {ds} k={k} FAILED\n{traceback.format_exc()}", flush=True)

    cost_rows = []
    for ds, k in pairs:
        try:
            cost_rows.append(mods["06_budget"].cost_probe(ds, k))
        except Exception:
            print(f"[cost] {ds} k={k} FAILED\n{traceback.format_exc()}", flush=True)
    if cost_rows:
        pd.DataFrame(cost_rows).to_csv(os.path.join(OUT, "T3_cost.csv"), index=False)

    if summary:
        allm = pd.concat(summary, ignore_index=True)
        allm.to_csv(os.path.join(OUT, "T1_metrics_all.csv"), index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
