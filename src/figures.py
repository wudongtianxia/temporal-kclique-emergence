"""Step 8: figures for the report. English labels only, to avoid depending on
system CJK fonts in matplotlib."""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({"figure.dpi": 160, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3})


def events_frame():
    frames = []
    for p in glob.glob(os.path.join(OUT, "tables", "*_events.csv")):
        frames.append(pd.read_csv(p))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fig_emergence(ev):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    for (ds, k), g in ev.groupby(["dataset", "k"]):
        ax[0].plot(g.window, g.n_pos_total, marker=".", ms=3, lw=1,
                   label=f"{ds} k={k}")
        ax[1].plot(g.window, g.n_pool, marker=".", ms=3, lw=1)
    ax[0].set_xlabel("window index")
    ax[0].set_ylabel("newly emerging k-cliques")
    ax[0].set_title("Emergence rate per transition")
    ax[1].set_xlabel("window index")
    ax[1].set_ylabel("candidate pool size")
    ax[1].set_title("Candidate pool growth")
    ax[0].legend(fontsize=5.5, ncol=1)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_emergence_pool.png"))
    plt.close(fig)


def fig_coverage(ev):
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for (ds, k), g in ev.groupby(["dataset", "k"]):
        cov = g.coverage.where(g.n_pos_total > 0)   # empty windows have no target
        ax.plot(g.window, cov, marker=".", ms=3, lw=1, label=f"{ds} k={k}")
    ax.set_xlabel("window index")
    ax.set_ylabel("coverage = in-pool / all emerging")
    ax.set_title("Candidate pool coverage (windows without emergence left blank)")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=5.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_coverage.png"))
    plt.close(fig)


def fig_walkforward():
    files = glob.glob(os.path.join(OUT, "T1b_walkforward_*.csv"))
    if not files:
        return
    res = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    order = ["struct", "struct+temp", "struct+temp+ctx", "all(+search)"]
    data = [res[res.config == c].auc.values for c in order]
    ax[0].boxplot(data, tick_labels=order, showfliers=False)
    ax[0].set_ylabel("per-window AUC")
    ax[0].set_title("Feature-family ablation (walk-forward)")
    ax[0].tick_params(axis="x", labelrotation=12, labelsize=7)
    for (ds, k), g in res.groupby(["dataset", "k"]):
        piv = g.pivot_table(index="window", columns="config", values="auc")
        ax[1].plot(piv.index, piv["struct"], marker=".", ms=3, lw=1,
                   label=f"{ds[:14]} k={k} struct")
        ax[1].plot(piv.index, piv["all(+search)"], marker="s", ms=3, lw=1,
                   label=f"{ds[:14]} k={k} all")
    ax[1].set_xlabel("window index")
    ax[1].set_ylabel("AUC")
    ax[1].legend(fontsize=5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_ablation.png"))
    plt.close(fig)


def fig_budget():
    files = glob.glob(os.path.join(OUT, "T3_budget_*.csv"))
    if not files:
        return
    res = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    n = len(res.dataset.unique()) * res.k.nunique()
    fig, axes = plt.subplots(2, 3, figsize=(8.6, 5.0), squeeze=False)
    flat = [a for row in axes for a in row]
    for ax, (key, g) in zip(flat, res.groupby(["dataset", "k"])):
        piv = g.pivot_table(index="frac", columns="strategy", values="recall")
        for c in piv.columns:
            ax.plot(piv.index * 100, piv[c].values, marker="o", ms=3, label=c)
        ax.set_xscale("log")
        ax.set_title(f"{key[0][:16]} k={key[1]}", fontsize=8)
        ax.set_xlabel("top f% of pool verified")
        ax.set_ylabel("recall of emerging cliques")
    used = len(res.groupby(["dataset", "k"]))
    for a in flat[used:]:
        a.axis("off")
    flat[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_budget.png"))
    plt.close(fig)


def fig_importance():
    files = sorted(glob.glob(os.path.join(OUT, "T1_importance_*.csv")))
    if not files:
        return
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    agg = {}
    for f in files:
        d = pd.read_csv(f)
        tot = d.gain.sum()
        for _, r in d.iterrows():
            agg[r.feature] = agg.get(r.feature, 0) + r.gain / max(1, tot)
    s = pd.Series(agg).sort_values(ascending=False).head(14)[::-1]
    ax.barh(s.index, s.values)
    ax.set_xlabel("normalised gain (sum over datasets)")
    ax.set_title("Feature importance (M1)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_importance.png"))
    plt.close(fig)


def fig_t2():
    files = glob.glob(os.path.join(OUT, "T2_raw_*.csv"))
    if not files:
        return
    frames = []
    for f in files:
        d = pd.read_csv(f)
        stem = os.path.basename(f)[len("T2_raw_"):-4]
        ds, kk = stem.rsplit("_k", 1)
        d["dataset"], d["k"] = ds, int(kk)
        frames.append(d)
    res = pd.concat(frames, ignore_index=True)
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    for (ds, k), g in res.groupby(["dataset", "k"]):
        ax.plot(g[g.model == "naive_last"].y.values, label=f"{ds[:14]} k={k} actual")
        ax.plot(g[g.model == "naive_last"].pred.values, ls="--", lw=1,
                label=f"{ds[:14]} k={k} naive")
        ax.plot(g[g.model == "gbdt"].pred.values, ls=":", lw=1,
                label=f"{ds[:14]} k={k} gbdt")
    ax.set_xlabel("test window (order)")
    ax.set_ylabel("emerging cliques")
    ax.set_title("T2: one-step forecasting")
    ax.legend(fontsize=4.5, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig6_t2.png"))
    plt.close(fig)


if __name__ == "__main__":
    ev = events_frame()
    if len(ev):
        fig_emergence(ev)
        fig_coverage(ev)
    fig_walkforward()
    fig_budget()
    fig_importance()
    fig_t2()
    print("figures:", sorted(os.listdir(FIG)))
