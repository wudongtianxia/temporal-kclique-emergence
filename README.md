# Temporal k-clique emergence: data, models and evaluation

Code and results for a data-analysis project on **forecasting the emergence of
k-cliques in temporal graphs**, plus a budgeted verification analysis of whether
such forecasts can be used to prune clique enumeration.

Everything here runs on real public data and on a two-core, 4 GB, CPU-only
machine. No result in `results/` was hand-written; every table is emitted by
the scripts below.

## Problem

Given a temporal graph and a fixed window length, let `S_t` be the cumulative
snapshot at the end of window `t` and `C_t^k` the set of all k-cliques in
`S_t`. A k-clique **emerges** in window `t+1` if it belongs to
`C_{t+1}^k \ C_t^k`.

* **T1** — predict, for a candidate node set that is one edge short of being a
  k-clique at time `t`, whether it closes within the next window.
* **T2** — forecast the number of emerging k-cliques in the next window.
* **T3** — if only the top fraction of candidates can be verified, how many of
  the true emergences are recovered, and what does verification cost compared
  with plain enumeration.

The candidate universe is exact: every size-k node set missing exactly one edge
in `S_t`. Its coverage of the emergence set is measured, not assumed
(`results/tables/*_events.csv`, column `coverage`); the part that cannot be
covered is the part whose k-cliques involve a node that first appears in the
prediction window.

## Data

Six real SNAP temporal edge lists are downloaded (`data/*.gz`, integrity
checked against the server's `Content-Length`). The experiments use:

| dataset | windows | window length |
|---|---|---|
| CollegeMsg | 28 | 7 days |
| email-Eu-core-temporal | 58 | 14 days |
| sx-askubuntu-a2q | 88 | 30 days |
| sx-superuser-a2q | 93 | 30 days |

`sx-mathoverflow-a2q` and `wiki-talk-temporal` were downloaded but excluded:
their candidate pools exceed 10^7 per window (mathoverflow) or the graph itself
is ~5M edges (wiki-talk), which is not tractable on this machine. The exclusion
is measured in `results/candidate_scale.csv` style probes and reported in the
report; no result was obtained by silently subsampling those graphs.

## Layout

```
src/kclique.py            windowing, clique enumeration with search-state stats,
                          candidate pool construction, feature extraction
src/pipeline.py           shared feature groups and model factory
src/verify_enum.py        gate 1: enumerator must equal networkx exactly
src/build_tables.py       candidate tables + per-window sampling statistics
src/train_t1.py           T1 models and holdout evaluation
src/walkforward.py        expanding-window walk-forward + feature ablation
src/train_t2.py           T2 count forecasting baselines
src/integration.py        T2 from calibrated T1 probabilities
src/budget_analysis.py    T1 -> budgeted recall curves and cost accounting
src/transfer.py           cross-dataset zero-shot transfer
src/ablation_stats.py     pairwise feature-family tests (bootstrap CI + Wilcoxon)
src/figures.py            report figures
src/report_build.py       builds the Word report from results/*.csv
src/run_all.py            runs every analysis for every table on disk
```

## Reproduce

```
python3 -m pip install -r requirements.txt
python3 src/dataset_stats.py          # profile the raw edge lists
python3 src/verify_enum.py            # correctness gate (must print PASS)
python3 src/build_tables.py CollegeMsg 3
python3 src/compact_tables.py
python3 src/run_all.py                # all tables -> all metrics
python3 src/ablation_stats.py         # feature-family paired tests
python3 src/integration.py CollegeMsg 3   # calibrated sum-of-probabilities
python3 src/transfer.py               # zero-shot transfer (k=3)
python3 src/figures.py
python3 src/report_build.py           # Word report
```

`build_tables.py <dataset> <k>` accepts `CollegeMsg`, `email-Eu-core-temporal`,
`sx-askubuntu-a2q`, `sx-superuser-a2q`.

## Honesty notes

* The candidate pool is sampled for negatives (case-control, all positives
  kept). Every window's pool size and sampling rate is stored, so a reader can
  undo the sampling.
* Metrics are macro-averaged over windows because base rates differ by more than
  an order of magnitude between early and late windows.
* Two hypotheses of the project were tested and one of them failed: the
  enumeration search-state features do **not** improve prediction beyond noise.
  Paired tests over 4-33 windows give delta AUC between -0.010 and +0.002 with
  all p-values >= 0.15, and the five features rank 20th and below in importance.
  That negative result is reported with its test statistic instead of being
  dropped.
* Cost accounting is reported as it is: scoring the whole candidate pool takes
  200-1000x the time of a full clique enumeration on the same snapshot, so there
  is no wall-clock speed-up at this scale.
* Coverage is computed only on windows that actually contain emergences; windows
  with no emergence are listed separately, because counting them as zero
  coverage silently biases the number (this happened during development).
* Two windows have an empty candidate pool yet contain emergences - the graph
  had not formed a single (k-1)-clique yet. This is reported, not hidden.

## What is in this repository

Only what is needed to reproduce the study: `src/`, `requirements.txt`, this README.

* Raw edge lists (`data/*.gz`, ~9.5 MB) and the per-window candidate tables
  (`results/tables/*.pkl`, up to 404 MB) are not tracked (see `.gitignore`); the
  commands above re-download and regenerate them.
* The metric files (`results/*.csv`), the figures and the written report belong to
  the submission package, not to this repository, so the repo stays small.