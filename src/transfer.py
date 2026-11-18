"""Step 7: cross-dataset zero-shot transfer (leave-one-dataset-out).

Train on every dataset except one, test on the held-out dataset, using the same
feature space. Feature values are in different units across graphs (degree,
interaction counts), so the result measures how far the *structural+temporal*
signal generalises without any recalibration. Reported as-is, including the
negative outcomes.
"""
import os
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import ALL_FEATURES, load, split_windows

OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def get(dataset, k):
    df = load(dataset, k)
    tr_ws, va_ws, te_ws = split_windows(df)
    return (df[df.window.isin(tr_ws)].reset_index(drop=True),
            df[df.window.isin(te_ws)].reset_index(drop=True))


def main(pairs):
    rows = []
    for held in pairs:
        tr = pd.concat([get(*p)[0] for p in pairs if p != held], ignore_index=True)
        te = get(*held)[1]
        m = lgb.LGBMClassifier(objective="binary", n_estimators=200,
                              learning_rate=0.05, num_leaves=15,
                              min_child_samples=100, subsample=0.9,
                              subsample_freq=1, colsample_bytree=0.7,
                              reg_lambda=5.0, random_state=0, n_jobs=2, verbose=-1)
        m.fit(tr[ALL_FEATURES], tr.label.values)
        p = m.predict_proba(te[ALL_FEATURES])[:, 1]
        rows.append({"held_out": f"{held[0]} k={held[1]}",
                     "auc": roc_auc_score(te.label.values, p),
                     "ap": average_precision_score(te.label.values, p),
                     "n_train": len(tr), "n_test": len(te),
                     "pos_test": int(te.label.sum())})
        print(rows[-1])
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "T4_transfer.csv"), index=False)
    return out


if __name__ == "__main__":
    # only pairs that share the same k can be pooled
    main([("CollegeMsg", 3), ("email-Eu-core-temporal", 3)])
