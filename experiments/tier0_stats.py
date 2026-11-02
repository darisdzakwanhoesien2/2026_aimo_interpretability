#!/usr/bin/env python3
"""Uncertainty and selection-bias check for the Tier-0 surface-feature classifier.

The training script reports CV accuracy at a threshold chosen on the same CV
predictions, which is optimistic. This script reports:
  - the all-False baseline with a bootstrap interval,
  - repeated 5-fold CV accuracy with the threshold chosen inside each training fold (nested),
  - AUC with a bootstrap interval and a label-permutation p-value.

Usage:
    python experiments/tier0_stats.py --csv <math-robust-final.csv> [--out experiments/tier0_stats.json]
Run it with scikit-learn 1.8.0 (the competition pin) so results match the shipped artifact.
"""

import argparse
import csv
import json
import pathlib
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import feature_matrix  # noqa: E402

SEEDS = 20
N_BOOT = 2000
N_PERM = 1000
GRID = np.linspace(0.15, 0.85, 71)


def load(csv_path):
    problems, y = [], []
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            if row["permutation_type"] == "OVERALL":
                problems.append(row["original_problem"])
                y.append(int(row["model_is_robust"].strip().lower() == "true"))
    return problems, np.asarray(y)


def make(name):
    if name == "logreg":
        return Pipeline([("s", StandardScaler()), ("c", LogisticRegression(max_iter=2000, class_weight="balanced", C=0.3))])
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0, min_samples_leaf=15, random_state=0)


def best_threshold(proba, y):
    return max(GRID, key=lambda t: ((proba >= t).astype(int) == y).mean())


def nested_accuracy(X, y, name, seed):
    """Threshold picked on out-of-fold probabilities of the training part only."""
    outer = StratifiedKFold(5, shuffle=True, random_state=seed)
    correct = 0
    for tr, te in outer.split(X, y):
        inner = StratifiedKFold(5, shuffle=True, random_state=seed + 1)
        oof = np.zeros(len(tr))
        for itr, ite in inner.split(X[tr], y[tr]):
            oof[ite] = make(name).fit(X[tr][itr], y[tr][itr]).predict_proba(X[tr][ite])[:, 1]
        thr = best_threshold(oof, y[tr])
        proba = make(name).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        correct += int(((proba >= thr).astype(int) == y[te]).sum())
    return correct / len(y)


def oof_proba(X, y, name, seed):
    out = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        out[te] = make(name).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default=str(ROOT / "experiments" / "tier0_stats.json"))
    args = ap.parse_args()
    problems, y = load(args.csv)
    X = feature_matrix(problems)
    rng = np.random.default_rng(0)
    n, pos = len(y), int(y.sum())

    base = (y == 0).astype(float)
    boots = [rng.choice(base, n).mean() for _ in range(N_BOOT)]
    result = {
        "n": n, "n_robust": pos,
        "all_false_accuracy": round(float(base.mean()), 4),
        "all_false_ci95": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)],
        "models": {},
    }
    for name in ("logreg", "hgb"):
        accs = [nested_accuracy(X, y, name, s) for s in range(SEEDS)]
        p = oof_proba(X, y, name, 0)
        auc = roc_auc_score(y, p)
        auc_boot = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, n, n)
            if len(set(y[idx])) == 2:
                auc_boot.append(roc_auc_score(y[idx], p[idx]))
        perm = [roc_auc_score(rng.permutation(y), p) for _ in range(N_PERM)]
        result["models"][name] = {
            "nested_cv_accuracy_mean": round(float(np.mean(accs)), 4),
            "nested_cv_accuracy_sd_over_seeds": round(float(np.std(accs)), 4),
            "auc": round(float(auc), 4),
            "auc_ci95": [round(float(np.percentile(auc_boot, 2.5)), 4), round(float(np.percentile(auc_boot, 97.5)), 4)],
            "auc_permutation_p": round(float((1 + sum(a >= auc for a in perm)) / (1 + N_PERM)), 4),
        }
        print(name, result["models"][name], flush=True)
    pathlib.Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
