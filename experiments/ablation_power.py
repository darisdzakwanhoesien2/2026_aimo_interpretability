#!/usr/bin/env python3
"""E6 feature-group ablation (AUC) and E7 minimum detectable accuracy gain for the Tier-0 surface-feature classifier.

E6: out-of-fold AUC, mean and sd over SEEDS stratified 5-fold shuffles, for each feature group alone and with the group left out.
E7: power of a one-sided exact binomial test of H0 "accuracy <= BASE" at alpha 0.05 on n problems, for a classifier whose true accuracy is BASE + gain.

Usage: python experiments/ablation_power.py --csv <math-robust-final.csv> [--out experiments/ablation_power.json]
Run with the scikit-learn version recorded in the output (the shipped artifact pins 1.8.0).
"""
import argparse
import json
import math
import pathlib
import sys

import numpy as np
import sklearn
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from features import FEATURE_NAMES, feature_matrix  # noqa: E402
from tier0_stats import load, make, oof_proba  # noqa: E402

SEEDS = 20
GROUPS = {
    "length": ["char_len", "word_len", "log_char_len", "n_sentences", "mean_sentence_len"],
    "numbers_and_operators": ["n_digits", "digit_frac", "n_numbers", "n_operators", "n_equals", "n_variables_single"],
    "latex_and_case": ["n_latex_cmd", "n_dollar", "latex_frac", "has_fraction_bar", "uppercase_frac"],
    "keywords": [n for n in FEATURE_NAMES if n.startswith("has_") and n != "has_fraction_bar"],
}
assert sorted(sum(GROUPS.values(), [])) == sorted(FEATURE_NAMES)


def mean_auc(X, y, name):
    a = [roc_auc_score(y, oof_proba(X, y, name, s)) for s in range(SEEDS)]
    return round(float(np.mean(a)), 4), round(float(np.std(a)), 4)


def binom_power(n, p0, p1, alpha=0.05):
    pmf = lambda k, p: math.comb(n, k) * p**k * (1 - p) ** (n - k)
    crit = next(k for k in range(n + 1) if sum(pmf(j, p0) for j in range(k, n + 1)) <= alpha)
    return round(sum(pmf(j, p1) for j in range(crit, n + 1)), 3), crit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default=str(ROOT / "experiments" / "ablation_power.json"))
    args = ap.parse_args()
    problems, y = load(args.csv)
    X = feature_matrix(problems)
    idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
    cols = lambda names: [idx[n] for n in names]
    res = {"sklearn": sklearn.__version__, "n": int(len(y)), "seeds": SEEDS, "auc": {}}
    for name in ("logreg", "hgb"):
        r = {"all": mean_auc(X, y, name)}
        for g, names in GROUPS.items():
            rest = [n for n in FEATURE_NAMES if n not in names]
            r[f"only_{g}"] = mean_auc(X[:, cols(names)], y, name)
            r[f"without_{g}"] = mean_auc(X[:, cols(rest)], y, name)
        res["auc"][name] = r
        print(name, r, flush=True)
    n, base = len(y), round(float((y == 0).mean()), 4)
    res["power"] = {"n": n, "baseline_accuracy": base, "gains": {}}
    for g in (0.02, 0.03, 0.05, 0.08, 0.10, 0.15):
        power, crit = binom_power(n, base, base + g)
        res["power"]["gains"][str(g)] = {"power": power, "min_correct": crit}
    print(res["power"])
    pathlib.Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
