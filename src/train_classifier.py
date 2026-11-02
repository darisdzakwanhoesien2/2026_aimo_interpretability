"""Train the Tier-0 surface-feature robustness classifier on the public CSVs.

Data: examples/baselines/data/math-robust-final.csv, the OVERALL rows only
(one row per problem; per-permutation-type rows dropped). This is `qwen3-8b:low`
on Hendrycks MATH — the only public labelled data.

The bar is the majority-class baseline (always-"spurious"): ~0.745 accuracy here.
We report grouped 5-fold CV accuracy and only keep the model if it beats that
convincingly; otherwise the submission should ship all-False.

Output: artifacts/tier0_classifier.joblib  (pipeline + FEATURE_NAMES + threshold + CV report)
"""

from __future__ import annotations

import csv
import json
import pathlib
import sys

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import FEATURE_NAMES, feature_matrix  # noqa: E402

CSV = ROOT / "examples" / "baselines" / "data" / "math-robust-final.csv"
OUT = ROOT / "artifacts" / "tier0_classifier.joblib"


def load_overall() -> tuple[list[str], list[str], np.ndarray]:
    problems, model_ids, y = [], [], []
    with CSV.open() as fh:
        for row in csv.DictReader(fh):
            if row["permutation_type"] != "OVERALL":
                continue
            problems.append(row["original_problem"])
            model_ids.append(row["model_id"])
            y.append(row["model_is_robust"].strip().lower() == "true")
    return problems, model_ids, np.asarray(y, dtype=int)


def main() -> None:
    problems, model_ids, y = load_overall()
    X = feature_matrix(problems, model_ids)
    n, pos = len(y), int(y.sum())
    majority = 1.0 - pos / n  # always predict "spurious" (0)
    print(f"n={n}  robust={pos} ({pos/n:.3f})  majority-baseline acc={majority:.3f}")

    candidates = {
        "logreg": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=0.3)),
        ]),
        "hgb": HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200,
            l2_regularization=1.0, min_samples_leaf=15, random_state=0,
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    results = {}
    best_name, best_acc, best_thr = None, -1.0, 0.5
    for name, est in candidates.items():
        proba = cross_val_predict(est, X, y, cv=cv, method="predict_proba")[:, 1]
        # tune threshold for accuracy under the real class prior
        thr_grid = np.linspace(0.15, 0.85, 71)
        accs = [(t, ((proba >= t).astype(int) == y).mean()) for t in thr_grid]
        thr, acc = max(accs, key=lambda p: p[1])
        auc = _safe_auc(y, proba)
        results[name] = {"cv_acc_best_thr": float(acc), "threshold": float(thr), "auc": float(auc)}
        print(f"  {name:8s} CV acc={acc:.3f} @thr={thr:.2f}  auc={auc:.3f}")
        if acc > best_acc:
            best_name, best_acc, best_thr = name, acc, thr

    beats_majority = best_acc >= majority + 0.02
    print(f"\nbest: {best_name}  acc={best_acc:.3f}  "
          f"({'BEATS' if beats_majority else 'does NOT beat'} majority {majority:.3f} + 0.02 margin)")

    final = candidates[best_name]
    if best_name == "logreg":
        final = CalibratedClassifierCV(final, method="sigmoid", cv=5)
    final.fit(X, y)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": final,
            "feature_names": list(FEATURE_NAMES),
            "threshold": float(best_thr),
            "model_name": best_name,
            "beats_majority": bool(beats_majority),
            "majority_baseline_acc": float(majority),
            "cv": results,
            "n_train": n,
            "trained_on": "math-robust-final.csv OVERALL (qwen3-8b:low, Hendrycks MATH)",
        },
        OUT,
    )
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    print(json.dumps(results, indent=2))
    if not beats_majority:
        print("\n*** classifier does not clear the bar — solution.py will ship all-False. ***")


def _safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    try:
        return roc_auc_score(y, p)
    except ValueError:
        return float("nan")


if __name__ == "__main__":
    main()
