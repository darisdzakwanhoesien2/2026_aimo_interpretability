"""Loads the bundled Tier-0 artifact and scores problems.

The artifact only overrides the all-False default when it was trained to a CV
accuracy that clears the majority-class baseline (`beats_majority` flag set by
src/train_classifier.py). Today it is not set, so `predict` returns all-False.
"""

from __future__ import annotations

import pathlib

import joblib
import numpy as np

from features import FEATURE_NAMES, feature_matrix

_ARTIFACT = pathlib.Path(__file__).with_name("artifacts") / "tier0_classifier.joblib"


class RobustnessPredictor:
    def __init__(self) -> None:
        self.active = False
        self.pipeline = None
        self.threshold = 0.5
        if _ARTIFACT.is_file():
            bundle = joblib.load(_ARTIFACT)
            if bundle.get("beats_majority") and list(bundle.get("feature_names", ())) == list(
                FEATURE_NAMES
            ):
                self.pipeline = bundle["pipeline"]
                self.threshold = float(bundle["threshold"])
                self.active = True

    def predict(self, model_id: str, problems: list[str]) -> list[bool]:
        if not self.active or self.pipeline is None:
            return [False] * len(problems)
        X = feature_matrix(problems, [model_id] * len(problems))
        proba = self.pipeline.predict_proba(X)[:, 1]
        return [bool(p >= self.threshold) for p in np.asarray(proba)]
