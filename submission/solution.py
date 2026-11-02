"""AIMO Interpretability Challenge — submission entry point.

Contract (ingestion.py): the harness groups cases by model and calls
    are_robust(model_id: str, problems: list[str]) -> list[bool]
once per model. Any exception, wrong length, or non-bool element invalidates that
model's ENTIRE batch. So this function must never raise and must always return a
list[bool] of exactly len(problems).

Current strategy — Tier 0:
  Surface text features (no model internals — see note) give CV AUC ~0.65 on the
  public data but do NOT beat the ~0.745 majority-class accuracy. So unless a
  bundled artifact is marked `beats_majority`, we return the majority class:
  False ("spurious") for every problem. That is a legitimate strong baseline here.

  Why no internals: the intended features (hidden-state probes, generation
  uncertainty, self-consistency under functional variants) require running the
  cached competition models. They were not reproducible on the development
  machine (8 GB RAM). The extraction code is specced in ../plan.md; this file is
  the packaging + contract shell it will plug into.
"""

from __future__ import annotations

_PREDICTOR = None


def _get_predictor():
    global _PREDICTOR
    if _PREDICTOR is None:
        try:
            from robustness_predictor import RobustnessPredictor

            _PREDICTOR = RobustnessPredictor()
        except Exception:
            _PREDICTOR = False  # sentinel: never try again
    return _PREDICTOR or None


def are_robust(model_id: str, problems: list[str]) -> list[bool]:
    n = len(problems)
    try:
        predictor = _get_predictor()
        if predictor is None:
            return [False] * n
        out = predictor.predict(str(model_id), [str(p) for p in problems])
        if not isinstance(out, list) or len(out) != n:
            return [False] * n
        return [bool(x) for x in out]
    except Exception:
        return [False] * n
