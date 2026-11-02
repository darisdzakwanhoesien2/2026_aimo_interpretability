"""Feature extraction from (problem_text, model_id) — the only inputs `are_robust` receives.

Deliberately lightweight and dependency-free (stdlib + numpy): the evaluation
container ships sklearn but the point of this module is that it also runs during
offline training on the public CSVs with no model, no torch, no network.

These are *surface* features. They cannot see the model's reasoning, so this is a
weak signal by construction (see plan.md §2 — the real levers are behavioural /
internal). It exists to (a) lock the submission contract and packaging, and
(b) test whether any cheap signal clears the ~0.74 majority-class baseline.
"""

from __future__ import annotations

import re

import numpy as np

# Ordered — the trained artifact stores this so inference matches training exactly.
FEATURE_NAMES: tuple[str, ...] = (
    "char_len",
    "word_len",
    "log_char_len",
    "n_digits",
    "digit_frac",
    "n_numbers",
    "n_latex_cmd",
    "n_dollar",
    "latex_frac",
    "n_operators",
    "n_equals",
    "has_prove",
    "has_find",
    "has_compute",
    "has_how_many",
    "has_probability",
    "has_percent",
    "has_fraction_bar",
    "n_sentences",
    "mean_sentence_len",
    "uppercase_frac",
    "n_variables_single",
)

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_LATEX_RE = re.compile(r"\\[a-zA-Z]+")
_SENT_RE = re.compile(r"[.!?]+")
_SINGLE_VAR_RE = re.compile(r"(?<![A-Za-z])[a-zA-Z](?![A-Za-z])")


def extract_features(problem: str, model_id: str = "") -> dict[str, float]:
    """Return the ordered feature dict for one problem. `model_id` reserved for
    a future per-model term (only one model in the public data, so unused here)."""
    text = problem or ""
    n_char = len(text)
    words = text.split()
    n_word = len(words)
    digits = sum(c.isdigit() for c in text)
    numbers = _NUMBER_RE.findall(text)
    latex = _LATEX_RE.findall(text)
    sentences = [s for s in _SENT_RE.split(text) if s.strip()]
    low = text.lower()

    feats = {
        "char_len": float(n_char),
        "word_len": float(n_word),
        "log_char_len": float(np.log1p(n_char)),
        "n_digits": float(digits),
        "digit_frac": digits / n_char if n_char else 0.0,
        "n_numbers": float(len(numbers)),
        "n_latex_cmd": float(len(latex)),
        "n_dollar": float(text.count("$")),
        "latex_frac": len(latex) / n_word if n_word else 0.0,
        "n_operators": float(sum(text.count(op) for op in ("+", "-", "*", "/", "^"))),
        "n_equals": float(text.count("=")),
        "has_prove": float("prove" in low or "show that" in low),
        "has_find": float("find" in low),
        "has_compute": float("compute" in low or "calculate" in low or "evaluate" in low),
        "has_how_many": float("how many" in low or "number of" in low),
        "has_probability": float("probab" in low),
        "has_percent": float("percent" in low or "%" in text),
        "has_fraction_bar": float("\\frac" in text or "\\dfrac" in text),
        "n_sentences": float(len(sentences)),
        "mean_sentence_len": (n_word / len(sentences)) if sentences else float(n_word),
        "uppercase_frac": sum(c.isupper() for c in text) / n_char if n_char else 0.0,
        "n_variables_single": float(len(set(_SINGLE_VAR_RE.findall(text)))),
    }
    return {k: feats[k] for k in FEATURE_NAMES}


def feature_matrix(problems: list[str], model_ids: list[str] | None = None) -> np.ndarray:
    model_ids = model_ids or [""] * len(problems)
    rows = [extract_features(p, m) for p, m in zip(problems, model_ids)]
    return np.asarray([[r[k] for k in FEATURE_NAMES] for r in rows], dtype=np.float64)
