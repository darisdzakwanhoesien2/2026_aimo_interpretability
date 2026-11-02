# Technical report — outline

**AIMO Interpretability Challenge 2026.** Report is **required for award
consideration** (Main or Small track) and is separately eligible for the
**$5,000 writeup prizes** (generalization studies, actionable interpretability,
negative results, efficient methods). Due **within two weeks of 2026-11-01**
(i.e. by **2026-11-15**). Format: not yet specified by organizers — assume a
short workshop paper (~4–8 pages + appendix), NeurIPS style, arXiv-able.

Write it **as the experiments happen**, not after. Every §5 experiment below maps
to a plan.md tier and should dump a row into `report/results.csv` when it runs.

---

## 0. Target framing — pick ONE primary claim

The report is stronger with a single thesis. Candidates, decide by ~Oct 12:

- **A "Efficient method"** — "a single-forward-pass feature ensemble matches
  multi-sample behavioral methods at Nx lower cost." Fits the tiebreaker + an
  efficiency writeup prize. Safe; likely true.
- **B "Generalization study"** — "which robustness signals transfer across models
  (leave-one-model-out) and which are model-specific." Directly answers the
  competition's stated scientific question. Needs all 4 models' data.
- **C "Actionable interpretability"** — "a probe direction / SAE feature that
  tracks shortcut reliance, validated causally by activation patching between a
  problem and its functional variant." Highest risk, highest writeup value,
  Small-track only.
- **D "Negative result"** — "internal representations add little over behavioral
  uncertainty for predicting perturbation robustness, and here's the controlled
  evidence." Valid prize category; only write this if the CV genuinely shows it.

Primary claim: **TBD**. Secondary claims get one subsection each.

---

## 1. Abstract
Problem, our method in one sentence, headline number (accuracy vs the ~0.74
majority baseline + the official baselines), and the one scientific takeaway.

## 2. Introduction
- Final-answer accuracy hides shortcut reasoning (cite GSM-Symbolic, GSM-IC, the challenge paper [arXiv:2607.13899]).
- Task: predict, from problem + model, whether the model's answer survives deductive-chain-preserving perturbation.
- What makes it hard: `permutation_type` hidden at eval; ~25/75 class imbalance; 3600 s offline budget; `gpt-oss-120b` only as bf16 across 8 GPUs.
- Contributions (3 bullets, mirror the primary claim).

## 3. Task and data
- Formal setup: `are_robust(model_id, problems) -> list[bool]`; alias resolution; per-model batching; scoring (accuracy over all labels, coverage, runtime tiebreak).
- Label derivation: robust ⟺ `absolute_accuracy_decay == 0`; the five perturbation families; variants authored by `gpt-oss-120b`.
- Datasets used: `val-sample`, `augmented-sample-math-agg`, `aimo-interp-challenge-sample-full`, the checked-in `math-robust-*.csv`. Public labels = `qwen3-8b:low` on MATH; private test = olympiad problems, 4 models.
- EDA figures: class balance per model, decay histogram, base-accuracy vs label, problem length/topic vs label.

## 4. Method
- **4.1 Features.** The 14 uncertainty features (reproduced) + temporal trace features (uncertainty slope/linearity/spikes, cf. [arXiv:2605.07776]) + self-consistency spread + P(True) + per-layer hidden-state summaries + self-generated variant-consistency.
- **4.2 Variant generation** (offline + at eval): rule-based symbolic re-instantiation + surface perturbations mirroring the five families; sympy-checked; templates in `prompts/`.
- **4.3 Classifier.** Regress `absolute_accuracy_decay` (HistGBR / RF), `StratifiedGroupKFold` grouped by problem, **threshold tuned on expected leaderboard accuracy under the class prior**, per-model recalibration.
- **4.4 B-gated inference.** Cheap features for all; escalate to variant-consistency only for near-threshold instances. Time budgeting.
- **4.5 Packaging.** `solution.py` total-catch → `[False]*n`; vendored deps; version-matched artifacts; Docker parity test.

## 5. Experiments
| # | Experiment | Plan tier | Metric(s) | Feeds claim |
|---|---|---|---|---|
| E1 | Majority-class + `always-false` on val-sample | — | acc, coverage | baseline line |
| E2 | Reproduce official probe baseline | A+ | acc, balanced acc, MAE | baseline parity |
| E3 | Reproduce official uncertainty baseline (target 0.63 bal acc) | A+ | acc, balanced acc, MAE | baseline parity |
| E4 | Our feature ensemble, grouped CV | A+ | acc vs 0.74, balanced acc, AUC | primary |
| E5 | Ablations: drop each feature group | A+ | Δacc | primary / D |
| E6 | Leave-one-model-out CV | A+ | per-model acc, transfer gap | B |
| E7 | Variant-consistency signal alone + added | B | Δ balanced acc | primary |
| E8 | B-gated vs full-B vs A+-only: accuracy × runtime | B-gated | acc, mean runtime ×3 | A (efficiency) |
| E9 | Threshold sensitivity / per-model recalibration | A+ | acc curve | primary |
| E10 | (Small) SAE feature or probe direction + activation patching | C | causal Δ, qualitative | C |
| E11 | Final: private-leaderboard result, both tracks | — | official acc, rank | headline |

## 6. Results & discussion
- Table: baselines vs ours, per track, with runtime.
- Which signals matter and which transfer (E5/E6).
- Failure analysis: which problems/models we get wrong; is it a family we can't simulate.
- Honest negative results.

## 7. Limitations
- Label = one offline perturbation experiment's outcome, not ground-truth "robustness".
- Public labels are one model / one problem source; olympiad test may behave differently.
- `gpt-oss-120b` internals limited by sharding → behavioral-only for the biggest model.

## 8. Reproducibility
Repo, pinned dataset revisions, Docker image, seeds, artifact provenance, compute used.

## 9. Peer-review commitment
Note the Nov 15–30 obligation (review ≤3 submissions); keep review notes in `report/reviews/`.

---

## Checklist before submission
- [ ] Primary claim chosen and supported by ≥2 experiments
- [ ] Every number reproducible from a script in `experiments/`
- [ ] Baselines reproduced, not just cited
- [ ] Runtime reported (it's the tiebreaker and an award category)
- [ ] Negative results included where found
- [ ] Disclosure: external data, LLM coding assistants
- [ ] arXiv + Codabench submission-ID cross-reference
