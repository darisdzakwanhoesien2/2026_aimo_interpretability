# Technical report: final outline (replaces the plan-stage `outline.md`)

**AIMO Interpretability Challenge 2026.** Due 2026-11-15 (two weeks after the 2026-11-01 deadline). Required for award consideration. Also eligible for the writeup prizes, whose categories include negative results and efficient methods. Format is not specified by the organizers; assume a short workshop paper (4 to 6 pages plus appendix).
Drafted 2026-10-31. Every number below was measured on 2026-10-31 and has a file behind it; anything marked TODO has not been run.

## 0. What the report can honestly claim

The plan-stage outline assumed model internals, generated-variant consistency and gated inference. None of that was built: the development machine has 8 GiB of RAM and no GPU, so no competition model was ever run. What exists is a surface-text classifier that ships as an all-False predictor. The report therefore has one primary claim, and the other three candidate claims are dropped:

> **Primary claim (negative result).** Cheap text-only features carry a small but statistically real signal about whether a model's reasoning survives perturbation (AUC 0.65 to 0.66, permutation p about 0.004), but not enough to beat the 74.5% majority-class baseline in accuracy. Under honest threshold selection, accuracy falls to 0.70 to 0.71.

Dropped: "efficient method" (no timing comparison exists), "generalization study" (one model, one problem source), "actionable interpretability" (no internals touched).

## 1. Abstract (4 sentences)
Task, what we shipped (an all-False floor in both tracks), the headline numbers from section 6, and the one lesson: with 141 labelled problems from one model, a 2-point accuracy gain cannot be told from noise, so baselines must be reported with intervals.

## 2. Introduction
- Final-answer accuracy hides shortcut reasoning; the task asks for a per-(model, problem) robust or spurious label.
- Why this is hard for a small team: the model must be run offline inside the container, labels are about 25% positive, and the public labels cover one model on easy problems.
- Contributions, three bullets: (1) a reproducible, contract-tested minimal submission, (2) a measurement of how much surface text predicts robustness, with the selection bias of the usual CV-threshold protocol corrected, (3) a short list of what a labelled set this small can and cannot support.

## 3. Task and data
- Contract: `are_robust(model_id, problems) -> list[bool]`; one exception, wrong length or non-bool invalidates the whole batch; accuracy over all labels; coverage must be 1.0.
- Labels: robust iff `absolute_accuracy_decay == 0` (36 of 141 problems, 25.5%); all-False gives 105/141 = 0.7447.
- Public data: `math-robust-final.csv` OVERALL rows, 141 rows, 137 unique problems, one model (`qwen3-8b:low`), Hendrycks MATH. The private test is olympiad-level and covers four models.
- **Important:** the 137-case `val-sample` used by the local harness is the same set of problems as the training CSV (exact text overlap 137 of 137). The local score of 0.7372 (101/137) therefore says nothing about generalization of any trained classifier; it only confirms the packaging.
- Figures: class balance, decay histogram, problem length against label.

## 4. Method
- **4.1 Features.** 22 surface features from `src/features.py` (length, digit and LaTeX counts, keyword flags, sentence statistics). No model internals, no generation.
- **4.2 Classifiers.** Balanced logistic regression and a depth-3 gradient-boosted classifier; threshold tuned for accuracy.
- **4.3 Ship rule.** The predictor activates only if `beats_majority` is true (CV accuracy at least majority + 0.02). It is false, so the shipped predictor returns False for every problem.
- **4.4 Packaging.** `solution.py` catches every exception and returns `[False]*n`; artifacts pinned to scikit-learn 1.8.0; both tracks built from one directory (the small track adds an empty `small.txt`).

## 5. Experiments (all runnable without a GPU)

| # | Experiment | State | Evidence |
|---|---|---|---|
| E1 | All-False baseline with interval | done | 0.7447, bootstrap 95% CI [0.674, 0.816] (`experiments/tier0_stats.json`) |
| E2 | Repeated 5-fold CV with the threshold chosen inside each training fold, 20 seeds | done | logistic regression 0.702 (sd 0.018 over seeds), gradient boosting 0.710 (sd 0.018) |
| E3 | The shipped artifact's own CV number | done | logistic regression 0.716 and gradient boosting 0.7305 at a threshold chosen on the same predictions; 1 to 2 points more optimistic than E2 |
| E4 | Ranking signal: AUC with bootstrap interval and permutation test | done | logistic regression 0.650 [0.545, 0.750], p = 0.005; gradient boosting 0.664 [0.565, 0.753], p = 0.003 |
| E5 | Local harness on the bundled sample, both tracks | done | accuracy 0.7372, coverage 1.0, 0 invalid (`kit/BUILD_RECORD.md`); not independent, see section 3 |
| E6 | Feature-group ablation: length, digits and LaTeX, keywords, sentence statistics | TODO, about 30 minutes | which group carries the AUC |
| E7 | Minimum detectable gain: with 141 rows and 36 positives, how large must a true accuracy gain be to be seen | TODO, simulation | table of gain against power |
| E8 | Official baselines cited, not reproduced: uncertainty profiling, grouped OOF balanced accuracy 0.626, plain accuracy 0.688 (organizers' artifact, from the task docs) | cite only | state clearly that these were not re-run here |
| E9 | Final private-leaderboard result, both tracks | after 2026-11-15 notice | official accuracy and rank, filled in when known |

## 6. Results and discussion
- Table 1: all-False, shipped (identical to all-False), logistic regression, gradient boosting, with intervals, and the official uncertainty baseline cited.
- The ordering of E2 against E1: both classifiers sit below the baseline's point estimate but inside its interval, so the honest statement is "not better", not "worse".
- E4 against E2: a nonzero AUC with no accuracy gain is expected at 25% prevalence; the threshold, not the signal, is the limit.
- E3: show the optimism of the usual protocol on this data.
- What would change the conclusion: per-problem hidden states or generation uncertainty from the actual models, which this work did not reach.

## 7. Limitations
- One model, one source, 141 rows; intervals are wide.
- Labels come from a single offline perturbation experiment by one generator, not from ground-truth robustness.
- No test of the private distribution (olympiad problems) or of the other three models.
- Training and local validation problems coincide.

## 8. Reproducibility
- Repo branch `feat/simulacrabench-aimo`, commit `a1e3567` for the submission; zips and hashes in `kit/BUILD_RECORD.md` (main `c3b31407…`, small `ce6e6bd9…`).
- Environment: Python 3.12, scikit-learn 1.8.0, joblib 1.5.3, numpy 2.5.4 for the checks.
- Commands: `experiments/tier0_stats.py --csv <math-robust-final.csv>`; `kit/scripts/build.py solutions [--small]`; `kit/scripts/run_local.py`.
- Disclosure: LLM coding assistant used for code and drafting; no external data beyond the organizers' files.

## 9. Peer-review commitment
Up to 3 submissions between 2026-11-15 and 11-30; keep notes in `report/reviews/`.

## Calendar
| By | Task |
|---|---|
| 2026-11-01 | submit both zips; save confirmations |
| 2026-11-06 | E6 and E7 done and added to `experiments/` |
| 2026-11-10 | full draft |
| 2026-11-13 | verification pass (Claude re-runs E1 to E7 and checks every number against its file), then submit by 11-15 |

## Checklist before submission
- [ ] Every number in the text equals the value in a file under `experiments/` or `kit/`
- [ ] The non-independence of the local validation set is stated
- [ ] Official baselines are labelled "cited", not "reproduced"
- [ ] Both submission IDs and the final rank filled in, or the section says results were not available
- [ ] LLM-assistant disclosure included
