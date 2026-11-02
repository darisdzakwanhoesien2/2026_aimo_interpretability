# AIMO Interpretability Challenge 2026 — project plan

Shared task: **AIMO Interpretability Challenge @ NeurIPS 2026** (competition
workshop Dec 11–12). Predict whether a frontier reasoning model's math reasoning
is **robust** or **spurious** on newly-published olympiad problems.

- Task spec & sources: [`documentation.md`](documentation.md)
- Research grounding: [`literature_review.md`](literature_review.md)
- **Final submission deadline: 2026-11-01** · Technical report: by 2026-11-15 · Peer review: Nov 15–30
- Today: **2026-09-07** → ~8 weeks to the deadline.

## 0. Status (2026-09-07)

**Submittable `solution.py` exists, validated through the real harness. It ships
the majority-class baseline (all-`False`) because no cheap signal beats it yet.**

| Thing | State |
|---|---|
| `submission/solution.py` + `robustness_predictor.py` + `features.py` | ✅ contract-correct: never raises, exact-length `list[bool]`, coverage 1.0 |
| Run through `getting-started/scripts/run_local.py` (main + small) on a 137-case fixture built from the public CSV | ✅ `accuracy 0.737, coverage 1.0, invalid_predictions 0` |
| `output/solution-tier0-{main,small}.zip` built + re-validated through the harness | ✅ |
| Tier-0 **surface-feature classifier** (`src/train_classifier.py`, 22 text features, HGB/logreg, grouped CV) | trained — **CV acc 0.716–0.730, AUC ~0.66 → does NOT beat the 0.745 majority baseline.** Artifact bundled but gated OFF (`beats_majority=False`). |
| val-sample (private), model internals, behavioural features | ❌ blocked — needs an HF token, and the dev machine (8 GB RAM / ~4 GB disk) cannot run the 7–120 B competition models |

**Takeaway (a real result for the report):** problem *surface* features carry a
little signal (AUC 0.66) but not enough to beat "always spurious" on accuracy.
The discriminating signal is behavioural / internal, exactly as `literature_review.md`
argues — and that needs the competition GPUs, not this laptop.

**Official repos read** (`examples/getting-started`, `examples/baselines`), paper
abstract + author list read, `documentation.md` + `literature_review.md` verified.
Report outline drafted (`report/outline.md`).

### What unblocks progress
- An HF token with `aimo-interp/val-sample` access → real labels, all 4 models.
- Compute that can run `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` / `Olmo-3-7B` (a
  Colab / Kaggle / cloud box) → extract the Tier-A uncertainty features (the 14
  from the official baseline + our additions) and retrain the classifier.
- Then the same `solution.py` shell picks up the artifact automatically once it
  clears the bar (`RobustnessPredictor` checks `beats_majority`).

**Resolved since the first draft:**
- `permutation_type` is **not** given at eval — only problem strings. The five families are `distract, rename, rephrase, typos, domain`.
- The label ≈ "robust ⟺ `absolute_accuracy_decay == 0`". Public data is **~25% robust / 75% spurious → a constant-`False` predictor scores ≈ 0.74**. That, not 0.5, is the bar.
- The model is **named, not passed** — `model_id` is a `name:effort` alias (e.g. `qwen3-8b:low` → `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`); we load the cached model.
- `are_robust` is called **once per model** with all that model's problems (harness batches by `model_id`). One exception ⇒ that whole model's batch scored invalid.

**Still gating the design:**
1. Full alias list + reasoning-effort for all 4 models.
2. Test-run size (# models × # problems) inside 3600 s → per-instance budget.
3. Per-model label balance on the private set (esp. `gpt-oss-120b`).
4. Whether hidden states are usable for the 8-way-sharded bf16 `gpt-oss-120b`.

## 1. Task in one screen

- **Interface:** `are_robust(model_id, problems: list[str]) -> list[bool]` in `solution.py`. Harness batches by model → one call per model, all its problems. Return native `bool`, exact length, order preserved. **Any exception / bad return ⇒ that model's whole batch invalid (= wrong + coverage < 1).** Wrap everything; fall back to `[False]*n`.
- `model_id` is a Codabench alias (`qwen3-8b:low`, …), resolve to the HF checkpoint ourselves. Only problem strings are provided — no IDs, no `permutation_type`, no network.
- **We run the model ourselves** in the offline container (cache at `/app/data/huggingface`, `local_files_only=True`); may use weights, activations, logits, generated CoT.
- **Label:** robust ⟺ model solves base + all perturbed variants with zero accuracy loss. Spurious otherwise. **~25/75 robust/spurious on public data.**
- **Metric:** accuracy = correct / *all labels* (so coverage must be 1.0). **Tiebreak:** mean runtime ×3.
- **Hard limit:** 3600 s/run; 8× RTX PRO 6000 (96 GB), 192 CPU. Preinstalled libs only (torch 2.12.1, transformers 5.13.0, sklearn 1.8.0, …) — vendor anything else.
- **Submission:** ZIP, `solution.py` at root (+ artifacts); root `small.txt` → Small track. One method → both tracks.

## 2. Strategy

Ship a **cheap, calibrated ensemble classifier** built from offline-trained
artifacts, and separately pursue a **mechanistic writeup** on the Small Models
Track for the $5,000 writeup pool. Rationale:

- **The bar is ≈ 0.74, not 0.5.** Constant-`False` already scores ~0.74 on public
  data; the official uncertainty baseline gets only 0.688 *plain* accuracy (0.626
  balanced). Winning = adding true-positive robust detections without flipping
  many spurious cases. So: regress `absolute_accuracy_decay`, then **tune the
  threshold directly on expected leaderboard accuracy under the real class prior**
  — do not default to 0.5, and measure every tier against the majority-class line.
- The leaderboard rewards accuracy then efficiency. A linear/GBM classifier over
  pre-extracted features is both accurate (matches the official baseline family)
  and near-instant at eval time — strong on the tiebreaker.
- No training at eval time: all model-internal feature extraction that *can* be
  precomputed for the *cached* models is precomputed and shipped; at eval we only
  run forward passes on the (unseen) problems + a small sklearn model.
- The writeup prizes explicitly reward generalization studies, actionable
  interpretability, and negative results — a mechanistic SAE/patching study on
  Olmo-3-7B or DeepSeek-R1-Qwen3-8B is a realistic second deliverable even if it
  doesn't top the leaderboard.

### Decision: submit the same ensemble to both tracks.

Main track uses all 4 models' features; Small track ZIP is identical + `small.txt`.
The Small-track mechanistic additions are for the report, not the scored run.

## 3. Features (per (model, problem) instance)

All extracted by us running the cached model on the problem. Three groups:

**A. Behavioral / uncertainty (model-agnostic, works even for `gpt-oss-120b`)**
- answer-token log-prob; mean/min token entropy over the CoT
- self-consistency: k∈{5,8} samples → answer-distribution entropy, top-1 share
- semantic entropy over the k samples (cluster by final answer / equivalence)
- CoT length, # of arithmetic steps, # of self-corrections ("wait", "actually")
- self-verification: ask the model P(True) on its own answer (Kadavath-style)
- **variant consistency** (the core signal): generate m deductive-chain-preserving
  variants of the problem (number swaps, entity renames, clause reorders — see §5),
  re-run, measure answer-agreement rate and reasoning-trace similarity. Low
  agreement under meaning-preserving edits ⇒ spurious.

**B. Probe features (7B–27B models; skip / degrade gracefully for 120B)**
- hidden state at the last problem token and at the answer token, layers {25%,50%,75%,last}
- linear-probe margin per layer (probe trained offline on train split), averaged
- optional: logit-lens answer rank/entropy trajectory across layers

**C. Static / cheap**
- which model (alias one-hot + reasoning-effort) — label rates differ by model
- problem surface stats (length, # quantities, topic bucket via keyword rules)
- ~~`permutation_type`~~ — confirmed **not** available at eval; dropped.

**Baseline-parity note:** groups A/B already reproduce the two official baselines
(14 uncertainty features; last-token per-layer hidden states). Ship those first as
Tier A+, then add the deltas above.

Classifier: gradient-boosted trees (sklearn `HistGradientBoostingClassifier`,
which is in the eval env) + logistic-regression fallback, calibrated
(`CalibratedClassifierCV`), trained on the released train split with
**grouped CV by problem** (never let variants of one problem span train/test folds)
and additionally checked with **leave-one-model-out CV** to estimate generalization
to unseen model behavior.

## 4. Approach tiers (build in this order)

| Tier | What | Needs internals? | Eval cost | Purpose |
|---|---|---|---|---|
| **A** | Uncertainty features (group A, no variants) → GBM | no | low | day-1 baseline, beat `[False]*n` |
| **A+** | + linear probe (group B) on 7–27B models | yes (small models) | low | match/beat official baseline |
| **B** | + variant-consistency (self-generated functional variants) | no | **high** | the discriminating signal |
| **B-gated** | run A+ first; only escalate to B for instances with |p−0.5| < τ | mixed | tuned | leaderboard submission (accuracy ≈ B, cost ≪ B) |
| **C** | SAE features (Small track) + activation patching original↔variant | yes | very high | **writeup only**, not scored |

Leaderboard submission = **B-gated**. Report = B-gated results + Tier-C study.

## 5. Functional-variant generation (Tier B core)

The label is about surviving deductive-chain-preserving perturbation, so we
reproduce that process offline and at eval:
- **Symbolic re-instantiation:** parse numeric quantities, resample within
  type-consistent ranges, recompute the gold answer with a solver/CAS where the
  problem structure allows; otherwise use answer-agnostic checks (consistency
  across variants rather than correctness).
- **Surface perturbations:** entity/name swaps, unit-preserving rephrases, clause
  reordering, irrelevant-but-plausible clause insertion (GSM-IC style).
- **Templates** kept in [`prompts/`](prompts/) (to be created). Validate a sample
  by hand — a broken variant (changed answer) poisons the consistency signal.
- At eval time, generation must fit the time budget → cap m (≈3–5) and only for
  gated instances.

Risk: variant generation with no internet and no strong CAS in the eval env.
Mitigation: vendor `sympy` (pure Python) into the ZIP; keep generation rule-based,
not model-based, where possible; precompute nothing problem-specific (problems are
unseen) but precompute and ship the generation code + templates.

## 6. Compute budget (3600 s / run)

The harness calls `are_robust` **once per model** with all that model's problems,
so we control batching — batch generation/forward passes across the 8 GPUs freely.
The 3600 s covers *all* models' problems in one run, not per model.

Worked example, **assume** test run = 4 models × 150 problems = 600 instances
(VERIFY — drives everything):
- Tier A+ forward pass + probe: ~2–4 s/instance → ~40 min for all 600. OK.
- Tier B full (k=8 samples + m=4 variants × k=3): ~60–120 s/instance → 10–20 h. **Infeasible for all.**
- **B-gated:** if ~20% of instances land near the boundary → 120 × 90 s ≈ 3 h. Still tight.
  → tighten τ, reduce k/m, or batch aggressively across the 8 GPUs (vLLM-style
  batched generation; check what's installed — `accelerate` is, vLLM may not be → vendor or use HF `generate` with batching).
- **Contingency:** if the run is bigger than assumed, fall back to Tier A+ only
  (fast, safe) and keep B-gated as the "if it fits" submission.

Action: measure real per-instance latency on the val sample in week 1 and
back-solve the achievable k, m, τ.

## 7. Repo layout

```
shared_task_aimo_interp_2026/
├── plan.md  documentation.md  literature_review.md  README.md
├── src/
│   ├── data.py          # load val-sample / train / val, schema checks
│   ├── models.py        # offline model loading, batched generation, hidden-state hooks
│   ├── features/
│   │   ├── uncertainty.py   # group A
│   │   ├── probe.py         # group B (train + inference)
│   │   └── variants.py      # group A "variant consistency" + §5 generation
│   ├── classifier.py    # train GBM/LR, calibrate, grouped/LOMO CV
│   ├── gate.py          # B-gated escalation logic
│   └── solution_build.py # assemble the Codabench ZIP
├── prompts/             # variant-generation + self-verification templates
├── scripts/
│   ├── fetch_data.sh
│   ├── extract_features.sh   # long-running, checkpointed
│   └── build_submission.sh
├── experiments/         # runs/, metrics.csv, CV results  (gitignored contents)
├── data/                # val-sample/, train/, val/  (gitignored)
├── output/              # submission_main.zip, submission_small.zip
├── report/              # outline.md (done), draft, results.csv, reviews/
├── examples/            # cloned getting-started + baselines (gitignored)
└── literature/          # downloaded papers
```

Mirror the checkpoint-everything discipline from the DocSem project: every
feature-extraction pass writes per-instance JSONL to `experiments/runs/` and
resumes on restart; expect the process to die.

## 8. Immediate next actions (week of 2026-09-07)

1. [x] `git clone` getting-started + baselines into `examples/`; read harness + both example solutions. *(done 2026-09-07)*
2. [ ] Join Discord, read pins + `#announcements`; skim GitHub issues; **ask for the full model-alias list and expected test-run size**.
3. [ ] Register on Codabench (comp 16180); get an HF token with `aimo-interp/val-sample` access.
4. [ ] `uv sync` in `examples/getting-started`; `import_hf_dataset.py`; run the `always-true` and `trained-probe` (artifact-less → all-False) solutions locally to confirm the harness + get the majority-class number on val-sample.
5. [ ] `src/data.py` + EDA: per-model label balance, decay distribution, base_accuracy==0 handling, problem-length/topic vs label.
6. [ ] Reproduce **both** official baselines end to end (probe + uncertainty) on `qwen3-8b:low` → real numbers, harness proven.
7. [ ] Submit an `always-false` ZIP to *both* tracks — validates packaging + gives the true majority baseline on the private val.
8. [ ] Measure per-model wall-clock for one forward pass + one greedy generation → fill §6 with real numbers.
9. [ ] Decide hidden-state feasibility for sharded `gpt-oss-120b`.

## 9. Milestones

| By | Deliverable |
|---|---|
| **Sep 14** | Data understood; baseline reproduced; Tier A submitted; latency measured; §6 numbers real |
| **Sep 28** | Tier A+ (probe ensemble) beating baseline on grouped + LOMO CV |
| **Oct 12** | Tier B variant-consistency working; B-gated pipeline fits the time budget on a full dry run |
| **Oct 19** | Small-track Tier-C study underway (SAE/patching on one <10B model) |
| **Oct 26** | Submission frozen; 3× dry-run timing; both ZIPs built + Codabench-validated |
| **Oct 12** | Primary report claim chosen (`report/outline.md` §0) |
| **Nov 1** | Final submission uploaded (both tracks) |
| **Nov 15** | Technical report submitted (draft from `report/outline.md`) |
| **Nov 15–30** | Peer review 3 submissions |

## 10. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| **Can't beat the ~0.74 majority baseline** | high | threshold-tune on leaderboard-accuracy not 0.5; treat "always-False + a few high-precision robust flips" as the floor design; a method that can't clear it on CV doesn't ship |
| Per-model balance on private test unknown (maybe not 25/75) | med | don't hard-code the prior; recalibrate threshold per `model_id`; LOMO CV |
| Test run too large for Tier B | high | B-gated + Tier-A-only fallback, both pre-built |
| `gpt-oss-120b` internals inaccessible | med | group A + C features are model-agnostic; probe only weights small models |
| Variant generation wrong (changed answer) | high | rule-based + sympy check + hand-validated sample; consistency-not-correctness metric |
| Vendored dep missing at eval | high | test the ZIP in a matched offline Docker image before every submission |
| Label definition misread | high | reproduce baseline first; confirm on Discord; EDA on val-sample labels |
| Overfitting to the 4 known models | med | leave-one-model-out CV is a gating metric, not just a diagnostic |
| Non-`bool` / exception → invalid | med | `solution.py` wraps everything in try/except → default `False`; unit-test the contract |

## 11. Working style (carried from DocSem project)

- Checkpoint per instance to disk; resume on restart; never trust `/tmp`.
- Every bug found by *running* against real data, not by inspection.
- One long-running supervisor script per expensive pass (disk + resource guards).
- Keep a dated status log at the top of this file (§0) as things progress.
