# AIMO Interpretability Challenge 2026 — submission

Open-source implementation for the **AIMO Interpretability Challenge @
NeurIPS 2026** (Codabench: https://www.codabench.org/competitions/16180/),
published to satisfy the organizers' open-source requirement for award
consideration. This is a snapshot extracted from a private monorepo (see
"Provenance" below), not a live mirror — development continues elsewhere.

Task: given a math problem and the name of one of four frontier reasoning
models, predict whether that model's answer is invariant to five families of
deductive-chain-preserving perturbation (**robust**) or depends on a brittle
shortcut (**spurious**), without access to the model's internals or the
perturbation type, and without a network connection.

## What's here

- **[`report/report.tex`](report/report.tex) / [`report/report.pdf`](report/report.pdf)** — the technical report (NeurIPS 2026 format). Start here.
- **[`submission/`](submission/)** — the code that was actually zipped and uploaded to Codabench: `solution.py`, `features.py`, `robustness_predictor.py`, `artifacts/tier0_classifier.joblib`.
- **[`experiments/`](experiments/)** — `tier0_stats.py`/`.json` (bootstrapped baseline + nested-CV accuracy + AUC with a permutation test) and `ablation_power.py`/`.json` (feature-group ablation).
- **[`kit/BUILD_RECORD.md`](kit/BUILD_RECORD.md)** — SHA-256 of the exact zips uploaded, build commands, and the checks run before upload.
- **[`documentation.md`](documentation.md)** — the task spec, interface, and compute limits, verified against the organizers' repos.
- **[`literature_review.md`](literature_review.md)**, **[`plan.md`](plan.md)**, **[`report/outline_final.md`](report/outline_final.md)** — research grounding, approach, and the report's own planning notes.

## Result

Both tracks shipped the all-`False` baseline (the bundled surface-text
classifier was gated off because it did not beat the baseline under
cross-validation — see the report, Section 3). On the Codabench competition
phase this scored **0.48 (Main) / 0.47 (Small)**, well below the 0.745 floor
measured on the public label file; the full negative-result writeup,
including why that floor didn't transfer, is in the report.

## Reproduce

```bash
python -m venv .venv && . .venv/bin/activate
pip install scikit-learn==1.8.0 joblib==1.5.3 numpy==2.5.4
python experiments/tier0_stats.py --csv <math-robust-final.csv>   # baseline + CV stats
python experiments/ablation_power.py                               # feature-group ablation
```

Rebuilding the submission zips requires the organizers' getting-started kit
(https://github.com/aimo-interp/getting-started) — see `kit/BUILD_RECORD.md`
for the exact steps and the hashes to check the rebuild against.

Compile the report: `cd report && pdflatex report.tex && pdflatex report.tex`
(needs `report/neurips_2026.sty`, included here).

## Provenance

Extracted as a fresh snapshot (no commit history carried over) from
`darisdzakwanhoesien2/research_collaboration`, branch
`feat/simulacrabench-aimo`, folder `shared_task_aimo_interp_2026`, at commit
`aa09ee1f2923ca7f233961480239384591ad77eb`. That repository is private and
holds several unrelated research projects; this repository contains only
what's relevant to this challenge entry.
