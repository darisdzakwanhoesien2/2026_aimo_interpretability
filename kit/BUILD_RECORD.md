# AIMO Interpretability: submission build record

Built and checked 2026-10-31 from branch `feat/simulacrabench-aimo` (commit `a1e3567`), folder `shared_task_aimo_interp_2026/submission/`.

## Artifacts (in `kit/dist/`)

| File | Track | SHA-256 |
|---|---|---|
| `solution-tier0.zip` | main | `c3b31407805f809a2498d0684fd7ef6b21475e041d3ba92a86fcc77cca59f3d8` |
| `solution-tier0-small.zip` | small (adds an empty `small.txt`) | `ce6e6bd981593e581bbd791cd8abf3f488ec192e743823a42cdfa0b9534761f3` |

Rebuilding twice gave the same hash. Contents match the branch files exactly (`solution.py` 295dd953219b, `robustness_predictor.py` 5483809f6497, `features.py` 1f84cecfd32f, `artifacts/tier0_classifier.joblib` f2eabfcac02a, first 12 hex digits). The older zips in `output/` have the same contents but different bytes (timestamps).

## Rebuild

```bash
cd shared_task_aimo_interp_2026/kit
python3.10 scripts/build.py solutions            # dist/solution-tier0.zip
python3.10 scripts/build.py solutions --small    # dist/solution-tier0-small.zip
```

`kit/` is the organizers' getting-started kit (scripts, components, tests, sample data) with `solutions/tier0/` copied from `submission/`. The organizers' `solutions/` examples (53 MB) were left out.

Only this file is committed from `kit/`. To rebuild, clone the organizers' getting-started repository (https://github.com/aimo-interp/getting-started), copy `submission/` into its `solutions/tier0/`, and run the two commands above from there.

## Checks run (Python 3.12, scikit-learn 1.8.0, joblib 1.5.3, numpy 2.5.4, matching the competition pins)

- `scripts/run_local.py` on the bundled validation sample, both tracks: accuracy 0.7372, coverage 1.0, invalid_predictions 0.
- `tests/test_ingestion_contract.py`: 5 tests passed.
- `are_robust` on empty input, one item, a 100,000-character item, control characters, 1,000 items, non-string items and a `None` model id: always a list of `bool` of the right length, never raises.

## What the submission is

An all-False prediction. The bundled classifier is flagged `beats_majority: False` (majority baseline 0.7447; best cross-validated accuracy 0.7305 with `hgb`, AUC 0.664, trained on 141 rows from one model), so the predictor stays inactive and returns False for every problem. It is a labelled floor, not a model that predicts robustness.

Note: the pickled artifact was written with scikit-learn 1.8.0 and numpy 2.x. It does not load under scikit-learn 1.7.2 or numpy 1.26 (the pipeline's default Python 3.10 environment); the solution catches that and returns all-False either way.
