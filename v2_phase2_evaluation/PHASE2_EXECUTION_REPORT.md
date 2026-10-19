# Phase 2 execution report

**Status: protocol and isolated utilities prepared; model training not performed.**

## Actual dataset execution

The split utility ran against the exact dataset and schema files in the V1 release archive.

- Source: `Network_Data/phisingData.csv`
- SHA-256: `e20bf913b6c2e669f4e875e5d223be2542cc7c79a7b62d284ae9d9627c3d9678`
- Rows: 11,055; model input features: 30; target: `Result`
- Target mapping: raw `-1 → 0 (phishing)`, raw `1 → 1 (legitimate)`
- Encoded feature value domain checked: `{-1, 0, 1}`
- Outer split: 8,843 training rows / 2,212 final holdout rows
- Feature groups shared across outer train/holdout: 0
- Inner group-aware CV: five folds inside outer training only
- Training rows assigned to exactly one validation fold: 8,843 / 8,843
- Shared groups across any inner fold's train/validation: 0
- Conflicting-label feature-vector groups retained: 64 groups / 357 rows

The split results align with the Phase 1 seed-42 group-aware split (8,843/2,212 rows). Group IDs in saved manifests are stable SHA-256 hashes of the 30-feature vectors; the splitter uses first-occurrence factorized group codes to preserve Phase 1 fold ordering for the same source row order.

## Executed tests

- `python -m py_compile v2_phase2_evaluation/evaluation_protocol.py v2_phase2_evaluation/build_evaluation_plan.py v2_phase2_evaluation/tests/test_evaluation_protocol.py` — passed.
- `python -m unittest discover -s v2_phase2_evaluation/tests -v` from project root — 13 tests passed.
- `python -m unittest discover -s tests -v` from inside `v2_phase2_evaluation` — 13 tests passed.
- `python v2_phase2_evaluation/build_evaluation_plan.py` — passed against the actual V1 UCI CSV and schema; integrity verification reports PASS.
- Generated split manifests/reports were rebuilt twice in the same environment; SHA-256 output files were identical across the two runs.
- Static scan of the Phase 2 package found no estimator `.fit()` calls or model-selection/training code.

## Environment used for execution

- Python 3.13.5
- pandas 2.2.3
- NumPy 2.3.5
- scikit-learn 1.8.0
- SciPy 1.17.0

The output metadata records the dataset/schema/split hashes and installed versions. The user's local Conda environment must rerun the utility/tests before Phase 3; split IDs can differ across scikit-learn versions. No estimator was fitted and no model performance metric was computed.

## V1 preservation

No V1 application source file, model artifact, preprocessor artifact, training pipeline, GitHub branch, or Render configuration was changed by this implementation. Work was created in a separate temporary copy for packaging. The actual local Git tree on the user's PC cannot be inspected by this execution environment; see `GIT_BASELINE_CHECK.md` before creating `v2-development`.
