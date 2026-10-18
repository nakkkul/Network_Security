# Phase 6 — Code-only, leakage-safe evaluation pipeline

**Current stage:** implemented as code and read-only checks only. No estimator was fitted, no predictions were generated, and the reserved holdout was not scored while preparing or testing this package.

## Safe command (read-only)

From the project root, with the V2 environment activated:

```cmd
python -m v2_phase6_evaluation.preflight
```

The command verifies the pinned SHA-256 hashes of the canonical CSV, schema, split manifest, and inner-CV assignment; checks source-row mapping and target encoding; recomputes feature-group IDs; and checks outer/inner group separation. It prints a summary to the terminal and writes no files.

If it fails, stop. Do not regenerate the manifests, edit `protocol_metadata.json`, or continue with selection until the mismatch is investigated.

## Selection design (not to be run in Phase 6)

`selection.py` builds `GridSearchCV` objects but does not execute them during import or preflight. The future selection runner:

- accepts only the 8,843 outer-training rows in its model-selection interface;
- uses the exact five validation-fold assignments in `inner_cv_assignment.csv`;
- keeps exact feature groups together in each fold;
- wraps `KNNImputer(n_neighbors=3, weights="uniform")` and each classifier in a scikit-learn `Pipeline`, so preprocessing is fitted only within each fold's training data;
- selects with phishing-positive F1 (`pos_label=0`, `zero_division=0`), with `refit=False`;
- writes only a new `evaluation_artifacts/phase6_cv_selection_<run_id>/selection_report.json` after an explicitly authorized future run;
- does not fit a final estimator, save model artifacts, access holdout predictions/scores, or write to `final_model/`.

The selection runner is disabled unless all three checks are satisfied: `allow_fitting=True`, the explicit confirmation argument equals the authorization token in `selection.py`, and environment variable `NETWORK_SECURITY_PHASE6_ALLOW_MODEL_FIT` contains that same token. **Do not set these during Phase 6.** They are a tripwire for a separate future approval, not permission granted by this code-only phase.

## Tests

The Phase 6 tests are designed not to call `GridSearchCV.fit`, `Estimator.fit`, `predict`, or scoring APIs. They validate the read-only protocol checks, train-only row exposure, precomputed CV fold mapping, pipeline construction, class-0 scorer configuration, and the default-disabled fit guard.

```cmd
python -m unittest tests.test_phase6_evaluation_protocol -v
python -m unittest discover -s tests -v
python -m unittest discover -s v2_phase2_evaluation\tests -v
```

## Scope and restrictions

- No changes to `app.py`, the legacy MongoDB pipeline, prediction endpoint, `final_model/`, V1 branches, S3 settings, existing datasets, schema, or Phase 2 manifests.
- No pickle files are loaded.
- No training or holdout evaluation is performed by the preflight or tests.
- No commit, push, merge, or deployment is authorized by this patch.
