# V2 evaluation protocol — decision record

## Status

Phase 2 protocol implementation; **no classifier fitting or hyperparameter search is permitted in this phase**.

## Dataset contract

- Source: `Network_Data/phisingData.csv`.
- Schema: `data_schema/schema.yaml`.
- Input features: the schema's ordered columns excluding `Result` (30 features).
- Raw target: `Result ∈ {-1, 1}`.
- Mapped target: `-1 → 0 (phishing)`; `1 → 1 (legitimate)`.
- Primary analysis preserves all rows, duplicates, and conflicting labels.

## Group identity

The manifest stores a stable SHA-256 digest of canonical JSON containing every input feature in schema order, normalized as integers. The target is not included. For splitter ordering, the code uses `MultiIndex.from_frame(X).factorize(sort=False)` to align with Phase 1; that factorization is reproducible for the exact dataset hash and original row order. Identical feature vectors always share one group, including rows with different target labels.

## Outer split

Use `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` and reserve fold index 0 as the final holdout. The remaining four folds form the training partition. The actual manifest is written and hashed. The test partition is not used for model/hyperparameter selection.

This is a deterministic, approximately 20% group-aware holdout for the given dataset row order and installed scikit-learn version; it is not promised to have an exact row count across all library versions. Regenerate and review the manifest after version changes.

## Inner cross-validation

Use `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=43)` on outer-training rows only. Group IDs are supplied to the splitter. Each outer-training row belongs to exactly one validation fold assignment; for fold `k`, all other folds form the CV training partition. Tests verify group isolation for every fold.

## Conflict policy

There are 64 feature-vector groups with both mapped labels in the audited source snapshot. The primary analysis retains their rows and keeps each group wholly in a single partition/fold. It does not attempt to infer a corrected label. Phase 3 should add a sensitivity analysis excluding all rows from conflicting groups and compare it transparently with the primary all-row evaluation. That sensitivity result must not silently replace the primary result because excluding those groups changes the evaluated population.

## Phase 3 metric contract

Primary candidate selection metric: phishing F1 (`pos_label=0`). Secondary metrics:
- phishing precision/recall/F1;
- legitimate precision/recall/F1;
- macro F1;
- balanced accuracy and accuracy;
- confusion matrix with class order `[0,1]`;
- ROC-AUC and average precision for phishing where a valid class-0 probability/score is available.

For ROC-AUC/average precision, convert labels to a binary indicator where actual phishing is positive (`y_true == 0`) and supply a score where larger values mean more likely phishing. Call the PR summary `average precision` explicitly; do not claim it is necessarily a linearly interpolated trapezoidal PR-AUC.

## Acceptance criteria before Phase 3

1. Correct ordered 30 feature columns plus `Result` are present; no missing/non-finite values.
2. Target labels are exactly `{-1,1}` before mapping.
3. Source data and schema hashes are stored.
4. No feature-group hash overlaps across outer train/holdout.
5. Inner CV includes only outer-training rows.
6. No group crosses a CV validation/training boundary.
7. Every outer-training row is a validation row exactly once across the inner folds.
8. Phishing class semantics are explicitly tested.
9. Conflicting-label groups are retained and counted.
10. No estimator has been fitted in Phase 2.
