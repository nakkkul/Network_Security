# V2 Phase 2 — Evaluation Protocol and Split Plan

**No classifier was fitted or tuned to create this plan.** The files record a fixed, group-aware outer holdout and group-aware inner CV folds for later Phase 3 model selection.

## Dataset identity

- SHA-256: `e20bf913b6c2e669f4e875e5d223be2542cc7c79a7b62d284ae9d9627c3d9678`
- Rows: 11,055
- Input features: 30
- Target: `Result`; raw `-1 → 0 (phishing)`, raw `1 → 1 (legitimate)`
- Allowed encoded feature values: `[-1, 0, 1]`
- Python/pandas/NumPy/scikit-learn/SciPy: 3.11.0 / 3.0.6 / 2.4.6 / 1.9.1 / 1.17.1

## Outer holdout (reserved before model selection)

- Method: `StratifiedGroupKFold; first fold reserved as final holdout`
- Seed: 42; folds: 5; reserved fold index: 0
- Training rows: 8,843; phishing share: 44.31%; feature groups: 4,634
- Holdout rows: 2,212; phishing share: 44.30%; feature groups: 1,151
- Shared feature groups across train/holdout: **0**
- Policy: do not use this holdout to select models or tune parameters. Evaluate the selected model on it once after all selection decisions are frozen.

## Inner CV (outer training rows only)

- Method: `StratifiedGroupKFold within outer training partition only`
- Seed: 43; folds: 5
- Training rows covered by validation exactly once: 8,843
- Shared feature groups across any fold's train/validation: **0**

| Fold | Training rows | Validation rows | Train phishing share | Validation phishing share | Shared groups |
|---:|---:|---:|---:|---:|---:|
| 0 | 7,071 | 1,772 | 44.31% | 44.30% | 0 |
| 1 | 7,077 | 1,766 | 44.31% | 44.28% | 0 |
| 2 | 7,073 | 1,770 | 44.31% | 44.29% | 0 |
| 3 | 7,075 | 1,768 | 44.30% | 44.34% | 0 |
| 4 | 7,076 | 1,767 | 44.30% | 44.31% | 0 |

## Conflicting-label groups

- Groups whose identical 30-feature vector has both labels: 64.
- Rows in those groups: 357.
- Policy: **retain all rows in the primary split**; never deduplicate or relabel silently. Since identical feature vectors can carry conflicting labels, the model sees contradictory targets within the same assigned partition/fold. Phase 3 should add a clearly labelled sensitivity analysis excluding entire conflicting groups, but the all-row result remains primary.
- Limitation: exact feature-vector equality is not proof of same website identity; no raw URL/website identifier is available in this dataset.

## Metric protocol (for Phase 3; no scores calculated in Phase 2)

- Primary model-selection metric: phishing-class F1, using `pos_label=0` and group-aware CV on the outer training partition only.
- Also report precision/recall/F1 separately for phishing class 0 and legitimate class 1, macro F1, balanced accuracy, accuracy, and a confusion matrix with labels ordered `[0, 1]`.
- Where a fitted model emits valid scores, compute phishing ROC-AUC and average precision from the score/probability for class 0. In scikit-learn terminology, report the latter explicitly as average precision rather than implying a trapezoidal interpolation of the PR curve.
- Do not call predicted class 0 the positive class implicitly; all phishing metrics must explicitly invert or set class semantics.

## Reproducibility and verification

- Source dataset SHA-256: `e20bf913b6c2e669f4e875e5d223be2542cc7c79a7b62d284ae9d9627c3d9678`
- Schema SHA-256: `0fe172eb670067a0048c62b601d5b63d311be827a5ee81c65f0fbd3dc480ad83`
- Split manifest SHA-256: `3dc82b79e2feb8f7b5fd23045be577be92d73d14aeed0005d3577782749f3ac7`
- CV assignment SHA-256: `9540bfa60ee0955455e8ba8ae3e039f6c83741f7b2760bac1db9a10fb1291cc9`
- Verification: **PASS**; outer overlaps = 0; inner CV group overlaps = 0.
- The selected fold assignment is a deterministic function of the dataset row order, group definition, random seeds, and scikit-learn version. Preserve the generated manifests; if the library version changes, compare the regenerated manifests and hashes rather than assuming identical indices.

## Limitations / decisions still open

1. Grouping prevents only identical engineered-feature vectors from crossing partitions; it does not ensure domain-level separation or establish real-world performance.
2. `StratifiedGroupKFold` approximates stratification while respecting groups; class proportions can vary slightly and should be reported.
3. The final holdout has both classes in this run, but Phase 3 must check score availability and metric preconditions explicitly.
4. Phase 3 should compare a conventional stratified random split as a diagnostic baseline, while using the group-aware protocol as the primary estimate.
5. No model training, tuning, artifact generation, or deployment was performed in this phase.
