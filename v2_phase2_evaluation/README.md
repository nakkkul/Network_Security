# V2 Phase 2 — Reproducible Evaluation Protocol

This folder is an isolated, evaluation-only addition. It **does not fit estimators, tune models, retrain, or modify V1 artifacts**.

## Purpose

- Validate the original UCI Phishing Websites tabular dataset against `data_schema/schema.yaml`.
- Map target labels consistently (`Result=-1 → 0 phishing`, `Result=1 → 1 legitimate`).
- Define exact duplicate feature vectors by the 30 input features only; the target is excluded from group identity.
- Create a deterministic group-aware holdout and group-aware inner CV assignment inside the training partition.
- Verify there is no feature-group overlap across outer train/holdout or inner CV train/validation splits.
- Provide a tested metric helper with phishing class `0` explicitly treated as the security-positive class.

## Where it goes

Copy this entire folder into the repository root on a separate `v2-development` branch:

```text
Network_Security/
├── Network_Data/phisingData.csv
├── data_schema/schema.yaml
└── v2_phase2_evaluation/
    ├── evaluation_protocol.py
    ├── build_evaluation_plan.py
    └── tests/test_evaluation_protocol.py
```

## Requirements

Use the project's V2 Conda environment, with pandas, NumPy, SciPy, scikit-learn, and PyYAML installed. The utility is compatible with modern scikit-learn versions that provide `StratifiedGroupKFold`. Package versions are recorded in output metadata; preserve the emitted split manifest instead of assuming indexes remain identical after package upgrades.

## Generate the split plan

From the repository root:

```bash
python v2_phase2_evaluation/build_evaluation_plan.py
```

Optional explicit paths:

```bash
python v2_phase2_evaluation/build_evaluation_plan.py --data-path Network_Data/phisingData.csv --schema-path data_schema/schema.yaml --output-dir v2_phase2_evaluation/output
```

The script writes under `v2_phase2_evaluation/output/` only. It does not write anywhere inside `final_model/` or the V1 application.

## Run unit tests

From the folder containing this README:

```bash
python -m unittest discover -s tests -v
```

Or from repository root:

```bash
python -m unittest discover -s v2_phase2_evaluation/tests -v
```

## Protocol choices

- Fixed seed: 42 for the outer split; 43 for inner CV.
- Outer holdout: first fold of `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`, approximately 20% of rows.
- Inner CV: five folds of `StratifiedGroupKFold` applied only to the outer training partition.
- Group identity in saved manifests: SHA-256 of canonical JSON for the 30 integer-valued feature entries; `Result` is not part of the key. The splitter uses deterministic first-occurrence factorized codes to preserve Phase 1 group ordering for the exact dataset hash and row order.
- Conflicting-label policy: retain all rows in the primary analysis, never silently relabel/deduplicate, and keep each feature-vector group together. A separate sensitivity analysis excluding entire conflicting groups should be reported later.
- Model selection (Phase 3): phishing-class F1 with explicit `pos_label=0`; final holdout is evaluated only after model selection is frozen.

## Generated outputs

- `split_manifest.csv`: original zero-based row index, train/holdout assignment, group hash, raw/mapped labels, and conflicting-group flag.
- `inner_cv_assignment.csv`: training-row fold assignment for the validation fold; other folds form that CV iteration's train partition.
- `split_diagnostics.json`: row/class/group diagnostics.
- `protocol_metadata.json`: data/schema hashes, seeds, feature list, environment versions, output hashes, and validation status.
- `evaluation_protocol_report.md`: readable summary and limitations.

## Interpretability and limitations

This only prevents exact duplicate engineered-feature vectors from crossing partitions. The source dataset has no raw URL or website ID, so the split cannot guarantee domain/site independence. `StratifiedGroupKFold` attempts to preserve class proportions while respecting groups; report the actual proportions. This phase does not compute classifier performance.

## Local Git safety before adding this folder

Read `GIT_BASELINE_CHECK.md` before creating `v2-development`. The actual local repository state cannot be inspected from this package; verify branch/ref hashes in your VS Code terminal and only add these files on the new V2 branch.
