#!/usr/bin/env python3
"""Build a reproducible V2 evaluation split plan without fitting classifiers.

Run from project root:
    python v2_phase2_evaluation/build_evaluation_plan.py

Optional:
    python v2_phase2_evaluation/build_evaluation_plan.py --data-path Network_Data/phisingData.csv --schema-path data_schema/schema.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from evaluation_protocol import (  # noqa: E402
    DEFAULT_INNER_FOLDS,
    DEFAULT_OUTER_FOLDS,
    DEFAULT_SEED,
    ProtocolError,
    create_split_plan,
    load_schema_feature_columns,
    sha256_file,
    verify_split_plan,
)


def display_path(path: Path, project_root: Path) -> str:
    """Prefer a portable project-relative path in metadata when possible."""
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=str(PROJECT_ROOT / "Network_Data" / "phisingData.csv"))
    parser.add_argument("--schema-path", default=str(PROJECT_ROOT / "data_schema" / "schema.yaml"))
    parser.add_argument("--output-dir", default=str(HERE / "output"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--outer-folds", type=int, default=DEFAULT_OUTER_FOLDS)
    parser.add_argument("--inner-folds", type=int, default=DEFAULT_INNER_FOLDS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_path = Path(args.data_path).resolve()
    schema_path = Path(args.schema_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not data_path.is_file():
        print(f"ERROR: dataset not found: {data_path}", file=sys.stderr)
        return 2
    if not schema_path.is_file():
        print(f"ERROR: schema not found: {schema_path}", file=sys.stderr)
        return 2

    try:
        feature_columns = load_schema_feature_columns(schema_path)
        df = pd.read_csv(data_path)
        source_hash = sha256_file(data_path)
        plan = create_split_plan(
            df,
            feature_columns,
            source_sha256=source_hash,
            seed=args.seed,
            outer_folds=args.outer_folds,
            inner_folds=args.inner_folds,
            allowed_feature_values=(-1, 0, 1),
        )
        verification = verify_split_plan(plan)
    except (ProtocolError, ValueError, KeyError) as exc:
        print(f"ERROR: evaluation protocol validation failed: {exc}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    plan.row_manifest.to_csv(output_dir / "split_manifest.csv", index=False)
    plan.cv_assignment.to_csv(output_dir / "inner_cv_assignment.csv", index=False)
    (output_dir / "split_diagnostics.json").write_text(
        json.dumps(plan.summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metadata = dict(plan.metadata)
    metadata.update({
        "source_path": display_path(data_path, PROJECT_ROOT),
        "source_sha256": source_hash,
        "schema_path": display_path(schema_path, PROJECT_ROOT),
        "schema_sha256": sha256_file(schema_path),
        "verification": verification,
        "row_manifest_sha256": sha256_file(output_dir / "split_manifest.csv"),
        "inner_cv_assignment_sha256": sha256_file(output_dir / "inner_cv_assignment.csv"),
    })
    (output_dir / "protocol_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = render_report(plan.summary, metadata, verification)
    (output_dir / "evaluation_protocol_report.md").write_text(report, encoding="utf-8")

    print("V2 evaluation split plan created. No estimator was fitted.")
    print(f"Dataset SHA-256: {source_hash}")
    print(f"Feature count: {len(feature_columns)}")
    outer = plan.summary["outer_split"]
    print(f"Train rows: {outer['train']['rows']}; holdout rows: {outer['holdout']['rows']}")
    print(f"Shared train/holdout groups: {outer['shared_feature_groups']}")
    print(f"Conflicting-label feature groups: {plan.summary['conflicting_label_policy']['conflicting_feature_group_count']}")
    print(f"Integrity verification: {verification['status']}")
    print(f"Outputs: {output_dir}")
    return 0


def render_report(summary: dict, metadata: dict, verification: dict) -> str:
    outer = summary["outer_split"]
    cv = summary["inner_cv"]
    conflict = summary["conflicting_label_policy"]
    lines = [
        "# V2 Phase 2 — Evaluation Protocol and Split Plan",
        "",
        "**No classifier was fitted or tuned to create this plan.** The files record a fixed, group-aware outer holdout and group-aware inner CV folds for later Phase 3 model selection.",
        "",
        "## Dataset identity",
        "",
        f"- SHA-256: `{metadata['source_sha256']}`",
        f"- Rows: {metadata['row_count']:,}",
        f"- Input features: {metadata['input_feature_count']}",
        f"- Target: `{metadata['target_column']}`; raw `-1 → 0 (phishing)`, raw `1 → 1 (legitimate)`",
        f"- Allowed encoded feature values: `{metadata['allowed_feature_values']}`",
        f"- Python/pandas/NumPy/scikit-learn/SciPy: {metadata['python_version']} / {metadata['pandas_version']} / {metadata['numpy_version']} / {metadata['scikit_learn_version']} / {metadata['scipy_version']}",
        "",
        "## Outer holdout (reserved before model selection)",
        "",
        f"- Method: `{outer['method']}`",
        f"- Seed: {outer['random_state']}; folds: {outer['n_splits']}; reserved fold index: {outer['fold_index_used_as_holdout']}",
        f"- Training rows: {outer['train']['rows']:,}; phishing share: {outer['train']['phishing_0_share']:.2%}; feature groups: {outer['train_feature_groups']:,}",
        f"- Holdout rows: {outer['holdout']['rows']:,}; phishing share: {outer['holdout']['phishing_0_share']:.2%}; feature groups: {outer['holdout_feature_groups']:,}",
        f"- Shared feature groups across train/holdout: **{outer['shared_feature_groups']}**",
        "- Policy: do not use this holdout to select models or tune parameters. Evaluate the selected model on it once after all selection decisions are frozen.",
        "",
        "## Inner CV (outer training rows only)",
        "",
        f"- Method: `{cv['method']}`",
        f"- Seed: {cv['random_state']}; folds: {cv['n_splits']}",
        f"- Training rows covered by validation exactly once: {cv['rows_covered_by_validation_once']:,}",
        f"- Shared feature groups across any fold's train/validation: **{cv['shared_feature_groups_across_fold_train_validation']}**",
        "",
        "| Fold | Training rows | Validation rows | Train phishing share | Validation phishing share | Shared groups |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for fold in cv["folds"]:
        lines.append(
            f"| {fold['fold']} | {fold['train_rows']:,} | {fold['validation_rows']:,} | {fold['train_phishing_share']:.2%} | {fold['validation_phishing_share']:.2%} | {fold['shared_feature_groups']} |"
        )
    lines += [
        "",
        "## Conflicting-label groups",
        "",
        f"- Groups whose identical 30-feature vector has both labels: {conflict['conflicting_feature_group_count']}.",
        f"- Rows in those groups: {conflict['rows_in_conflicting_feature_groups']}.",
        f"- Policy: **retain all rows in the primary split**; never deduplicate or relabel silently. Since identical feature vectors can carry conflicting labels, the model sees contradictory targets within the same assigned partition/fold. Phase 3 should add a clearly labelled sensitivity analysis excluding entire conflicting groups, but the all-row result remains primary.",
        "- Limitation: exact feature-vector equality is not proof of same website identity; no raw URL/website identifier is available in this dataset.",
        "",
        "## Metric protocol (for Phase 3; no scores calculated in Phase 2)",
        "",
        "- Primary model-selection metric: phishing-class F1, using `pos_label=0` and group-aware CV on the outer training partition only.",
        "- Also report precision/recall/F1 separately for phishing class 0 and legitimate class 1, macro F1, balanced accuracy, accuracy, and a confusion matrix with labels ordered `[0, 1]`.",
        "- Where a fitted model emits valid scores, compute phishing ROC-AUC and average precision from the score/probability for class 0. In scikit-learn terminology, report the latter explicitly as average precision rather than implying a trapezoidal interpolation of the PR curve.",
        "- Do not call predicted class 0 the positive class implicitly; all phishing metrics must explicitly invert or set class semantics.",
        "",
        "## Reproducibility and verification",
        "",
        f"- Source dataset SHA-256: `{metadata['source_sha256']}`",
        f"- Schema SHA-256: `{metadata['schema_sha256']}`",
        f"- Split manifest SHA-256: `{metadata['row_manifest_sha256']}`",
        f"- CV assignment SHA-256: `{metadata['inner_cv_assignment_sha256']}`",
        f"- Verification: **{verification['status']}**; outer overlaps = 0; inner CV group overlaps = 0.",
        "- The selected fold assignment is a deterministic function of the dataset row order, group definition, random seeds, and scikit-learn version. Preserve the generated manifests; if the library version changes, compare the regenerated manifests and hashes rather than assuming identical indices.",
        "",
        "## Limitations / decisions still open",
        "",
        "1. Grouping prevents only identical engineered-feature vectors from crossing partitions; it does not ensure domain-level separation or establish real-world performance.",
        "2. `StratifiedGroupKFold` approximates stratification while respecting groups; class proportions can vary slightly and should be reported.",
        "3. The final holdout has both classes in this run, but Phase 3 must check score availability and metric preconditions explicitly.",
        "4. Phase 3 should compare a conventional stratified random split as a diagnostic baseline, while using the group-aware protocol as the primary estimate.",
        "5. No model training, tuning, artifact generation, or deployment was performed in this phase.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
