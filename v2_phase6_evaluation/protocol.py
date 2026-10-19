"""Read-only validation of the frozen Phase 2 data and split protocol.

This module deliberately performs no estimator fitting, prediction, holdout
scoring, manifest generation, or file writes.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml


# These values pin Phase 6 to the exact Phase 2 assets already verified on
# v2-development. Any intentional data/split change requires a protocol revision.
EXPECTED_SHA256 = {
    "dataset": "e20bf913b6c2e669f4e875e5d223be2542cc7c79a7b62d284ae9d9627c3d9678",
    "schema": "0fe172eb670067a0048c62b601d5b63d311be827a5ee81c65f0fbd3dc480ad83",
    "split_manifest": "3dc82b79e2feb8f7b5fd23045be577be92d73d14aeed0005d3577782749f3ac7",
    "inner_cv_assignment": "9540bfa60ee0955455e8ba8ae3e039f6c83741f7b2760bac1db9a10fb1291cc9",
}
EXPECTED_ROW_COUNT = 11_055
EXPECTED_FEATURE_COUNT = 30
EXPECTED_TRAIN_ROWS = 8_843
EXPECTED_HOLDOUT_ROWS = 2_212
EXPECTED_INNER_FOLDS = 5
EXPECTED_CONFLICTING_GROUPS = 64
EXPECTED_CONFLICTING_ROWS = 357
TARGET_MAPPING = {"-1": 0, "1": 1}
TARGET_COLUMN = "Result"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Phase6ProtocolError(RuntimeError):
    """Raised when a frozen Phase 2 benchmark invariant is not satisfied."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_feature_group_id(row: Iterable[Any]) -> str:
    """Reproduce Phase 2's canonical SHA-256 hash for the 30 input features."""
    normalized: list[int] = []
    for value in row:
        if pd.isna(value):
            raise Phase6ProtocolError("Feature vectors must not contain missing values.")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise Phase6ProtocolError(f"Feature value {value!r} is not numeric.") from exc
        if not math.isfinite(number) or not number.is_integer():
            raise Phase6ProtocolError(f"Feature value {value!r} is not a finite integer.")
        normalized.append(int(number))
    encoded = json.dumps(normalized, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _integer_series(frame: pd.DataFrame, column: str) -> pd.Series:
    try:
        numbers = pd.to_numeric(frame[column], errors="raise")
    except Exception as exc:
        raise Phase6ProtocolError(f"Column {column!r} must contain numeric values.") from exc
    values = numbers.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise Phase6ProtocolError(f"Column {column!r} must contain finite integers only.")
    return pd.Series(values.astype(np.int64), index=frame.index, name=column)


def _parse_boolean_series(series: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapping = {"true": True, "false": False, "1": True, "0": False}
    normalized = series.astype(str).str.strip().str.lower()
    if not normalized.isin(mapping).all():
        raise Phase6ProtocolError(f"Column {name!r} must contain only boolean values.")
    return normalized.map(mapping).astype(bool)


def _load_schema_columns(schema_path: Path) -> list[str]:
    try:
        with schema_path.open("r", encoding="utf-8") as stream:
            schema = yaml.safe_load(stream)
    except Exception as exc:
        raise Phase6ProtocolError(f"Unable to read schema file: {schema_path}") from exc
    if not isinstance(schema, dict) or not isinstance(schema.get("columns"), list):
        raise Phase6ProtocolError("Schema must contain a 'columns' list.")
    columns: list[str] = []
    for entry in schema["columns"]:
        if not isinstance(entry, dict) or len(entry) != 1:
            raise Phase6ProtocolError("Every schema column entry must be a one-key mapping.")
        columns.append(str(next(iter(entry.keys()))))
    if len(columns) != len(set(columns)):
        raise Phase6ProtocolError("Schema contains duplicate column names.")
    return columns


@dataclass(frozen=True)
class VerifiedPhase2Protocol:
    """Verified split assignments without exposing holdout features or labels."""

    root: Path
    dataset_path: Path
    feature_columns: tuple[str, ...]
    target_column: str
    metadata: dict[str, Any]
    # Intentionally omits target_raw/target_mapped to keep holdout labels out of
    # downstream model-selection interfaces.
    row_manifest: pd.DataFrame
    # This assignment contains only outer-training rows and their labels.
    cv_assignment: pd.DataFrame
    hashes: dict[str, str]
    summary: dict[str, Any]


def verify_phase2_protocol(root: str | Path | None = None) -> VerifiedPhase2Protocol:
    """Verify dataset/schema/manifests and group boundaries without fitting.

    The dataset file is read to verify source-row alignment and feature-group
    hashes. No scores or predictions are computed, and no holdout labels are
    returned to the caller.
    """
    project_root = Path(root or PROJECT_ROOT).resolve()
    dataset_path = project_root / "Network_Data" / "phisingData.csv"
    schema_path = project_root / "data_schema" / "schema.yaml"
    output_dir = project_root / "v2_phase2_evaluation" / "output"
    manifest_path = output_dir / "split_manifest.csv"
    cv_path = output_dir / "inner_cv_assignment.csv"
    metadata_path = output_dir / "protocol_metadata.json"

    required_paths = {
        "dataset": dataset_path,
        "schema": schema_path,
        "split_manifest": manifest_path,
        "inner_cv_assignment": cv_path,
        "protocol_metadata": metadata_path,
    }
    missing = [name for name, path in required_paths.items() if not path.is_file()]
    if missing:
        raise Phase6ProtocolError(
            "Required Phase 2 files are missing: " + ", ".join(missing)
            + ". Stop; do not regenerate or replace the frozen manifests."
        )

    hashes = {
        "dataset": sha256_file(dataset_path),
        "schema": sha256_file(schema_path),
        "split_manifest": sha256_file(manifest_path),
        "inner_cv_assignment": sha256_file(cv_path),
    }
    for asset_name, expected in EXPECTED_SHA256.items():
        if hashes[asset_name] != expected:
            raise Phase6ProtocolError(
                f"Frozen {asset_name} SHA-256 mismatch: expected {expected}, got {hashes[asset_name]}. "
                "Stop; do not regenerate the manifest or continue with model selection."
            )

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise Phase6ProtocolError("protocol_metadata.json is not valid JSON.") from exc

    expected_metadata_hashes = {
        "source_sha256": EXPECTED_SHA256["dataset"],
        "schema_sha256": EXPECTED_SHA256["schema"],
        "row_manifest_sha256": EXPECTED_SHA256["split_manifest"],
        "inner_cv_assignment_sha256": EXPECTED_SHA256["inner_cv_assignment"],
    }
    for key, expected in expected_metadata_hashes.items():
        if metadata.get(key) != expected:
            raise Phase6ProtocolError(f"Metadata field {key!r} does not match the pinned Phase 2 asset hash.")
    if metadata.get("row_count") != EXPECTED_ROW_COUNT:
        raise Phase6ProtocolError("Metadata row_count differs from the frozen Phase 2 row count.")
    if metadata.get("input_feature_count") != EXPECTED_FEATURE_COUNT:
        raise Phase6ProtocolError("Metadata feature count differs from the frozen Phase 2 schema.")
    if metadata.get("target_column") != TARGET_COLUMN:
        raise Phase6ProtocolError("Unexpected target column in protocol metadata.")
    if metadata.get("target_mapping") != TARGET_MAPPING:
        raise Phase6ProtocolError("Target mapping differs from the frozen Phase 2 mapping.")
    if metadata.get("inner_n_splits") != EXPECTED_INNER_FOLDS:
        raise Phase6ProtocolError("Inner fold count differs from the frozen Phase 2 protocol.")
    if metadata.get("no_estimators_fitted") is not True:
        raise Phase6ProtocolError("Phase 2 metadata must confirm that no estimator was fitted to create the plan.")

    all_schema_columns = _load_schema_columns(schema_path)
    feature_columns = tuple(metadata.get("feature_columns_in_order", []))
    if len(feature_columns) != EXPECTED_FEATURE_COUNT or len(set(feature_columns)) != EXPECTED_FEATURE_COUNT:
        raise Phase6ProtocolError("Metadata must identify exactly 30 unique ordered feature columns.")
    if all_schema_columns != [*feature_columns, TARGET_COLUMN]:
        raise Phase6ProtocolError("Ordered schema columns do not match the frozen feature list and target.")

    try:
        dataset = pd.read_csv(dataset_path, low_memory=False)
        manifest = pd.read_csv(manifest_path)
        cv = pd.read_csv(cv_path)
    except Exception as exc:
        raise Phase6ProtocolError("Unable to parse the canonical CSV or Phase 2 manifests.") from exc

    expected_dataset_columns = [*feature_columns, TARGET_COLUMN]
    if list(dataset.columns) != expected_dataset_columns:
        raise Phase6ProtocolError("Canonical CSV columns/order differ from the frozen schema.")
    if len(dataset) != EXPECTED_ROW_COUNT:
        raise Phase6ProtocolError(f"Expected {EXPECTED_ROW_COUNT} source rows, found {len(dataset)}.")

    feature_matrix = dataset.loc[:, list(feature_columns)]
    if feature_matrix.isna().any().any():
        raise Phase6ProtocolError("Canonical feature matrix contains missing values.")
    try:
        feature_values = feature_matrix.to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise Phase6ProtocolError("Canonical feature matrix contains non-numeric values.") from exc
    if not np.isfinite(feature_values).all():
        raise Phase6ProtocolError("Canonical feature matrix contains NaN or infinite values.")
    if not np.equal(feature_values, np.floor(feature_values)).all():
        raise Phase6ProtocolError("Canonical feature matrix contains non-integer values.")
    allowed_values = set(metadata.get("allowed_feature_values", []))
    if allowed_values != {-1, 0, 1} or not np.isin(feature_values, list(allowed_values)).all():
        raise Phase6ProtocolError("Canonical features violate the frozen allowed-value set {-1, 0, 1}.")

    raw_target = _integer_series(dataset, TARGET_COLUMN)
    if set(raw_target.unique().tolist()) != {-1, 1}:
        raise Phase6ProtocolError("Canonical target must contain exactly raw labels -1 and 1.")
    mapped_target = raw_target.map({-1: 0, 1: 1}).astype(np.int64)

    required_manifest_columns = [
        "source_row_index", "partition", "feature_group_id", "target_raw",
        "target_mapped", "is_conflicting_feature_group",
    ]
    if list(manifest.columns) != required_manifest_columns:
        raise Phase6ProtocolError("Split manifest columns do not match the required Phase 2 structure.")
    manifest["source_row_index"] = _integer_series(manifest, "source_row_index")
    manifest["target_raw"] = _integer_series(manifest, "target_raw")
    manifest["target_mapped"] = _integer_series(manifest, "target_mapped")
    manifest["is_conflicting_feature_group"] = _parse_boolean_series(
        manifest["is_conflicting_feature_group"], "is_conflicting_feature_group"
    )
    manifest = manifest.sort_values("source_row_index").reset_index(drop=True)
    source_indices = manifest["source_row_index"].to_numpy(dtype=np.int64)
    if len(np.unique(source_indices)) != len(manifest) or not np.array_equal(source_indices, np.arange(len(dataset))):
        raise Phase6ProtocolError("Manifest source_row_index must cover every canonical CSV row exactly once.")
    if set(manifest["partition"].astype(str).unique()) != {"train", "holdout"}:
        raise Phase6ProtocolError("Manifest must contain exactly 'train' and 'holdout' partitions.")

    if not np.array_equal(manifest["target_raw"].to_numpy(), raw_target.to_numpy()):
        raise Phase6ProtocolError("Manifest raw targets do not align with canonical source rows.")
    if not np.array_equal(manifest["target_mapped"].to_numpy(), mapped_target.to_numpy()):
        raise Phase6ProtocolError("Manifest mapped targets do not align with canonical source rows.")

    computed_groups = np.asarray([stable_feature_group_id(row) for row in feature_values], dtype=object)
    if not np.array_equal(manifest["feature_group_id"].astype(str).to_numpy(), computed_groups):
        raise Phase6ProtocolError("Manifest feature-group IDs do not match the canonical feature rows.")

    labels_by_group = pd.DataFrame({"group_id": computed_groups, "target": mapped_target.to_numpy()})
    group_label_counts = labels_by_group.groupby("group_id")["target"].nunique()
    conflicting_group_ids = set(group_label_counts[group_label_counts > 1].index.astype(str))
    expected_conflict_flags = pd.Series(computed_groups).isin(conflicting_group_ids).to_numpy(dtype=bool)
    if not np.array_equal(manifest["is_conflicting_feature_group"].to_numpy(dtype=bool), expected_conflict_flags):
        raise Phase6ProtocolError("Manifest conflict flags do not match feature groups with conflicting target labels.")
    conflicting_rows = int(expected_conflict_flags.sum())
    if len(conflicting_group_ids) != EXPECTED_CONFLICTING_GROUPS or conflicting_rows != EXPECTED_CONFLICTING_ROWS:
        raise Phase6ProtocolError("Conflicting-label group counts differ from the frozen Phase 2 diagnostics.")

    train_manifest = manifest.loc[manifest["partition"] == "train"].copy()
    holdout_manifest = manifest.loc[manifest["partition"] == "holdout"].copy()
    if len(train_manifest) != EXPECTED_TRAIN_ROWS or len(holdout_manifest) != EXPECTED_HOLDOUT_ROWS:
        raise Phase6ProtocolError("Outer train/holdout row counts differ from the frozen Phase 2 split.")
    if set(train_manifest["target_mapped"].unique()) != {0, 1} or set(holdout_manifest["target_mapped"].unique()) != {0, 1}:
        raise Phase6ProtocolError("Both outer partitions must contain phishing class 0 and legitimate class 1.")
    partition_counts = manifest.groupby("feature_group_id")["partition"].nunique()
    if (partition_counts > 1).any():
        raise Phase6ProtocolError("Identical feature groups cross the outer train/holdout boundary.")

    required_cv_columns = ["source_row_index", "feature_group_id", "validation_fold", "target_raw", "target_mapped"]
    if list(cv.columns) != required_cv_columns:
        raise Phase6ProtocolError("Inner CV assignment columns do not match the required Phase 2 structure.")
    for column in ("source_row_index", "validation_fold", "target_raw", "target_mapped"):
        cv[column] = _integer_series(cv, column)
    cv = cv.sort_values("source_row_index").reset_index(drop=True)
    if cv["source_row_index"].duplicated().any():
        raise Phase6ProtocolError("A training source row has more than one inner-CV assignment.")
    expected_train_rows = set(train_manifest["source_row_index"].astype(int).tolist())
    if set(cv["source_row_index"].astype(int).tolist()) != expected_train_rows:
        raise Phase6ProtocolError("Inner CV rows must exactly cover the outer training rows and exclude holdout rows.")
    if set(cv["validation_fold"].astype(int).unique().tolist()) != set(range(EXPECTED_INNER_FOLDS)):
        raise Phase6ProtocolError("Inner CV assignments must contain exactly folds 0 through 4.")

    manifest_lookup = manifest.set_index("source_row_index")
    for cv_row in cv.itertuples(index=False):
        source_index = int(cv_row.source_row_index)
        source_record = manifest_lookup.loc[source_index]
        if str(cv_row.feature_group_id) != str(source_record.feature_group_id):
            raise Phase6ProtocolError("Inner CV feature-group IDs do not match the frozen split manifest.")
        if int(cv_row.target_raw) != int(source_record.target_raw) or int(cv_row.target_mapped) != int(source_record.target_mapped):
            raise Phase6ProtocolError("Inner CV target labels do not match the outer-training manifest.")
        if str(source_record.partition) != "train":
            raise Phase6ProtocolError("A reserved outer-holdout row appears in inner CV assignments.")

    group_fold_counts = cv.groupby("feature_group_id")["validation_fold"].nunique()
    if (group_fold_counts > 1).any():
        raise Phase6ProtocolError("An identical feature group crosses inner validation folds.")
    for fold_number in range(EXPECTED_INNER_FOLDS):
        validation = cv.loc[cv["validation_fold"] == fold_number]
        training = cv.loc[cv["validation_fold"] != fold_number]
        if set(validation["feature_group_id"].astype(str)).intersection(set(training["feature_group_id"].astype(str))):
            raise Phase6ProtocolError(f"Inner fold {fold_number} has feature groups shared by training and validation.")
        if set(validation["target_mapped"].astype(int).unique()) != {0, 1}:
            raise Phase6ProtocolError(f"Inner fold {fold_number} validation partition does not contain both classes.")
        if set(training["target_mapped"].astype(int).unique()) != {0, 1}:
            raise Phase6ProtocolError(f"Inner fold {fold_number} training partition does not contain both classes.")

    summary = {
        "status": "PASS",
        "row_count": len(dataset),
        "feature_count": len(feature_columns),
        "outer_train_rows": len(train_manifest),
        "reserved_holdout_rows": len(holdout_manifest),
        "outer_shared_feature_groups": 0,
        "inner_cv_fold_count": EXPECTED_INNER_FOLDS,
        "inner_cv_training_rows_covered_once": len(cv),
        "inner_cv_shared_feature_groups_by_fold": {str(fold): 0 for fold in range(EXPECTED_INNER_FOLDS)},
        "conflicting_feature_groups": len(conflicting_group_ids),
        "rows_in_conflicting_feature_groups": conflicting_rows,
        "estimator_fitted": False,
        "holdout_scored": False,
        "holdout_predictions_generated": False,
    }

    # Keep the outer holdout labels out of the returned manifest.
    public_manifest = manifest[[
        "source_row_index", "partition", "feature_group_id", "is_conflicting_feature_group"
    ]].copy()
    public_train_cv = cv.copy()
    public_train_cv["source_row_index"] = public_train_cv["source_row_index"].astype(np.int64)
    public_train_cv["validation_fold"] = public_train_cv["validation_fold"].astype(np.int64)
    public_train_cv["target_raw"] = public_train_cv["target_raw"].astype(np.int64)
    public_train_cv["target_mapped"] = public_train_cv["target_mapped"].astype(np.int64)

    safe_metadata = {
        key: metadata[key]
        for key in (
            "protocol_version", "source_path", "source_sha256", "schema_path", "schema_sha256",
            "row_manifest_sha256", "inner_cv_assignment_sha256", "row_count",
            "input_feature_count", "feature_columns_in_order", "target_column", "target_mapping",
            "allowed_feature_values", "group_definition", "outer_n_splits", "outer_seed",
            "outer_holdout_fold", "inner_n_splits", "inner_seed",
        )
    }
    return VerifiedPhase2Protocol(
        root=project_root,
        dataset_path=dataset_path,
        feature_columns=feature_columns,
        target_column=TARGET_COLUMN,
        metadata=safe_metadata,
        row_manifest=public_manifest,
        cv_assignment=public_train_cv,
        hashes=hashes,
        summary=summary,
    )
