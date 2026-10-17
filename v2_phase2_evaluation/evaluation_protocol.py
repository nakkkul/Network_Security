"""Leakage-aware split planning and classification-metric helpers for V2.

This module never fits an estimator. It only validates the tabular dataset,
creates reproducible group-aware split assignments, checks group isolation,
and calculates metrics from predictions supplied by a caller.
"""
from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import sklearn
import scipy
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

TARGET_COLUMN = "Result"
RAW_TO_MODEL_LABEL = {-1: 0, 1: 1}  # phishing=0, legitimate=1
MODEL_TO_CLASS_NAME = {0: "phishing", 1: "legitimate"}
DEFAULT_SEED = 42
DEFAULT_OUTER_FOLDS = 5
DEFAULT_INNER_FOLDS = 5


class ProtocolError(ValueError):
    """Raised when the dataset cannot safely be used for split planning."""


@dataclass(frozen=True)
class SplitPlan:
    """Row-level split assignments (source row indexes are zero-based)."""

    feature_columns: tuple[str, ...]
    target_column: str
    row_manifest: pd.DataFrame
    cv_assignment: pd.DataFrame
    summary: dict[str, Any]
    metadata: dict[str, Any]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_feature_group_id(row: Iterable[Any]) -> str:
    """Hash a feature vector canonically; target labels are never included."""
    normalized: list[int] = []
    for value in row:
        if pd.isna(value):
            raise ProtocolError("Feature vectors cannot contain missing values.")
        number = float(value)
        if not np.isfinite(number) or not number.is_integer():
            raise ProtocolError(f"Feature value {value!r} is not a finite integer.")
        normalized.append(int(number))
    encoded = json.dumps(normalized, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def make_feature_group_ids(features: pd.DataFrame) -> np.ndarray:
    """Return deterministic SHA-256 group IDs for rows with identical features."""
    if features.empty:
        raise ProtocolError("The feature dataframe is empty.")
    if features.columns.duplicated().any():
        raise ProtocolError("Duplicate feature column names are not allowed.")
    values = features.to_numpy()
    return np.asarray([stable_feature_group_id(row) for row in values], dtype=object)


def load_schema_feature_columns(schema_path: str | Path, target_column: str = TARGET_COLUMN) -> tuple[str, ...]:
    """Read the project's YAML schema and return ordered input columns only."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise ProtocolError("PyYAML is required to read data_schema/schema.yaml.") from exc

    with Path(schema_path).open("r", encoding="utf-8") as stream:
        schema = yaml.safe_load(stream)
    if not isinstance(schema, dict) or not isinstance(schema.get("columns"), list):
        raise ProtocolError("Schema must contain a 'columns' list.")

    all_columns: list[str] = []
    for item in schema["columns"]:
        if not isinstance(item, dict) or len(item) != 1:
            raise ProtocolError("Each schema column entry must be a one-key mapping.")
        all_columns.append(str(next(iter(item.keys()))))
    if len(all_columns) != len(set(all_columns)):
        raise ProtocolError("The schema contains duplicate column names.")
    if target_column not in all_columns:
        raise ProtocolError(f"Target column {target_column!r} is absent from schema.")
    return tuple(column for column in all_columns if column != target_column)


def validate_dataset(
    df: pd.DataFrame,
    feature_columns: Iterable[str],
    target_column: str = TARGET_COLUMN,
    *,
    allowed_feature_values: Iterable[int] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Validate source data and return (features, mapped labels, raw labels)."""
    feature_columns = tuple(feature_columns)
    if not feature_columns:
        raise ProtocolError("No feature columns were provided.")
    expected = list(feature_columns) + [target_column]
    if len(set(expected)) != len(expected):
        raise ProtocolError("Feature list contains duplicates or overlaps the target.")
    if list(df.columns) != expected:
        missing = [c for c in expected if c not in df.columns]
        unexpected = [c for c in df.columns if c not in expected]
        order_mismatch = not missing and not unexpected
        details = []
        if missing:
            details.append(f"missing columns={missing}")
        if unexpected:
            details.append(f"unexpected columns={unexpected}")
        if order_mismatch:
            details.append("column order differs from schema order")
        raise ProtocolError("Dataset schema mismatch: " + "; ".join(details))
    if df.empty:
        raise ProtocolError("Dataset has no rows.")
    if df.isna().any().any():
        missing_counts = df.isna().sum()
        columns = missing_counts[missing_counts > 0].to_dict()
        raise ProtocolError(f"Dataset contains missing values: {columns}")

    try:
        numeric = df.loc[:, expected].apply(pd.to_numeric, errors="raise")
    except Exception as exc:
        raise ProtocolError(f"Dataset contains non-numeric values: {exc}") from exc
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ProtocolError("Dataset contains infinite or non-finite numeric values.")
    feature_values = numeric.loc[:, list(feature_columns)].to_numpy(dtype=float)
    if not np.equal(feature_values, np.floor(feature_values)).all():
        raise ProtocolError("The selected UCI dataset expects integer-encoded features.")
    if allowed_feature_values is not None:
        allowed_values = set(int(value) for value in allowed_feature_values)
        observed_values = set(np.unique(feature_values).astype(int).tolist())
        bad_feature_values = sorted(observed_values - allowed_values)
        if bad_feature_values:
            raise ProtocolError(f"Features contain values outside the allowed domain {sorted(allowed_values)}: {bad_feature_values}")
    target_values = numeric[target_column].to_numpy(dtype=float)
    if not np.equal(target_values, np.floor(target_values)).all():
        raise ProtocolError("Target labels must be integer-valued before mapping.")
    raw_y = numeric[target_column].astype(int)
    bad_labels = sorted(set(raw_y.unique()) - set(RAW_TO_MODEL_LABEL))
    if bad_labels:
        raise ProtocolError(f"Target contains unsupported raw labels: {bad_labels}")
    y = raw_y.map(RAW_TO_MODEL_LABEL).astype(int)
    if set(y.unique()) != {0, 1}:
        raise ProtocolError("Both phishing (0) and legitimate (1) classes must be present.")
    X = numeric.loc[:, list(feature_columns)].copy()
    return X, y, raw_y


def _class_stats(y: pd.Series | np.ndarray) -> dict[str, Any]:
    arr = np.asarray(y, dtype=int)
    total = int(len(arr))
    counts = {int(label): int(np.sum(arr == label)) for label in (0, 1)}
    return {
        "rows": total,
        "phishing_0_count": counts[0],
        "phishing_0_share": counts[0] / total if total else None,
        "legitimate_1_count": counts[1],
        "legitimate_1_share": counts[1] / total if total else None,
    }


def _overlap_count(left: np.ndarray, right: np.ndarray) -> int:
    return len(set(left.tolist()).intersection(set(right.tolist())))


def _make_splitter(n_splits: int, seed: int) -> StratifiedGroupKFold:
    if n_splits < 2:
        raise ProtocolError("n_splits must be at least 2.")
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def create_split_plan(
    df: pd.DataFrame,
    feature_columns: Iterable[str],
    *,
    source_sha256: str,
    seed: int = DEFAULT_SEED,
    outer_folds: int = DEFAULT_OUTER_FOLDS,
    inner_folds: int = DEFAULT_INNER_FOLDS,
    target_column: str = TARGET_COLUMN,
    allowed_feature_values: Iterable[int] | None = None,
) -> SplitPlan:
    """Create deterministic group-aware holdout and inner CV assignments.

    The first fold from StratifiedGroupKFold(outer_folds) is reserved as the
    final holdout. The remaining rows are training-only. Inner CV is built
    exclusively inside that outer training partition.

    This function does not fit/tune a classifier or evaluate model predictions.
    """
    feature_columns = tuple(feature_columns)
    X, y, raw_y = validate_dataset(
        df, feature_columns, target_column, allowed_feature_values=allowed_feature_values
    )
    # Keep a portable SHA-256 group ID for manifests, but pass integer factorized
    # codes to scikit-learn. This matches the Phase 1 splitter's group ordering
    # while keeping the saved identifiers stable and independent of labels.
    group_ids = make_feature_group_ids(X)
    split_group_codes = pd.MultiIndex.from_frame(X).factorize(sort=False)[0]
    group_frame = pd.DataFrame({"group_id": group_ids, "y": y.to_numpy()})
    unique_labels_by_group = group_frame.groupby("group_id", sort=False)["y"].nunique()
    conflicting_groups = set(unique_labels_by_group[unique_labels_by_group > 1].index.tolist())

    outer_splitter = _make_splitter(outer_folds, seed)
    outer_train_idx, holdout_idx = next(outer_splitter.split(X, y, groups=split_group_codes))
    train_groups = group_ids[outer_train_idx]
    holdout_groups = group_ids[holdout_idx]
    overlap = _overlap_count(train_groups, holdout_groups)
    if overlap:
        raise ProtocolError(f"Outer split leaks {overlap} feature groups across train/holdout.")
    if len(np.unique(y.iloc[outer_train_idx])) != 2 or len(np.unique(y.iloc[holdout_idx])) != 2:
        raise ProtocolError("Outer training and holdout partitions must each contain both classes.")

    # Create a row-level manifest indexed to the original CSV order.
    partition = np.full(len(df), "train", dtype=object)
    partition[holdout_idx] = "holdout"
    group_conflict = np.asarray([g in conflicting_groups for g in group_ids], dtype=bool)
    manifest = pd.DataFrame({
        "source_row_index": np.arange(len(df), dtype=int),
        "partition": partition,
        "feature_group_id": group_ids,
        "target_raw": raw_y.to_numpy(dtype=int),
        "target_mapped": y.to_numpy(dtype=int),
        "is_conflicting_feature_group": group_conflict,
    })

    # Inner CV is generated only against the outer-training portion.
    X_train = X.iloc[outer_train_idx].reset_index(drop=True)
    y_train = y.iloc[outer_train_idx].reset_index(drop=True)
    stable_groups_train = group_ids[outer_train_idx]
    groups_train = split_group_codes[outer_train_idx]
    original_rows_train = outer_train_idx
    inner_splitter = _make_splitter(inner_folds, seed + 1)
    cv_assignments = np.full(len(X_train), -1, dtype=int)
    fold_rows: list[dict[str, Any]] = []
    inner_fold_records: list[dict[str, Any]] = []
    for fold, (inner_train_local, validation_local) in enumerate(
        inner_splitter.split(X_train, y_train, groups=groups_train)
    ):
        if _overlap_count(groups_train[inner_train_local], groups_train[validation_local]):
            raise ProtocolError(f"Inner CV fold {fold} leaks feature groups across train/validation.")
        if np.any(cv_assignments[validation_local] != -1):
            raise ProtocolError("An outer-training row appears in multiple inner validation folds.")
        cv_assignments[validation_local] = fold
        global_train_rows = original_rows_train[inner_train_local]
        global_validation_rows = original_rows_train[validation_local]
        fold_rows.append({
            "fold": int(fold),
            "train_rows": int(len(global_train_rows)),
            "validation_rows": int(len(global_validation_rows)),
            "train_feature_groups": int(len(np.unique(groups_train[inner_train_local]))),
            "validation_feature_groups": int(len(np.unique(groups_train[validation_local]))),
            "shared_feature_groups": int(_overlap_count(groups_train[inner_train_local], groups_train[validation_local])),
            "train_phishing_share": float(np.mean(y.to_numpy()[global_train_rows] == 0)),
            "validation_phishing_share": float(np.mean(y.to_numpy()[global_validation_rows] == 0)),
        })
        for row in global_train_rows:
            inner_fold_records.append({"source_row_index": int(row), "fold": int(fold), "role": "train"})
        for row in global_validation_rows:
            inner_fold_records.append({"source_row_index": int(row), "fold": int(fold), "role": "validation"})

    if np.any(cv_assignments < 0):
        raise ProtocolError("Some training rows were not assigned to an inner validation fold.")
    # Every duplicate group in outer training must belong to just one validation fold.
    assignment_frame = pd.DataFrame({"group_id": stable_groups_train, "validation_fold": cv_assignments})
    folds_per_group = assignment_frame.groupby("group_id")["validation_fold"].nunique()
    if (folds_per_group != 1).any():
        raise ProtocolError("An identical feature-vector group was assigned to multiple CV folds.")

    # Conflict labels are reported, not corrected or removed.
    target_by_group = group_frame.groupby("group_id")["y"].agg(lambda values: sorted(set(map(int, values))))
    conflict_groups_sorted = sorted(conflicting_groups)
    train_conflict_rows = int(manifest.loc[(manifest.partition == "train") & manifest.is_conflicting_feature_group].shape[0])
    holdout_conflict_rows = int(manifest.loc[(manifest.partition == "holdout") & manifest.is_conflicting_feature_group].shape[0])
    holdout_n = int(len(holdout_idx))
    train_n = int(len(outer_train_idx))
    summary = {
        "outer_split": {
            "method": "StratifiedGroupKFold; first fold reserved as final holdout",
            "n_splits": int(outer_folds),
            "fold_index_used_as_holdout": 0,
            "random_state": int(seed),
            "train": _class_stats(y.iloc[outer_train_idx]),
            "holdout": _class_stats(y.iloc[holdout_idx]),
            "train_feature_groups": int(len(np.unique(train_groups))),
            "holdout_feature_groups": int(len(np.unique(holdout_groups))),
            "shared_feature_groups": int(overlap),
            "train_fraction": train_n / len(df),
            "holdout_fraction": holdout_n / len(df),
            "train_rows_in_conflicting_groups": train_conflict_rows,
            "holdout_rows_in_conflicting_groups": holdout_conflict_rows,
        },
        "inner_cv": {
            "method": "StratifiedGroupKFold within outer training partition only",
            "n_splits": int(inner_folds),
            "random_state": int(seed + 1),
            "folds": fold_rows,
            "rows_covered_by_validation_once": int(len(cv_assignments)),
            "shared_feature_groups_across_fold_train_validation": 0,
        },
        "conflicting_label_policy": {
            "conflicting_feature_group_count": int(len(conflicting_groups)),
            "rows_in_conflicting_feature_groups": int(group_conflict.sum()),
            "policy": "Retain all rows in the primary split; keep each exact feature vector in one partition/fold; do not relabel or deduplicate. Conduct a separately labelled sensitivity analysis later.",
            "limitation": "Same feature vector does not establish same website identity because raw URLs/website IDs are absent.",
        },
        "target_mapping": {"-1": {"mapped": 0, "meaning": "phishing"}, "1": {"mapped": 1, "meaning": "legitimate"}},
        "no_estimators_fitted": True,
    }
    metadata = {
        "protocol_version": "1.0.0",
        "allowed_feature_values": sorted(set(int(v) for v in allowed_feature_values)) if allowed_feature_values is not None else None,
        "source_sha256": source_sha256,
        "row_count": int(len(df)),
        "input_feature_count": int(len(feature_columns)),
        "feature_columns_in_order": list(feature_columns),
        "target_column": target_column,
        "target_mapping": {"-1": 0, "1": 1},
        "group_definition": "SHA-256 of canonical JSON array of all 30 integer input features; target excluded",
        "outer_seed": int(seed),
        "outer_n_splits": int(outer_folds),
        "outer_holdout_fold": 0,
        "inner_seed": int(seed + 1),
        "inner_n_splits": int(inner_folds),
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "scikit_learn_version": sklearn.__version__,
        "scipy_version": scipy.__version__,
        "no_estimators_fitted": True,
        "row_order_matters_for_source_row_index": True,
    }
    cv_manifest = pd.DataFrame(inner_fold_records).sort_values(["fold", "source_row_index", "role"]).reset_index(drop=True)
    return SplitPlan(
        feature_columns=feature_columns,
        target_column=target_column,
        row_manifest=manifest,
        cv_assignment=pd.DataFrame({
            "source_row_index": original_rows_train.astype(int),
            "feature_group_id": stable_groups_train,
            "validation_fold": cv_assignments,
            "target_raw": raw_y.iloc[original_rows_train].to_numpy(dtype=int),
            "target_mapped": y.iloc[original_rows_train].to_numpy(dtype=int),
        }).sort_values("source_row_index").reset_index(drop=True),
        summary=summary,
        metadata=metadata | {"group_id_target_label_sets": {g: labels for g, labels in target_by_group.items() if len(labels) > 1}},
    )


def verify_split_plan(plan: SplitPlan) -> dict[str, Any]:
    """Independently validate row coverage and group isolation properties."""
    manifest = plan.row_manifest
    expected_rows = set(manifest["source_row_index"].astype(int).tolist())
    if len(expected_rows) != len(manifest):
        raise ProtocolError("Source row indexes in manifest are not unique.")
    if set(manifest["partition"].unique()) != {"train", "holdout"}:
        raise ProtocolError("Manifest must contain train and holdout partitions.")
    group_partitions = manifest.groupby("feature_group_id")["partition"].nunique()
    if (group_partitions != 1).any():
        raise ProtocolError("A feature group crosses the train/holdout boundary.")

    cv = plan.cv_assignment
    expected_train_rows = set(manifest.loc[manifest.partition == "train", "source_row_index"].astype(int))
    if set(cv.source_row_index.astype(int)) != expected_train_rows:
        raise ProtocolError("CV assignment rows do not exactly match the outer training partition.")
    if cv.source_row_index.duplicated().any():
        raise ProtocolError("A training source row has multiple CV validation assignments.")
    if cv.validation_fold.isna().any() or (cv.validation_fold < 0).any():
        raise ProtocolError("Invalid CV fold assignment.")
    group_folds = cv.groupby("feature_group_id")["validation_fold"].nunique()
    if (group_folds != 1).any():
        raise ProtocolError("An identical feature group crosses inner validation folds.")
    folds = sorted(cv.validation_fold.unique().astype(int).tolist())
    cv_fold_train_validation_overlaps: dict[str, int] = {}
    group_to_fold = dict(zip(cv.feature_group_id, cv.validation_fold.astype(int)))
    for fold in folds:
        validation = cv.loc[cv.validation_fold == fold]
        training_groups = set(cv.loc[cv.validation_fold != fold, "feature_group_id"])
        overlap = len(training_groups.intersection(set(validation.feature_group_id)))
        cv_fold_train_validation_overlaps[str(fold)] = overlap
        if overlap:
            raise ProtocolError(f"Inner CV fold {fold} has {overlap} shared feature groups.")
    return {
        "status": "PASS",
        "rows_in_manifest": int(len(manifest)),
        "unique_source_row_indexes": int(len(expected_rows)),
        "outer_shared_feature_groups": 0,
        "inner_cv_fold_count": int(len(folds)),
        "inner_cv_source_rows_covered_once": int(len(cv)),
        "inner_cv_shared_feature_groups_by_fold": cv_fold_train_validation_overlaps,
        "no_estimators_fitted": True,
    }


def compute_classification_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    *,
    phishing_score: Iterable[float] | None = None,
) -> dict[str, Any]:
    """Calculate metrics with phishing (class 0) explicitly treated as positive.

    `phishing_score`, when supplied, must be a score/probability whose larger
    values indicate stronger evidence for class 0. For probability scores it
    should be in [0, 1]. This helper does not fit or calibrate a model.
    """
    yt = np.asarray(list(y_true), dtype=int)
    yp = np.asarray(list(y_pred), dtype=int)
    if yt.ndim != 1 or yp.ndim != 1 or len(yt) == 0 or len(yt) != len(yp):
        raise ProtocolError("y_true and y_pred must be non-empty 1D arrays of equal length.")
    if not set(np.unique(yt)).issubset({0, 1}) or not set(np.unique(yp)).issubset({0, 1}):
        raise ProtocolError("Metrics require mapped labels 0=phishing and 1=legitimate.")
    phishing_precision = precision_score(yt, yp, pos_label=0, zero_division=0)
    phishing_recall = recall_score(yt, yp, pos_label=0, zero_division=0)
    phishing_f1 = f1_score(yt, yp, pos_label=0, zero_division=0)
    legitimate_precision = precision_score(yt, yp, pos_label=1, zero_division=0)
    legitimate_recall = recall_score(yt, yp, pos_label=1, zero_division=0)
    legitimate_f1 = f1_score(yt, yp, pos_label=1, zero_division=0)
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(yt, yp)),
        "phishing_precision": float(phishing_precision),
        "phishing_recall": float(phishing_recall),
        "phishing_f1": float(phishing_f1),
        "legitimate_precision": float(legitimate_precision),
        "legitimate_recall": float(legitimate_recall),
        "legitimate_f1": float(legitimate_f1),
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "confusion_matrix_labels": ["actual_phishing_0", "actual_legitimate_1"],
        "confusion_matrix_predicted_labels": ["predicted_phishing_0", "predicted_legitimate_1"],
        "confusion_matrix": cm.astype(int).tolist(),
        "roc_auc_phishing": None,
        "pr_auc_phishing_average_precision": None,
        "score_definition": "higher phishing_score means more likely class 0 / phishing",
    }
    if phishing_score is not None:
        score = np.asarray(list(phishing_score), dtype=float)
        if score.ndim != 1 or len(score) != len(yt):
            raise ProtocolError("phishing_score must have the same length as y_true.")
        if not np.isfinite(score).all():
            raise ProtocolError("phishing_score must contain only finite values.")
        # Scores may be calibrated probabilities in [0, 1] or finite decision
        # scores; in either case larger values must indicate phishing/class 0.
        phishing_true = (yt == 0).astype(int)
        if len(np.unique(phishing_true)) == 2:
            result["roc_auc_phishing"] = float(roc_auc_score(phishing_true, score))
            # sklearn's average_precision_score is the non-interpolated summary
            # of the precision-recall curve; call it average precision explicitly.
            result["pr_auc_phishing_average_precision"] = float(average_precision_score(phishing_true, score))
    return result
