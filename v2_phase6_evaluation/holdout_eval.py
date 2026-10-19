"""One-time evaluation of the frozen Phase 7 model on the reserved Phase 2 holdout.

This module does not fit or tune an estimator. It validates frozen protocol hashes,
the Phase 7 fit report, every Phase 7 artifact hash, and protected legacy artifacts
before making exactly one ``Pipeline.predict`` call on the 2,212 reserved rows.
A durable one-time marker is written before that prediction call; if execution is
interrupted after the marker, this module deliberately refuses to rerun automatically.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline

from .final_fit import (
    EXPECTED_RANDOM_STATE,
    EXPECTED_SELECTED_CANDIDATE,
    EXPECTED_SELECTED_PARAMETERS,
    EXPECTED_SELECTION_METRIC,
    EXPECTED_OUTER_TRAIN_ROWS,
    EXPECTED_RESERVED_HOLDOUT_ROWS,
    PROTECTED_ARTIFACT_SHA256,
)
from .protocol import EXPECTED_SHA256, PROJECT_ROOT, Phase6ProtocolError, sha256_file, verify_phase2_protocol

PHASE8_AUTH_ENV = "NETWORK_SECURITY_PHASE8_ALLOW_HOLDOUT_EVALUATION"
PHASE8_AUTH_TOKEN = "I_APPROVE_PHASE8_ONE_TIME_RESERVED_HOLDOUT_EVALUATION"
PHASE8_GUARD_FILENAME = "phase8_holdout_evaluation_once.json"
PHASE8_OUTPUT_PREFIX = "phase8_holdout_evaluation_"
EXPECTED_FEATURE_COUNT = 30
EXPECTED_TARGET_MAPPING = {"raw_-1": 0, "raw_1": 1}
EXPECTED_ARTIFACT_FILES = {
    "selected_pipeline.joblib",
    "classifier.joblib",
    "preprocessor.joblib",
    "outer_training_source_row_indices.csv",
}


class Phase8HoldoutEvaluationError(RuntimeError):
    """Raised when one-time holdout evaluation safeguards are not satisfied."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_name(path.name + ".tmp")
    with temp_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp_path.replace(path)


def _require_evaluation_authorization(*, allow_holdout_evaluation: bool, confirmation: str | None) -> None:
    if (
        allow_holdout_evaluation is not True
        or confirmation != PHASE8_AUTH_TOKEN
        or os.environ.get(PHASE8_AUTH_ENV) != PHASE8_AUTH_TOKEN
    ):
        raise Phase8HoldoutEvaluationError(
            "Phase 8 holdout evaluation is disabled. It requires --allow-holdout-evaluation, "
            f"--confirmation {PHASE8_AUTH_TOKEN}, and current-session environment variable "
            f"{PHASE8_AUTH_ENV}={PHASE8_AUTH_TOKEN}."
        )


def _git_value(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True,
            check=False, timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise Phase8HoldoutEvaluationError("Unable to verify Git provenance before holdout evaluation.") from exc
    if result.returncode != 0:
        raise Phase8HoldoutEvaluationError(
            f"Git provenance command failed: git {' '.join(args)}. {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _require_clean_development_checkout(root: Path) -> tuple[str, str]:
    branch = _git_value(root, "branch", "--show-current")
    if branch != "v2-development":
        raise Phase8HoldoutEvaluationError(
            f"Holdout evaluation is permitted only on v2-development; found {branch!r}."
        )
    status = _git_value(root, "status", "--porcelain", "--untracked-files=normal")
    if status:
        raise Phase8HoldoutEvaluationError(
            "Git working tree must be clean before the one-time evaluation. Commit the reviewed Phase 8 code "
            "first; do not add generated model artifacts to Git."
        )
    commit = _git_value(root, "rev-parse", "HEAD")
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit.lower()):
        raise Phase8HoldoutEvaluationError("Unable to verify a full Git commit SHA.")
    return branch, commit


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase8HoldoutEvaluationError(f"{label} is unreadable or invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise Phase8HoldoutEvaluationError(f"{label} must contain a JSON object.")
    return value


def _resolve_fit_report(root: Path, supplied_path: str | Path) -> tuple[Path, Path]:
    raw = Path(supplied_path)
    candidate = raw if raw.is_absolute() else root / raw
    try:
        fit_report_path = candidate.resolve(strict=True)
        artifacts_root = (root / "Artifacts").resolve(strict=True)
        fit_dir = fit_report_path.parent
        fit_dir.relative_to(artifacts_root)
    except (OSError, ValueError) as exc:
        raise Phase8HoldoutEvaluationError("Phase 7 fit report must be inside a project-controlled Artifacts/phase7_final_fit_... directory.") from exc
    if not fit_report_path.is_file() or fit_report_path.name != "fit_report.json":
        raise Phase8HoldoutEvaluationError("Expected an existing fit_report.json file.")
    if not fit_dir.name.startswith("phase7_final_fit_"):
        raise Phase8HoldoutEvaluationError("Fit report must be under a phase7_final_fit_<run_id> directory.")
    return fit_report_path, fit_dir


def _resolve_selection_report(root: Path, supplied_path: str) -> Path:
    candidate = root / Path(supplied_path)
    try:
        resolved = candidate.resolve(strict=True)
        evaluation_root = (root / "evaluation_artifacts").resolve(strict=True)
        resolved.relative_to(evaluation_root)
    except (OSError, ValueError) as exc:
        raise Phase8HoldoutEvaluationError("The referenced selection report must exist inside evaluation_artifacts/.") from exc
    if not resolved.is_file() or resolved.name != "selection_report.json":
        raise Phase8HoldoutEvaluationError("The referenced Phase 6 file must be selection_report.json.")
    return resolved


def _verify_protected_legacy_artifacts(root: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    for filename, expected in PROTECTED_ARTIFACT_SHA256.items():
        path = root / "final_model" / filename
        if not path.is_file():
            raise Phase8HoldoutEvaluationError(f"Protected legacy artifact is missing: {path}")
        current = sha256_file(path)
        if current != expected:
            raise Phase8HoldoutEvaluationError(f"Protected legacy artifact hash mismatch for {filename}; stopping.")
        observed[filename] = current
    return observed


def _validate_fit_and_selection_reports(
    root: Path,
    fit_report_path: Path,
    fit_dir: Path,
    current_hashes: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    fit = _read_json_object(fit_report_path, "Phase 7 fit report")
    required_values = {
        "phase": 7,
        "run_id": fit_dir.name.removeprefix("phase7_final_fit_"),
        "selected_candidate_by_inner_cv_only": EXPECTED_SELECTED_CANDIDATE,
        "outer_training_rows_fitted": EXPECTED_OUTER_TRAIN_ROWS,
        "reserved_holdout_rows": EXPECTED_RESERVED_HOLDOUT_ROWS,
        "fitted_estimator_saved": True,
        "holdout_rows_passed_to_estimator": False,
        "holdout_predictions_generated": False,
        "holdout_scored": False,
        "training_metrics_computed": False,
        "final_model_directory_modified": False,
        "pipeline_artifact_path": "selected_pipeline.joblib",
        "target_mapping": EXPECTED_TARGET_MAPPING,
    }
    for key, expected in required_values.items():
        if fit.get(key) != expected:
            raise Phase8HoldoutEvaluationError(
                f"Phase 7 fit report field {key!r} must be {expected!r}; found {fit.get(key)!r}."
            )

    if fit.get("git_branch") != "v2-development":
        raise Phase8HoldoutEvaluationError("Phase 7 fit report was not generated on v2-development.")
    implementation_commit = fit.get("implementation_git_commit")
    if not isinstance(implementation_commit, str) or len(implementation_commit) != 40 or any(
        char not in "0123456789abcdef" for char in implementation_commit.lower()
    ):
        raise Phase8HoldoutEvaluationError("Phase 7 fit report has no valid full implementation Git commit SHA.")

    for report_key, hash_key in (
        ("dataset_sha256", "dataset"),
        ("schema_sha256", "schema"),
        ("split_manifest_sha256", "split_manifest"),
        ("inner_cv_assignment_sha256", "inner_cv_assignment"),
    ):
        if fit.get(report_key) != current_hashes.get(hash_key):
            raise Phase8HoldoutEvaluationError(f"Phase 7 fit report {report_key} differs from the frozen Phase 2 protocol.")

    parameters = fit.get("classifier_parameters")
    expected_parameters = {
        "learning_rate": float(EXPECTED_SELECTED_PARAMETERS["classifier__learning_rate"]),
        "n_estimators": int(EXPECTED_SELECTED_PARAMETERS["classifier__n_estimators"]),
        "subsample": float(EXPECTED_SELECTED_PARAMETERS["classifier__subsample"]),
        "random_state": EXPECTED_RANDOM_STATE,
    }
    if parameters != expected_parameters:
        raise Phase8HoldoutEvaluationError("Phase 7 classifier parameters do not match the approved Phase 6 winner.")
    if fit.get("protected_final_model_hashes_before_and_after") != PROTECTED_ARTIFACT_SHA256:
        raise Phase8HoldoutEvaluationError("Phase 7 report does not prove the protected final_model hashes were preserved.")
    _verify_protected_legacy_artifacts(root)

    selection_rel = fit.get("selection_report_path")
    if not isinstance(selection_rel, str) or not selection_rel.replace("\\", "/").startswith("evaluation_artifacts/"):
        raise Phase8HoldoutEvaluationError("Phase 7 report has an invalid selection_report_path.")
    selection_path = _resolve_selection_report(root, selection_rel)
    expected_selection_hash = fit.get("selection_report_sha256")
    if not isinstance(expected_selection_hash, str) or sha256_file(selection_path) != expected_selection_hash:
        raise Phase8HoldoutEvaluationError("Phase 6 selection report hash differs from the Phase 7 fit report.")
    selection = _read_json_object(selection_path, "Phase 6 selection report")
    if selection.get("run_id") != fit.get("selection_run_id"):
        raise Phase8HoldoutEvaluationError("Selection run ID differs from the Phase 7 fit report.")
    if selection.get("project_git_commit") != fit.get("selection_source_git_commit"):
        raise Phase8HoldoutEvaluationError("Selection source commit differs from the Phase 7 fit report.")
    if selection.get("selected_candidate_by_inner_cv_only") != EXPECTED_SELECTED_CANDIDATE:
        raise Phase8HoldoutEvaluationError("Phase 6 report's selected candidate differs from the approved winner.")
    if selection.get("primary_selection_metric") != EXPECTED_SELECTION_METRIC:
        raise Phase8HoldoutEvaluationError("Phase 6 report's selection metric differs from the frozen rule.")
    for report_key, hash_key in (
        ("dataset_sha256", "dataset"),
        ("schema_sha256", "schema"),
        ("split_manifest_sha256", "split_manifest"),
        ("inner_cv_assignment_sha256", "inner_cv_assignment"),
    ):
        if selection.get(report_key) != current_hashes.get(hash_key):
            raise Phase8HoldoutEvaluationError(f"Phase 6 report {report_key} differs from the frozen Phase 2 protocol.")
    for flag in ("holdout_scored", "holdout_predictions_generated", "final_estimator_fitted", "model_artifacts_saved"):
        if selection.get(flag) is not False:
            raise Phase8HoldoutEvaluationError(f"Phase 6 report must record {flag}=false before holdout evaluation.")

    candidates = selection.get("candidate_results")
    chosen = candidates.get(EXPECTED_SELECTED_CANDIDATE) if isinstance(candidates, dict) else None
    if not isinstance(candidates, dict) or not isinstance(chosen, dict):
        raise Phase8HoldoutEvaluationError("Phase 6 report does not include the selected candidate's results.")
    ranked_scores: list[tuple[str, float]] = []
    for candidate_name, details in candidates.items():
        if not isinstance(details, dict):
            raise Phase8HoldoutEvaluationError(f"Phase 6 candidate {candidate_name!r} has malformed details.")
        try:
            candidate_score = float(details["best_mean_inner_cv_phishing_f1"])
        except (KeyError, TypeError, ValueError) as exc:
            raise Phase8HoldoutEvaluationError(f"Phase 6 candidate {candidate_name!r} has no valid CV F1.") from exc
        if not np.isfinite(candidate_score) or not 0.0 <= candidate_score <= 1.0:
            raise Phase8HoldoutEvaluationError(f"Phase 6 candidate {candidate_name!r} has an invalid CV F1.")
        ranked_scores.append((str(candidate_name), candidate_score))
    ranked_scores.sort(key=lambda item: (-item[1], item[0]))
    if not ranked_scores or ranked_scores[0][0] != EXPECTED_SELECTED_CANDIDATE:
        raise Phase8HoldoutEvaluationError("Phase 6 results no longer rank Gradient Boosting first under the frozen rule.")
    if chosen.get("best_parameters") != EXPECTED_SELECTED_PARAMETERS:
        raise Phase8HoldoutEvaluationError("Phase 6 selected parameters differ from the fitted pipeline configuration.")
    if chosen.get("metric") != "f1_score(pos_label=0, zero_division=0)":
        raise Phase8HoldoutEvaluationError("Phase 6 selected metric is not phishing-positive F1.")
    try:
        selected_score = float(chosen["best_mean_inner_cv_phishing_f1"])
    except (KeyError, TypeError, ValueError) as exc:
        raise Phase8HoldoutEvaluationError("Phase 6 report lacks a valid selected mean CV F1.") from exc
    if not np.isfinite(selected_score) or abs(selected_score - float(fit.get("selected_inner_cv_mean_phishing_f1", -1))) > 1e-12:
        raise Phase8HoldoutEvaluationError("Phase 7 fit report's selected CV score differs from Phase 6 evidence.")

    artifact_hashes = fit.get("artifacts_sha256")
    if not isinstance(artifact_hashes, dict) or set(artifact_hashes) != EXPECTED_ARTIFACT_FILES:
        raise Phase8HoldoutEvaluationError("Phase 7 report does not list exactly the four expected artifact hashes.")
    for filename in sorted(EXPECTED_ARTIFACT_FILES):
        artifact_path = (fit_dir / filename).resolve(strict=True)
        try:
            artifact_path.relative_to(fit_dir.resolve(strict=True))
        except ValueError as exc:
            raise Phase8HoldoutEvaluationError("A Phase 7 artifact path escapes the trusted output directory.") from exc
        if not artifact_path.is_file() or sha256_file(artifact_path) != artifact_hashes.get(filename):
            raise Phase8HoldoutEvaluationError(f"Phase 7 artifact hash verification failed for {filename}.")
    return fit, selection, fit_dir / "selected_pipeline.joblib", selection_path


def _training_row_index_digest(indices: Iterable[int]) -> str:
    serialized = json.dumps([int(value) for value in indices], separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _verify_training_row_provenance(fit_dir: Path, fit_report: dict[str, Any], evidence: Any) -> None:
    path = fit_dir / "outer_training_source_row_indices.csv"
    try:
        rows = pd.read_csv(path)
        values = pd.to_numeric(rows["source_row_index"], errors="raise").to_numpy(dtype=np.float64)
    except Exception as exc:
        raise Phase8HoldoutEvaluationError("Phase 7 training-row provenance file is unreadable or malformed.") from exc
    if list(rows.columns) != ["source_row_index"]:
        raise Phase8HoldoutEvaluationError("Training-row provenance file must contain only source_row_index.")
    if len(values) != EXPECTED_OUTER_TRAIN_ROWS or not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise Phase8HoldoutEvaluationError("Training-row provenance must contain exactly 8,843 integer row indices.")
    values_int = values.astype(np.int64)
    if len(np.unique(values_int)) != EXPECTED_OUTER_TRAIN_ROWS:
        raise Phase8HoldoutEvaluationError("Training-row provenance contains duplicate source row indices.")
    expected_indices = evidence.row_manifest.loc[
        evidence.row_manifest["partition"] == "train", "source_row_index"
    ].astype(np.int64).to_numpy()
    if not np.array_equal(values_int, expected_indices):
        raise Phase8HoldoutEvaluationError("Saved model's training-row provenance differs from the frozen outer-training partition.")
    if fit_report.get("training_source_row_indices_sha256") != _training_row_index_digest(values_int):
        raise Phase8HoldoutEvaluationError("Training-row index digest differs from the Phase 7 fit report.")


def _require_commit_is_ancestor(root: Path, ancestor: str, descendant: str) -> None:
    try:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=root, capture_output=True, text=True, check=False, timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise Phase8HoldoutEvaluationError("Unable to verify Phase 7 implementation commit ancestry.") from exc
    if result.returncode != 0:
        raise Phase8HoldoutEvaluationError(
            "Phase 7 model's implementation commit is not an ancestor of the current v2-development checkout."
        )


def _check_no_prior_evaluation(root: Path) -> Path:
    evaluation_root = root / "evaluation_artifacts"
    guard_path = evaluation_root / PHASE8_GUARD_FILENAME
    if guard_path.exists():
        raise Phase8HoldoutEvaluationError(
            f"A Phase 8 one-time guard already exists at {guard_path}. Do not rerun or delete it automatically; inspect its status first."
        )
    prior_reports = sorted(path for path in evaluation_root.glob(f"{PHASE8_OUTPUT_PREFIX}*") if path.is_dir())
    if prior_reports:
        raise Phase8HoldoutEvaluationError(
            "A Phase 8 evaluation output directory already exists. The reserved holdout must not be evaluated again."
        )
    return guard_path


def _create_one_time_guard(path: Path, payload: dict[str, Any]) -> None:
    """Create the durable at-most-once marker atomically; never auto-remove it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(str(path), flags, 0o600)
    except FileExistsError as exc:
        raise Phase8HoldoutEvaluationError("The one-time holdout-evaluation guard was created by another process; refusing to continue.") from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        # No prediction call has occurred yet, so it is safe to remove a partial marker.
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def metrics_from_predictions(y_true: Iterable[int], y_pred: Iterable[int]) -> dict[str, Any]:
    """Compute predeclared aggregate metrics; class 0 (phishing) is positive."""
    true = np.asarray(list(y_true), dtype=np.int64)
    pred = np.asarray(list(y_pred), dtype=np.int64)
    if true.ndim != 1 or pred.ndim != 1 or len(true) == 0 or len(true) != len(pred):
        raise Phase8HoldoutEvaluationError("True labels and predictions must be non-empty one-dimensional arrays of equal length.")
    if not set(np.unique(true).tolist()).issubset({0, 1}) or set(np.unique(true).tolist()) != {0, 1}:
        raise Phase8HoldoutEvaluationError("Evaluation labels must include exactly encoded classes 0 and 1.")
    if not set(np.unique(pred).tolist()).issubset({0, 1}):
        raise Phase8HoldoutEvaluationError("Predictions contain labels outside encoded classes 0 and 1.")

    p0, r0, f0, _ = precision_recall_fscore_support(true, pred, average="binary", pos_label=0, zero_division=0)
    p1, r1, f1, _ = precision_recall_fscore_support(true, pred, average="binary", pos_label=1, zero_division=0)
    # Class ordering is fixed: rows=true and columns=predicted, both [phishing 0, legitimate 1].
    cm = np.zeros((2, 2), dtype=np.int64)
    for actual, predicted in zip(true, pred):
        cm[int(actual), int(predicted)] += 1
    return {
        "accuracy": float(accuracy_score(true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "phishing_class_0": {
            "precision": float(p0), "recall": float(r0), "f1": float(f0),
            "support": int(np.sum(true == 0)),
        },
        "legitimate_class_1": {
            "precision": float(p1), "recall": float(r1), "f1": float(f1),
            "support": int(np.sum(true == 1)),
        },
        "confusion_matrix": {
            "label_order": ["phishing (0)", "legitimate (1)"],
            "rows_are": "true labels",
            "columns_are": "predicted labels",
            "counts": cm.tolist(),
            "phishing_positive_counts": {
                "true_positive": int(cm[0, 0]),
                "false_negative": int(cm[0, 1]),
                "false_positive": int(cm[1, 0]),
                "true_negative": int(cm[1, 1]),
            },
        },
    }


def _holdout_index_digest(indices: Iterable[int]) -> str:
    serialized = json.dumps([int(index) for index in indices], separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def run_phase8_holdout_evaluation(
    fit_report_path: str | Path,
    root: str | Path | None = None,
    *,
    allow_holdout_evaluation: bool = False,
    confirmation: str | None = None,
) -> Path:
    """Evaluate the verified Phase 7 pipeline exactly once on the frozen holdout.

    This function makes one call to ``pipeline.predict`` only after all protocol,
    report, artifact, and authorization checks pass. It never fits, tunes, or
    writes row-level predictions. A persistent guard prevents a second run.
    """
    _require_evaluation_authorization(
        allow_holdout_evaluation=allow_holdout_evaluation, confirmation=confirmation
    )
    project_root = Path(root or PROJECT_ROOT).resolve()
    branch, evaluation_commit = _require_clean_development_checkout(project_root)
    guard_path = _check_no_prior_evaluation(project_root)

    try:
        evidence = verify_phase2_protocol(project_root)
    except Phase6ProtocolError as exc:
        raise Phase8HoldoutEvaluationError(f"Frozen Phase 2 protocol verification failed: {exc}") from exc
    if evidence.summary.get("status") != "PASS":
        raise Phase8HoldoutEvaluationError("Frozen Phase 2 preflight did not pass.")
    if evidence.summary.get("outer_shared_feature_groups") != 0:
        raise Phase8HoldoutEvaluationError("Feature groups overlap between outer training and holdout partitions.")
    if evidence.summary.get("outer_train_rows") != EXPECTED_OUTER_TRAIN_ROWS or evidence.summary.get("reserved_holdout_rows") != EXPECTED_RESERVED_HOLDOUT_ROWS:
        raise Phase8HoldoutEvaluationError("Frozen outer partition row counts are not the expected values.")

    fit_path, fit_dir = _resolve_fit_report(project_root, fit_report_path)
    fit, selection, pipeline_path, selection_path = _validate_fit_and_selection_reports(
        project_root, fit_path, fit_dir, evidence.hashes
    )
    _require_commit_is_ancestor(project_root, fit["implementation_git_commit"], evaluation_commit)
    _verify_training_row_provenance(fit_dir, fit, evidence)
    protected_before = _verify_protected_legacy_artifacts(project_root)

    # Only trusted, hash-verified artifacts in the project-controlled Phase 7 directory are loaded.
    import joblib

    try:
        pipeline = joblib.load(pipeline_path)
    except Exception as exc:
        raise Phase8HoldoutEvaluationError("Unable to load the hash-verified Phase 7 pipeline artifact.") from exc
    if not isinstance(pipeline, Pipeline):
        raise Phase8HoldoutEvaluationError("The verified Phase 7 artifact is not an sklearn Pipeline.")
    if [name for name, _ in pipeline.steps] != ["preprocessor", "classifier"]:
        raise Phase8HoldoutEvaluationError("Pipeline steps differ from the reviewed Phase 7 pipeline structure.")
    if not isinstance(pipeline.named_steps["preprocessor"], KNNImputer):
        raise Phase8HoldoutEvaluationError("Pipeline preprocessor is not the expected KNNImputer.")
    preprocessor_params = pipeline.named_steps["preprocessor"].get_params(deep=False)
    if preprocessor_params.get("n_neighbors") != 3 or preprocessor_params.get("weights") != "uniform":
        raise Phase8HoldoutEvaluationError("Pipeline KNNImputer parameters differ from the Phase 7 fit report.")
    classifier = pipeline.named_steps["classifier"]
    if not isinstance(classifier, GradientBoostingClassifier):
        raise Phase8HoldoutEvaluationError("Pipeline classifier is not GradientBoostingClassifier.")
    expected_classifier_params = {
        "learning_rate": 0.1, "n_estimators": 256, "subsample": 0.9,
        "random_state": EXPECTED_RANDOM_STATE,
    }
    observed_params = {key: classifier.get_params(deep=False).get(key) for key in expected_classifier_params}
    if observed_params != expected_classifier_params:
        raise Phase8HoldoutEvaluationError("Loaded classifier parameters differ from the frozen Phase 6 selection.")
    if int(getattr(classifier, "n_features_in_", -1)) != EXPECTED_FEATURE_COUNT:
        raise Phase8HoldoutEvaluationError("Loaded classifier feature width differs from the frozen schema.")
    if set(np.asarray(getattr(classifier, "classes_", [])).astype(int).tolist()) != {0, 1}:
        raise Phase8HoldoutEvaluationError("Loaded classifier classes differ from encoded labels 0 and 1.")

    # The verified manifest exposes row partitions but intentionally not target labels.
    # Read source rows by their exact Phase 2 row indices and confirm hashes again.
    if sha256_file(evidence.dataset_path) != EXPECTED_SHA256["dataset"]:
        raise Phase8HoldoutEvaluationError("Canonical dataset hash changed immediately before holdout construction.")
    try:
        dataset = pd.read_csv(evidence.dataset_path, low_memory=False)
    except Exception as exc:
        raise Phase8HoldoutEvaluationError("Unable to reread the hash-verified canonical CSV.") from exc
    if list(dataset.columns) != [*evidence.feature_columns, evidence.target_column] or len(dataset) != 11_055:
        raise Phase8HoldoutEvaluationError("Canonical CSV schema or row count changed after protocol verification.")
    manifest = evidence.row_manifest.sort_values("source_row_index").reset_index(drop=True)
    holdout_rows = manifest.loc[manifest["partition"] == "holdout", "source_row_index"].astype(np.int64).to_numpy()
    if len(holdout_rows) != EXPECTED_RESERVED_HOLDOUT_ROWS or len(np.unique(holdout_rows)) != EXPECTED_RESERVED_HOLDOUT_ROWS:
        raise Phase8HoldoutEvaluationError("Frozen holdout row indices are invalid or incomplete.")
    if holdout_rows.min() < 0 or holdout_rows.max() >= len(dataset):
        raise Phase8HoldoutEvaluationError("Frozen holdout row index is outside the canonical dataset.")

    X_holdout = dataset.iloc[holdout_rows].loc[:, list(evidence.feature_columns)].copy()
    raw_y = pd.to_numeric(dataset.iloc[holdout_rows][evidence.target_column], errors="raise").astype(np.int64)
    y_holdout = raw_y.map({-1: 0, 1: 1}).astype(np.int64).to_numpy()
    if set(np.unique(y_holdout).tolist()) != {0, 1} or len(X_holdout) != EXPECTED_RESERVED_HOLDOUT_ROWS:
        raise Phase8HoldoutEvaluationError("Holdout features or mapped labels fail frozen protocol checks.")
    if sha256_file(evidence.dataset_path) != EXPECTED_SHA256["dataset"]:
        raise Phase8HoldoutEvaluationError("Canonical dataset hash changed after holdout rows were read.")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    output_root = project_root / "evaluation_artifacts"
    output_dir = output_root / f"{PHASE8_OUTPUT_PREFIX}{run_id}"
    marker_payload: dict[str, Any] = {
        "schema_version": "1.0",
        "phase": 8,
        "status": "started_do_not_rerun_automatically",
        "created_utc": _utc_now(),
        "run_id": run_id,
        "evaluation_git_branch": branch,
        "evaluation_git_commit": evaluation_commit,
        "dataset_sha256": evidence.hashes["dataset"],
        "split_manifest_sha256": evidence.hashes["split_manifest"],
        "fit_report_sha256": sha256_file(fit_path),
        "selected_pipeline_sha256": sha256_file(pipeline_path),
        "note": "Permanent at-most-once guard. If interrupted, inspect execution evidence before considering any recovery; do not delete this marker automatically.",
    }
    _create_one_time_guard(guard_path, marker_payload)
    try:
        output_dir.mkdir(parents=False, exist_ok=False)
    except Exception as exc:
        raise Phase8HoldoutEvaluationError(
            f"One-time guard was created at {guard_path}, but output directory creation failed. Do not rerun automatically."
        ) from exc

    # THE ONLY prediction call in this module. There is no fitting, thresholding,
    # tuning, test-set model selection, or second model invocation in this run.
    try:
        y_pred = np.asarray(pipeline.predict(X_holdout), dtype=np.int64)
        if y_pred.shape != (EXPECTED_RESERVED_HOLDOUT_ROWS,):
            raise Phase8HoldoutEvaluationError("Pipeline returned an unexpected number or shape of predictions.")
        metrics = metrics_from_predictions(y_holdout, y_pred)
        if sha256_file(evidence.dataset_path) != EXPECTED_SHA256["dataset"]:
            raise Phase8HoldoutEvaluationError("Canonical dataset hash changed during the evaluation pass.")
        protected_after = _verify_protected_legacy_artifacts(project_root)
        if protected_before != protected_after:
            raise Phase8HoldoutEvaluationError("Protected legacy artifact hashes changed during evaluation.")
    except Exception:
        # Marker intentionally remains: we cannot safely assume an interrupted run did not score the holdout.
        raise

    report = {
        "schema_version": "1.0",
        "phase": 8,
        "run_id": run_id,
        "created_utc": _utc_now(),
        "evaluation_git_branch": branch,
        "evaluation_git_commit": evaluation_commit,
        "evaluation_type": "one-time frozen reserved-holdout evaluation",
        "selected_candidate_by_inner_cv_only": EXPECTED_SELECTED_CANDIDATE,
        "selected_inner_cv_mean_phishing_f1": float(fit["selected_inner_cv_mean_phishing_f1"]),
        "classifier_parameters": expected_classifier_params,
        "training_fit_report_path": str(fit_path.relative_to(project_root)).replace("\\", "/"),
        "training_fit_report_sha256": sha256_file(fit_path),
        "training_run_id": fit["run_id"],
        "selection_report_path": str(selection_path.relative_to(project_root)).replace("\\", "/"),
        "selection_report_sha256": sha256_file(selection_path),
        "selection_run_id": selection["run_id"],
        "selected_pipeline_path": str(pipeline_path.relative_to(project_root)).replace("\\", "/"),
        "selected_pipeline_sha256": sha256_file(pipeline_path),
        "all_phase7_artifact_sha256": fit["artifacts_sha256"],
        "dataset_sha256": evidence.hashes["dataset"],
        "schema_sha256": evidence.hashes["schema"],
        "split_manifest_sha256": evidence.hashes["split_manifest"],
        "inner_cv_assignment_sha256": evidence.hashes["inner_cv_assignment"],
        "outer_training_rows": EXPECTED_OUTER_TRAIN_ROWS,
        "reserved_holdout_rows_evaluated": int(len(holdout_rows)),
        "holdout_source_row_indices_sha256": _holdout_index_digest(holdout_rows),
        "outer_shared_feature_groups": int(evidence.summary["outer_shared_feature_groups"]),
        "prediction_pass_count": 1,
        "holdout_predictions_generated": True,
        "holdout_scored": True,
        "holdout_used_for_model_selection_or_tuning": False,
        "retraining_performed": False,
        "final_model_directory_modified": False,
        "protected_final_model_hashes_before_and_after": protected_after,
        "target_mapping": EXPECTED_TARGET_MAPPING,
        "selection_metric": "phishing F1; positive label is encoded class 0",
        "metrics": metrics,
        "row_level_predictions_saved": False,
        "note": "These are final results for this frozen model and holdout. Do not tune or retrain against this holdout and continue reporting it as untouched.",
    }
    report_path = output_dir / "holdout_evaluation_report.json"
    _write_json_atomic(report_path, report)
    marker_payload["status"] = "completed_holdout_evaluation_do_not_rerun"
    marker_payload["completed_utc"] = _utc_now()
    marker_payload["report_path"] = str(report_path.relative_to(project_root)).replace("\\", "/")
    marker_payload["report_sha256"] = sha256_file(report_path)
    _write_json_atomic(guard_path, marker_payload)
    return report_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-report", required=True, help="Path to the verified Phase 7 fit_report.json under Artifacts/.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help="Project root directory.")
    parser.add_argument("--allow-holdout-evaluation", action="store_true", help="Acknowledge the one-time reserved-holdout evaluation.")
    parser.add_argument("--confirmation", required=True, help="Must equal the separately approved Phase 8 confirmation token.")
    args = parser.parse_args(argv)
    try:
        report_path = run_phase8_holdout_evaluation(
            args.fit_report,
            args.root,
            allow_holdout_evaluation=args.allow_holdout_evaluation,
            confirmation=args.confirmation,
        )
    except (Phase8HoldoutEvaluationError, Phase6ProtocolError) as exc:
        print(f"PHASE 8 EVALUATION ABORTED: {exc}", file=sys.stderr)
        return 2
    print(str(report_path))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
