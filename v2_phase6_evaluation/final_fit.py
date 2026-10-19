"""Explicitly authorized final fit on outer-training rows only.

This module does not fit an estimator on import. Its runner requires an explicit
Phase 7 confirmation argument, a matching one-session environment variable, a
clean v2-development checkout, an approved selection report, and the unchanged
protected V1-era model/preprocessor hashes.

The fitted pipeline and provenance report are written only under the existing
Git-ignored ``Artifacts/phase7_final_fit_<run_id>/`` directory. Nothing is
written to ``final_model/``. This code never calls predict/score and never
passes outer-holdout rows or labels to an estimator.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from .protocol import EXPECTED_SHA256, PROJECT_ROOT, sha256_file
from .selection import prepare_training_only_selection_plan

PHASE7_AUTH_ENV = "NETWORK_SECURITY_PHASE7_ALLOW_FINAL_FIT"
PHASE7_AUTH_TOKEN = "I_APPROVE_PHASE7_TRAINING_ONLY_FINAL_FIT"
EXPECTED_SELECTED_CANDIDATE = "Gradient Boosting"
EXPECTED_SELECTED_PARAMETERS = {
    "classifier__learning_rate": 0.1,
    "classifier__n_estimators": 256,
    "classifier__subsample": 0.9,
}
EXPECTED_RANDOM_STATE = 43
EXPECTED_OUTER_TRAIN_ROWS = 8_843
EXPECTED_RESERVED_HOLDOUT_ROWS = 2_212
EXPECTED_INNER_FOLDS = 5
EXPECTED_SELECTION_METRIC = "phishing F1; positive label is encoded class 0"
PROTECTED_ARTIFACT_SHA256 = {
    "model.pkl": "c64bc2259d3ab5d73c89165fe3f0ab00acb73283ca5f8d7a2320e0a2116ce1f1",
    "preprocessor.pkl": "6c25d32d895d34d565418d1924ef13f7d21373f0a29dc8adc6f80d337cba2a39",
}
OUTPUT_PARENT = "Artifacts"  # Existing .gitignore rule excludes this directory.


class Phase7FinalFitError(RuntimeError):
    """Raised when Phase 7 provenance or safety requirements are not satisfied."""


def _require_fit_authorization(*, allow_fitting: bool, confirmation: str | None) -> None:
    """Require explicit approval in both the function call and current CMD session."""
    if (
        allow_fitting is not True
        or confirmation != PHASE7_AUTH_TOKEN
        or os.environ.get(PHASE7_AUTH_ENV) != PHASE7_AUTH_TOKEN
    ):
        raise Phase7FinalFitError(
            "Phase 7 fitting is disabled. It requires allow_fitting=True, "
            f"confirmation={PHASE7_AUTH_TOKEN!r}, and current-session environment "
            f"variable {PHASE7_AUTH_ENV}={PHASE7_AUTH_TOKEN!r}."
        )


def _git_value(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True,
            check=False, timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise Phase7FinalFitError("Unable to verify Git provenance before fitting.") from exc
    if result.returncode != 0:
        raise Phase7FinalFitError(
            f"Git provenance command failed: git {' '.join(args)}. "
            f"Details: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _require_clean_development_checkout(root: Path) -> tuple[str, str]:
    branch = _git_value(root, "branch", "--show-current")
    if branch != "v2-development":
        raise Phase7FinalFitError(
            f"Final fitting is permitted only on v2-development; current branch is {branch!r}."
        )
    dirty = _git_value(root, "status", "--porcelain", "--untracked-files=normal")
    if dirty:
        raise Phase7FinalFitError(
            "Git working tree must be clean before final fitting so the run is traceable. "
            "Commit the approved Phase 7 code first; do not include model artifacts in Git."
        )
    commit = _git_value(root, "rev-parse", "HEAD")
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit.lower()):
        raise Phase7FinalFitError("Unable to determine a full Git commit SHA before fitting.")
    return branch, commit


def _resolve_selection_report(root: Path, supplied_path: str | Path) -> Path:
    raw = Path(supplied_path)
    candidate = raw if raw.is_absolute() else root / raw
    try:
        resolved = candidate.resolve(strict=True)
        allowed_root = (root / "evaluation_artifacts").resolve(strict=True)
        resolved.relative_to(allowed_root)
    except (OSError, ValueError) as exc:
        raise Phase7FinalFitError(
            "Selection report must be an existing file inside evaluation_artifacts/."
        ) from exc
    if not resolved.is_file() or resolved.name != "selection_report.json":
        raise Phase7FinalFitError("Expected an existing selection_report.json file.")
    return resolved


def _load_and_validate_selection_report(
    report_path: Path,
    *,
    current_protocol_hashes: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        selection_report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase7FinalFitError("Selection report is unreadable or invalid JSON.") from exc
    if not isinstance(selection_report, dict):
        raise Phase7FinalFitError("Selection report root must be a JSON object.")

    expected_report_hashes = {
        "dataset_sha256": "dataset",
        "schema_sha256": "schema",
        "split_manifest_sha256": "split_manifest",
        "inner_cv_assignment_sha256": "inner_cv_assignment",
    }
    for report_key, hash_key in expected_report_hashes.items():
        if selection_report.get(report_key) != current_protocol_hashes.get(hash_key):
            raise Phase7FinalFitError(
                f"Selection report {report_key} does not match the current frozen protocol."
            )

    expected_scalars = {
        "outer_training_rows": EXPECTED_OUTER_TRAIN_ROWS,
        "reserved_holdout_rows": EXPECTED_RESERVED_HOLDOUT_ROWS,
        "inner_cv_folds": EXPECTED_INNER_FOLDS,
        "primary_selection_metric": EXPECTED_SELECTION_METRIC,
        "selected_candidate_by_inner_cv_only": EXPECTED_SELECTED_CANDIDATE,
    }
    for key, expected in expected_scalars.items():
        if selection_report.get(key) != expected:
            raise Phase7FinalFitError(
                f"Selection report {key} must be {expected!r}; found {selection_report.get(key)!r}."
            )

    for flag in ("holdout_scored", "holdout_predictions_generated", "final_estimator_fitted", "model_artifacts_saved"):
        if selection_report.get(flag) is not False:
            raise Phase7FinalFitError(
                f"Selection report must explicitly record {flag}=false before Phase 7."
            )

    candidate_results = selection_report.get("candidate_results")
    if not isinstance(candidate_results, dict) or EXPECTED_SELECTED_CANDIDATE not in candidate_results:
        raise Phase7FinalFitError("Selection report is missing the selected candidate results.")

    # Verify that Gradient Boosting is still the top candidate by the recorded
    # metric, with the same deterministic tie-break rule as Phase 6.
    ranked: list[tuple[str, float]] = []
    for candidate, details in candidate_results.items():
        if not isinstance(details, dict):
            raise Phase7FinalFitError(f"Candidate result {candidate!r} must be an object.")
        try:
            score = float(details["best_mean_inner_cv_phishing_f1"])
        except (KeyError, TypeError, ValueError) as exc:
            raise Phase7FinalFitError(f"Candidate {candidate!r} has no valid mean inner-CV F1.") from exc
        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise Phase7FinalFitError(f"Candidate {candidate!r} has an invalid mean F1 score.")
        ranked.append((str(candidate), score))
    ranked.sort(key=lambda item: (-item[1], item[0]))
    if not ranked or ranked[0][0] != EXPECTED_SELECTED_CANDIDATE:
        raise Phase7FinalFitError(
            "The selection report's recorded scores no longer rank Gradient Boosting first."
        )

    selected_details = candidate_results[EXPECTED_SELECTED_CANDIDATE]
    if selected_details.get("metric") != "f1_score(pos_label=0, zero_division=0)":
        raise Phase7FinalFitError("Selected candidate metric differs from the frozen phishing-F1 metric.")
    actual_params = selected_details.get("best_parameters")
    if not isinstance(actual_params, dict) or actual_params != EXPECTED_SELECTED_PARAMETERS:
        raise Phase7FinalFitError(
            "Selected Gradient Boosting parameters differ from the reviewed Phase 6 result."
        )
    if not selection_report.get("project_git_commit"):
        raise Phase7FinalFitError("Selection report does not record its source Git commit.")
    if not selection_report.get("run_id"):
        raise Phase7FinalFitError("Selection report does not record its run ID.")

    return selection_report, selected_details


def _verify_protected_legacy_artifacts(root: Path) -> dict[str, str]:
    final_model_dir = root / "final_model"
    observed: dict[str, str] = {}
    for filename, expected_hash in PROTECTED_ARTIFACT_SHA256.items():
        artifact_path = final_model_dir / filename
        if not artifact_path.is_file():
            raise Phase7FinalFitError(f"Protected legacy artifact is missing: {artifact_path}")
        observed_hash = sha256_file(artifact_path)
        if observed_hash != expected_hash:
            raise Phase7FinalFitError(
                f"Protected legacy artifact hash mismatch for {filename}; stopping before fitting."
            )
        observed[filename] = observed_hash
    return observed


def _training_row_index_digest(source_row_indices: np.ndarray) -> str:
    payload = json.dumps(
        [int(value) for value in source_row_indices], separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def run_phase7_final_fit(
    selection_report_path: str | Path,
    root: str | Path | None = None,
    *,
    allow_fitting: bool = False,
    confirmation: str | None = None,
) -> Path:
    """Fit the selected Gradient Boosting pipeline on outer-training rows only.

    The saved pipeline combines KNNImputer and the selected classifier. Separate
    fitted component files are emitted for transparent artifact inspection. The
    function does not call predict/score, does not use holdout rows/labels as
    estimator input, and never writes to final_model/.
    """
    _require_fit_authorization(allow_fitting=allow_fitting, confirmation=confirmation)
    project_root = Path(root or PROJECT_ROOT).resolve()
    branch, implementation_commit = _require_clean_development_checkout(project_root)
    selection_path = _resolve_selection_report(project_root, selection_report_path)
    _verify_protected_legacy_artifacts(project_root)

    # The plan contains X/y for the 8,843 outer-training rows only. The source
    # reader verifies the frozen manifest but does not expose holdout arrays to
    # this function or to the estimator.
    plan = prepare_training_only_selection_plan(project_root)
    if len(plan.X_train) != EXPECTED_OUTER_TRAIN_ROWS or len(plan.y_train) != EXPECTED_OUTER_TRAIN_ROWS:
        raise Phase7FinalFitError("Final fit requires exactly 8,843 outer-training rows.")
    if len(plan.source_row_indices) != EXPECTED_OUTER_TRAIN_ROWS:
        raise Phase7FinalFitError("Training source-row mapping does not contain 8,843 rows.")
    if len(plan.feature_columns) != 30 or list(plan.X_train.columns) != list(plan.feature_columns):
        raise Phase7FinalFitError("Training feature schema/order differs from the frozen 30-feature protocol.")
    if set(plan.y_train.astype(int).unique().tolist()) != {0, 1}:
        raise Phase7FinalFitError("Outer training target must contain encoded classes 0 and 1.")

    selection_report, selected_details = _load_and_validate_selection_report(
        selection_path, current_protocol_hashes=plan.protocol_hashes
    )

    # Recheck the protected artifacts immediately before fitting to fail closed
    # if another process changed them during preflight.
    protected_before = _verify_protected_legacy_artifacts(project_root)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    artifacts_root = project_root / OUTPUT_PARENT
    final_dir = artifacts_root / f"phase7_final_fit_{run_id}"
    staging_dir = artifacts_root / f".phase7_final_fit_{run_id}.tmp"
    artifacts_root.mkdir(parents=True, exist_ok=True)
    if final_dir.exists() or staging_dir.exists():
        raise Phase7FinalFitError("Generated output directory already exists; refusing to overwrite it.")
    staging_dir.mkdir(parents=False, exist_ok=False)

    try:
        pipeline = Pipeline(
            steps=[
                ("preprocessor", KNNImputer(missing_values=np.nan, n_neighbors=3, weights="uniform")),
                (
                    "classifier",
                    GradientBoostingClassifier(
                        learning_rate=float(EXPECTED_SELECTED_PARAMETERS["classifier__learning_rate"]),
                        n_estimators=int(EXPECTED_SELECTED_PARAMETERS["classifier__n_estimators"]),
                        subsample=float(EXPECTED_SELECTED_PARAMETERS["classifier__subsample"]),
                        random_state=EXPECTED_RANDOM_STATE,
                    ),
                ),
            ]
        )

        # The only estimator fitting call in this module. Its arguments are the
        # verified outer-training data and labels; no holdout objects exist here.
        pipeline.fit(plan.X_train, plan.y_train)

        # Confirm the estimator fitted to the same feature width. No predictions
        # or metrics are computed in this phase.
        if int(pipeline.named_steps["classifier"].n_features_in_) != len(plan.feature_columns):
            raise Phase7FinalFitError("Fitted classifier feature count does not match the frozen schema.")
        if set(np.asarray(pipeline.named_steps["classifier"].classes_).astype(int).tolist()) != {0, 1}:
            raise Phase7FinalFitError("Fitted classifier classes differ from the expected encoded labels.")

        import joblib

        pipeline_path = staging_dir / "selected_pipeline.joblib"
        preprocessor_path = staging_dir / "preprocessor.joblib"
        classifier_path = staging_dir / "classifier.joblib"
        row_indices_path = staging_dir / "outer_training_source_row_indices.csv"
        joblib.dump(pipeline, pipeline_path, compress=3)
        joblib.dump(pipeline.named_steps["preprocessor"], preprocessor_path, compress=3)
        joblib.dump(pipeline.named_steps["classifier"], classifier_path, compress=3)
        pd.DataFrame({"source_row_index": plan.source_row_indices.astype(np.int64)}).to_csv(
            row_indices_path, index=False, lineterminator="\n"
        )

        protected_after = _verify_protected_legacy_artifacts(project_root)
        if protected_before != protected_after:
            raise Phase7FinalFitError("Protected legacy artifact hashes changed during fitting; stopping.")

        artifact_hashes = {
            pipeline_path.name: sha256_file(pipeline_path),
            preprocessor_path.name: sha256_file(preprocessor_path),
            classifier_path.name: sha256_file(classifier_path),
            row_indices_path.name: sha256_file(row_indices_path),
        }
        report = {
            "schema_version": "1.0",
            "phase": 7,
            "run_id": run_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_branch": branch,
            "implementation_git_commit": implementation_commit,
            "selection_run_id": selection_report["run_id"],
            "selection_report_path": str(selection_path.relative_to(project_root)).replace("\\", "/"),
            "selection_report_sha256": sha256_file(selection_path),
            "selection_source_git_commit": selection_report["project_git_commit"],
            "selected_candidate_by_inner_cv_only": EXPECTED_SELECTED_CANDIDATE,
            "selected_inner_cv_mean_phishing_f1": float(selected_details["best_mean_inner_cv_phishing_f1"]),
            "selection_metric": EXPECTED_SELECTION_METRIC,
            "classifier_parameters": {
                "learning_rate": float(EXPECTED_SELECTED_PARAMETERS["classifier__learning_rate"]),
                "n_estimators": int(EXPECTED_SELECTED_PARAMETERS["classifier__n_estimators"]),
                "subsample": float(EXPECTED_SELECTED_PARAMETERS["classifier__subsample"]),
                "random_state": EXPECTED_RANDOM_STATE,
            },
            "preprocessor": {"class": "KNNImputer", "n_neighbors": 3, "weights": "uniform"},
            "dataset_sha256": plan.protocol_hashes["dataset"],
            "schema_sha256": plan.protocol_hashes["schema"],
            "split_manifest_sha256": plan.protocol_hashes["split_manifest"],
            "inner_cv_assignment_sha256": plan.protocol_hashes["inner_cv_assignment"],
            "outer_training_rows_fitted": int(len(plan.X_train)),
            "reserved_holdout_rows": EXPECTED_RESERVED_HOLDOUT_ROWS,
            "training_source_row_indices_sha256": _training_row_index_digest(plan.source_row_indices),
            "target_mapping": {"raw_-1": 0, "raw_1": 1},
            "fitted_estimator_saved": True,
            "pipeline_artifact_path": pipeline_path.name,
            "artifacts_sha256": artifact_hashes,
            "protected_final_model_hashes_before_and_after": protected_after,
            "holdout_rows_passed_to_estimator": False,
            "holdout_predictions_generated": False,
            "holdout_scored": False,
            "training_metrics_computed": False,
            "final_model_directory_modified": False,
            "output_policy": "new ignored Artifacts/phase7_final_fit_<run_id>/ directory only",
            "artifact_format_note": "Joblib/pickle artifacts are trusted-local files; only load artifacts from this project-controlled output directory.",
        }
        _write_json(staging_dir / "fit_report.json", report)
        staging_dir.rename(final_dir)
    except Exception:
        # Never leave a partial output directory looking like a successful run.
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    return final_dir / "fit_report.json"


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Fit the approved Gradient Boosting pipeline on Phase 2 outer-training rows only."
    )
    parser.add_argument("--selection-report", required=True, help="Existing Phase 6 selection_report.json")
    parser.add_argument("--allow-fitting", action="store_true", help="Explicit Phase 7 fitting flag")
    parser.add_argument("--confirmation", required=True, help="Must equal the documented Phase 7 confirmation token")
    args = parser.parse_args(argv)

    report_path = run_phase7_final_fit(
        args.selection_report,
        allow_fitting=args.allow_fitting,
        confirmation=args.confirmation,
    )
    print(str(report_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
