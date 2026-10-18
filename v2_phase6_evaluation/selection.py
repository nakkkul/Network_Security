"""Training-only grouped-CV model-selection design for a separately approved phase.

Nothing in this module fits an estimator merely by being imported. The function
that performs inner-CV fitting has a two-part explicit authorization guard and
is intentionally not called by Phase 6 tests or the preflight CLI.
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .protocol import PROJECT_ROOT, verify_phase2_protocol


PHISHING_LABEL = 0
PHISHING_F1_SCORER = make_scorer(f1_score, pos_label=PHISHING_LABEL, zero_division=0)
FIT_AUTH_ENV = "NETWORK_SECURITY_PHASE6_ALLOW_MODEL_FIT"
FIT_AUTH_TOKEN = "I_APPROVE_MODEL_FITTING_IN_A_SEPARATELY_AUTHORIZED_PHASE"


class Phase6FitDisabledError(RuntimeError):
    """Raised unless a distinct, explicit future phase authorizes inner-CV fitting."""


@dataclass(frozen=True)
class TrainingOnlySelectionPlan:
    """Selection inputs consisting of outer-training rows only."""

    X_train: pd.DataFrame
    y_train: pd.Series
    source_row_indices: np.ndarray
    cv_splits: tuple[tuple[np.ndarray, np.ndarray], ...]
    feature_columns: tuple[str, ...]
    target_column: str
    protocol_hashes: dict[str, str]
    protocol_metadata: dict[str, Any]


def prepare_training_only_selection_plan(
    root: str | Path | None = None,
) -> TrainingOnlySelectionPlan:
    """Build training-only arrays from the frozen manifest and exact inner folds.

    The returned object does not contain the outer holdout feature rows or labels.
    This helper reads/verifies the canonical source files but never fits or predicts.
    """
    evidence = verify_phase2_protocol(root)

    try:
        # The full CSV is read for row-addressing, but only outer-training rows are
        # returned to the model-selection interface below.
        dataset = pd.read_csv(evidence.dataset_path, low_memory=False)
    except Exception as exc:
        raise RuntimeError("Unable to reload the verified canonical CSV.") from exc

    train_rows = evidence.row_manifest.loc[evidence.row_manifest["partition"] == "train"]
    train_rows = train_rows.sort_values("source_row_index").reset_index(drop=True)
    source_indices = train_rows["source_row_index"].to_numpy(dtype=np.int64)
    X_train = dataset.iloc[source_indices].loc[:, list(evidence.feature_columns)].copy().reset_index(drop=True)
    raw_y = pd.to_numeric(dataset.iloc[source_indices][evidence.target_column], errors="raise").astype(np.int64)
    y_train = raw_y.map({-1: 0, 1: 1}).astype(np.int64).reset_index(drop=True)
    if len(X_train) != 8_843 or len(y_train) != 8_843:
        raise RuntimeError("Training-only selection data must contain exactly 8,843 outer-training rows.")
    if set(y_train.unique().tolist()) != {0, 1}:
        raise RuntimeError("Outer training labels must include phishing class 0 and legitimate class 1.")

    row_to_position = {int(source_index): position for position, source_index in enumerate(source_indices)}
    cv_assignment = evidence.cv_assignment.sort_values("source_row_index").reset_index(drop=True)
    groups_by_source = dict(
        zip(cv_assignment["source_row_index"].astype(int), cv_assignment["feature_group_id"].astype(str))
    )
    all_positions = np.arange(len(source_indices), dtype=np.int64)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_number in range(5):
        validation_sources = cv_assignment.loc[
            cv_assignment["validation_fold"] == fold_number, "source_row_index"
        ].astype(int).tolist()
        validation_indices = np.asarray([row_to_position[source] for source in validation_sources], dtype=np.int64)
        validation_mask = np.ones(len(source_indices), dtype=bool)
        validation_mask[validation_indices] = False
        training_indices = all_positions[validation_mask]
        if not len(validation_indices) or not len(training_indices):
            raise RuntimeError(f"Inner CV fold {fold_number} is empty.")
        train_groups = {groups_by_source[int(source_indices[pos])] for pos in training_indices}
        validation_groups = {groups_by_source[int(source_indices[pos])] for pos in validation_indices}
        if train_groups.intersection(validation_groups):
            raise RuntimeError(f"Feature-group overlap detected in inner CV fold {fold_number}.")
        if set(y_train.iloc[training_indices].unique().tolist()) != {0, 1}:
            raise RuntimeError(f"Inner CV fold {fold_number} training labels do not contain both classes.")
        if set(y_train.iloc[validation_indices].unique().tolist()) != {0, 1}:
            raise RuntimeError(f"Inner CV fold {fold_number} validation labels do not contain both classes.")
        folds.append((training_indices, validation_indices))

    validation_count = np.zeros(len(X_train), dtype=np.int64)
    for _, validation_indices in folds:
        validation_count[validation_indices] += 1
    if not np.all(validation_count == 1):
        raise RuntimeError("Every outer-training row must be in exactly one inner validation fold.")

    return TrainingOnlySelectionPlan(
        X_train=X_train,
        y_train=y_train,
        source_row_indices=source_indices,
        cv_splits=tuple(folds),
        feature_columns=evidence.feature_columns,
        target_column=evidence.target_column,
        protocol_hashes=dict(evidence.hashes),
        protocol_metadata=dict(evidence.metadata),
    )


def build_candidate_searches(
    plan: TrainingOnlySelectionPlan,
    *,
    random_state: int = 43,
) -> dict[str, GridSearchCV]:
    """Construct, but do not run, the training-only CV search objects.

    Every candidate is a Pipeline so KNN imputation is learned inside each CV
    training fold. Search scoring is phishing-positive F1 (class 0), with refit
    disabled so this function never requests a full-training-data refit.
    """
    if len(plan.cv_splits) != 5:
        raise ValueError("The frozen Phase 2 protocol requires exactly five inner folds.")
    if len(plan.X_train) != len(plan.y_train) or len(plan.X_train) != 8_843:
        raise ValueError("Candidate searches must receive only the 8,843-row outer training partition.")

    models: dict[str, tuple[Any, dict[str, list[Any]]]] = {
        "Random Forest": (
            RandomForestClassifier(random_state=random_state, n_jobs=1),
            {"classifier__n_estimators": [8, 16, 32, 128, 256]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=random_state),
            {"classifier__criterion": ["gini", "entropy", "log_loss"]},
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=random_state),
            {
                "classifier__learning_rate": [0.1, 0.01, 0.05, 0.001],
                "classifier__subsample": [0.6, 0.7, 0.75, 0.85, 0.9],
                "classifier__n_estimators": [8, 16, 32, 64, 128, 256],
            },
        ),
        "Logistic Regression": (
            LogisticRegression(max_iter=2_000, random_state=random_state),
            {},
        ),
        "AdaBoost": (
            AdaBoostClassifier(random_state=random_state),
            {
                "classifier__learning_rate": [0.1, 0.01, 0.001],
                "classifier__n_estimators": [8, 16, 32, 64, 128, 256],
            },
        ),
    }

    searches: dict[str, GridSearchCV] = {}
    for name, (classifier, param_grid) in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", KNNImputer(missing_values=np.nan, n_neighbors=3, weights="uniform")),
                ("classifier", classifier),
            ]
        )
        searches[name] = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=PHISHING_F1_SCORER,
            cv=plan.cv_splits,
            refit=False,
            n_jobs=1,
            return_train_score=False,
            error_score="raise",
        )
    return searches


def _require_separate_fit_authorization(*, allow_fitting: bool, confirmation: str | None) -> None:
    """Safety tripwire: fit requires both a code argument and an explicit env var."""
    if (
        allow_fitting is not True
        or confirmation != FIT_AUTH_TOKEN
        or os.environ.get(FIT_AUTH_ENV) != FIT_AUTH_TOKEN
    ):
        raise Phase6FitDisabledError(
            "Estimator fitting is disabled unless a separate future phase is approved. "
            f"Both allow_fitting=True, confirmation={FIT_AUTH_TOKEN!r}, and environment "
            f"variable {FIT_AUTH_ENV}={FIT_AUTH_TOKEN!r} are required. Do not set these during Phase 6."
        )


def run_training_only_cv_selection(
    root: str | Path | None = None,
    *,
    allow_fitting: bool = False,
    confirmation: str | None = None,
) -> Path:
    """Run inner-CV model selection only after a distinct future authorization.

    This guarded function performs estimator fitting only inside the five saved
    inner folds on the 8,843 outer-training rows. It does not score the holdout,
    fit a final full-training estimator, save a model/preprocessor, or touch
    ``final_model/``. Its report is written to a unique new directory under
    ``evaluation_artifacts/``. This function is intentionally never called by
    Phase 6 implementation or tests.
    """
    _require_separate_fit_authorization(allow_fitting=allow_fitting, confirmation=confirmation)
    project_root = Path(root or PROJECT_ROOT).resolve()
    plan = prepare_training_only_selection_plan(project_root)
    searches = build_candidate_searches(plan)

    results: dict[str, dict[str, Any]] = {}
    for model_name, search in searches.items():
        # No outer-holdout arrays are available in `plan`; fit is only inner CV.
        search.fit(plan.X_train, plan.y_train)
        best_score = float(search.best_score_)
        if not np.isfinite(best_score):
            raise RuntimeError(f"Model {model_name!r} returned a non-finite inner-CV phishing F1 score.")
        results[model_name] = {
            "best_mean_inner_cv_phishing_f1": best_score,
            "best_parameters": search.best_params_,
            "metric": "f1_score(pos_label=0, zero_division=0)",
        }

    ranked = sorted(results.items(), key=lambda item: (-item[1]["best_mean_inner_cv_phishing_f1"], item[0]))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    output_dir = project_root / "evaluation_artifacts" / f"phase6_cv_selection_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": "1.0",
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_git_commit": _git_commit(project_root),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "dataset_sha256": plan.protocol_hashes["dataset"],
        "schema_sha256": plan.protocol_hashes["schema"],
        "split_manifest_sha256": plan.protocol_hashes["split_manifest"],
        "inner_cv_assignment_sha256": plan.protocol_hashes["inner_cv_assignment"],
        "outer_training_rows": int(len(plan.X_train)),
        "inner_cv_folds": len(plan.cv_splits),
        "primary_selection_metric": "phishing F1; positive label is encoded class 0",
        "candidate_results": results,
        "selected_candidate_by_inner_cv_only": ranked[0][0] if ranked else None,
        "reserved_holdout_rows": 2_212,
        "holdout_scored": False,
        "holdout_predictions_generated": False,
        "final_estimator_fitted": False,
        "model_artifacts_saved": False,
        "output_policy": "new evaluation_artifacts/ directory only; never final_model/ or production outputs",
    }
    report_path = output_dir / "selection_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report_path


def _git_commit(project_root: Path) -> str | None:
    """Return current commit without mutating Git state; unavailable is recorded as null."""
    import subprocess

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = completed.stdout.strip()
    return commit if completed.returncode == 0 and commit else None
