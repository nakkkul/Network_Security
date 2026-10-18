"""Dataset-independent tests for Phase 8 safeguards; never load a model or score holdout data."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from v2_phase6_evaluation.final_fit import (
    EXPECTED_RANDOM_STATE,
    EXPECTED_SELECTED_CANDIDATE,
    EXPECTED_SELECTED_PARAMETERS,
    EXPECTED_SELECTION_METRIC,
    EXPECTED_OUTER_TRAIN_ROWS,
    EXPECTED_RESERVED_HOLDOUT_ROWS,
    PROTECTED_ARTIFACT_SHA256,
)
from v2_phase6_evaluation.holdout_eval import (
    EXPECTED_ARTIFACT_FILES,
    PHASE8_AUTH_ENV,
    PHASE8_AUTH_TOKEN,
    PHASE8_GUARD_FILENAME,
    Phase8HoldoutEvaluationError,
    _check_no_prior_evaluation,
    _create_one_time_guard,
    _require_evaluation_authorization,
    _resolve_fit_report,
    _resolve_selection_report,
    _validate_fit_and_selection_reports,
    _training_row_index_digest,
    _verify_training_row_provenance,
    metrics_from_predictions,
)
from v2_phase6_evaluation.protocol import EXPECTED_SHA256


class Phase8HoldoutEvaluationTests(unittest.TestCase):
    def test_authorization_requires_boolean_token_and_current_session_variable(self):
        with patch.dict(os.environ, {PHASE8_AUTH_ENV: ""}, clear=False):
            with self.assertRaises(Phase8HoldoutEvaluationError):
                _require_evaluation_authorization(allow_holdout_evaluation=True, confirmation=PHASE8_AUTH_TOKEN)
            with patch.dict(os.environ, {PHASE8_AUTH_ENV: PHASE8_AUTH_TOKEN}, clear=False):
                with self.assertRaises(Phase8HoldoutEvaluationError):
                    _require_evaluation_authorization(allow_holdout_evaluation=False, confirmation=PHASE8_AUTH_TOKEN)
                with self.assertRaises(Phase8HoldoutEvaluationError):
                    _require_evaluation_authorization(allow_holdout_evaluation=True, confirmation="wrong")
                _require_evaluation_authorization(allow_holdout_evaluation=True, confirmation=PHASE8_AUTH_TOKEN)

    def test_fit_report_must_be_inside_phase7_artifact_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "Artifacts"
            good_dir = artifacts / "phase7_final_fit_test"
            good_dir.mkdir(parents=True)
            good_path = good_dir / "fit_report.json"
            good_path.write_text("{}", encoding="utf-8")
            resolved, resolved_dir = _resolve_fit_report(root, good_path)
            self.assertEqual(resolved, good_path.resolve())
            self.assertEqual(resolved_dir, good_dir.resolve())

            bad_dir = artifacts / "other"
            bad_dir.mkdir()
            bad_path = bad_dir / "fit_report.json"
            bad_path.write_text("{}", encoding="utf-8")
            with self.assertRaises(Phase8HoldoutEvaluationError):
                _resolve_fit_report(root, bad_path)

    def test_selection_report_must_be_inside_evaluation_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "evaluation_artifacts").mkdir()
            outside = root / "selection_report.json"
            outside.write_text("{}", encoding="utf-8")
            with self.assertRaises(Phase8HoldoutEvaluationError):
                _resolve_selection_report(root, str(outside))

    def test_metrics_treat_encoded_class_zero_as_phishing_positive(self):
        # Synthetic arrays only; this is not the canonical dataset or reserved holdout.
        result = metrics_from_predictions([0, 0, 1, 1], [0, 1, 0, 1])
        phishing = result["phishing_class_0"]
        self.assertEqual(phishing["precision"], 0.5)
        self.assertEqual(phishing["recall"], 0.5)
        self.assertEqual(phishing["f1"], 0.5)
        self.assertEqual(phishing["support"], 2)
        self.assertEqual(result["confusion_matrix"]["counts"], [[1, 1], [1, 1]])
        self.assertEqual(result["confusion_matrix"]["phishing_positive_counts"], {
            "true_positive": 1, "false_negative": 1, "false_positive": 1, "true_negative": 1,
        })

    def test_metrics_reject_mismatched_length_and_invalid_classes(self):
        with self.assertRaises(Phase8HoldoutEvaluationError):
            metrics_from_predictions([0, 1], [0])
        with self.assertRaises(Phase8HoldoutEvaluationError):
            metrics_from_predictions([0, 1], [0, 2])

    def test_existing_guard_or_prior_output_blocks_a_second_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evaluation_root = root / "evaluation_artifacts"
            evaluation_root.mkdir()
            marker = evaluation_root / PHASE8_GUARD_FILENAME
            marker.write_text('{"status":"completed"}', encoding="utf-8")
            with self.assertRaises(Phase8HoldoutEvaluationError):
                _check_no_prior_evaluation(root)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evaluation_root = root / "evaluation_artifacts"
            (evaluation_root / "phase8_holdout_evaluation_prior").mkdir(parents=True)
            with self.assertRaises(Phase8HoldoutEvaluationError):
                _check_no_prior_evaluation(root)

    def test_one_time_guard_creation_is_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / PHASE8_GUARD_FILENAME
            _create_one_time_guard(marker, {"status": "started"})
            self.assertTrue(marker.is_file())
            with self.assertRaises(Phase8HoldoutEvaluationError):
                _create_one_time_guard(marker, {"status": "second attempt"})
            saved = json.loads(marker.read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "started")

    def _synthetic_report_files(self, root: Path):
        fit_dir = root / "Artifacts" / "phase7_final_fit_test-run"
        fit_dir.mkdir(parents=True)
        selection_dir = root / "evaluation_artifacts" / "phase6_cv_selection_test-run"
        selection_dir.mkdir(parents=True)
        (root / "final_model").mkdir()
        for filename in PROTECTED_ARTIFACT_SHA256:
            (root / "final_model" / filename).write_bytes(b"placeholder")
        for filename in EXPECTED_ARTIFACT_FILES:
            (fit_dir / filename).write_bytes(b"placeholder")

        selection_report = {
            "schema_version": "1.0",
            "run_id": "selection-test-run",
            "project_git_commit": "1" * 40,
            "dataset_sha256": EXPECTED_SHA256["dataset"],
            "schema_sha256": EXPECTED_SHA256["schema"],
            "split_manifest_sha256": EXPECTED_SHA256["split_manifest"],
            "inner_cv_assignment_sha256": EXPECTED_SHA256["inner_cv_assignment"],
            "outer_training_rows": EXPECTED_OUTER_TRAIN_ROWS,
            "inner_cv_folds": 5,
            "primary_selection_metric": EXPECTED_SELECTION_METRIC,
            "selected_candidate_by_inner_cv_only": EXPECTED_SELECTED_CANDIDATE,
            "reserved_holdout_rows": EXPECTED_RESERVED_HOLDOUT_ROWS,
            "holdout_scored": False,
            "holdout_predictions_generated": False,
            "final_estimator_fitted": False,
            "model_artifacts_saved": False,
            "candidate_results": {
                EXPECTED_SELECTED_CANDIDATE: {
                    "best_mean_inner_cv_phishing_f1": 0.9438506766903793,
                    "best_parameters": dict(EXPECTED_SELECTED_PARAMETERS),
                    "metric": "f1_score(pos_label=0, zero_division=0)",
                }
            },
        }
        selection_path = selection_dir / "selection_report.json"
        selection_path.write_text(json.dumps(selection_report), encoding="utf-8")
        fit_report = {
            "phase": 7,
            "run_id": "test-run",
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
            "target_mapping": {"raw_-1": 0, "raw_1": 1},
            "dataset_sha256": EXPECTED_SHA256["dataset"],
            "schema_sha256": EXPECTED_SHA256["schema"],
            "split_manifest_sha256": EXPECTED_SHA256["split_manifest"],
            "inner_cv_assignment_sha256": EXPECTED_SHA256["inner_cv_assignment"],
            "classifier_parameters": {
                "learning_rate": 0.1, "n_estimators": 256, "subsample": 0.9,
                "random_state": EXPECTED_RANDOM_STATE,
            },
            "protected_final_model_hashes_before_and_after": dict(PROTECTED_ARTIFACT_SHA256),
            "selection_report_path": str(selection_path.relative_to(root)).replace("\\", "/"),
            "selection_report_sha256": "e" * 64,
            "selection_run_id": "selection-test-run",
            "selection_source_git_commit": "1" * 40,
            "implementation_git_commit": "2" * 40,
            "git_branch": "v2-development",
            "selected_inner_cv_mean_phishing_f1": 0.9438506766903793,
            "artifacts_sha256": {
                "selected_pipeline.joblib": "a" * 64,
                "classifier.joblib": "b" * 64,
                "preprocessor.joblib": "c" * 64,
                "outer_training_source_row_indices.csv": "d" * 64,
            },
        }
        fit_path = fit_dir / "fit_report.json"
        fit_path.write_text(json.dumps(fit_report), encoding="utf-8")
        return fit_path, fit_dir, selection_path, fit_report, selection_report

    def test_complete_fit_and_selection_provenance_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fit_path, fit_dir, selection_path, _, _ = self._synthetic_report_files(root)

            def fake_hash(path):
                path = Path(path)
                if path.name == "model.pkl":
                    return PROTECTED_ARTIFACT_SHA256["model.pkl"]
                if path.name == "preprocessor.pkl":
                    return PROTECTED_ARTIFACT_SHA256["preprocessor.pkl"]
                expected = {
                    "selection_report.json": "e" * 64,
                    "selected_pipeline.joblib": "a" * 64,
                    "classifier.joblib": "b" * 64,
                    "preprocessor.joblib": "c" * 64,
                    "outer_training_source_row_indices.csv": "d" * 64,
                }
                return expected[path.name]

            with patch("v2_phase6_evaluation.holdout_eval.sha256_file", side_effect=fake_hash):
                fit, selection, pipeline_path, resolved_selection = _validate_fit_and_selection_reports(
                    root, fit_path, fit_dir, dict(EXPECTED_SHA256)
                )
            self.assertEqual(fit["selected_candidate_by_inner_cv_only"], EXPECTED_SELECTED_CANDIDATE)
            self.assertEqual(selection["run_id"], "selection-test-run")
            self.assertEqual(pipeline_path.name, "selected_pipeline.joblib")
            self.assertEqual(resolved_selection, selection_path.resolve())

    def test_training_row_provenance_must_exactly_match_frozen_training_partition(self):
        with tempfile.TemporaryDirectory() as tmp:
            fit_dir = Path(tmp) / "phase7_final_fit_test"
            fit_dir.mkdir()
            training_indices = np.arange(EXPECTED_OUTER_TRAIN_ROWS, dtype=np.int64)
            pd.DataFrame({"source_row_index": training_indices}).to_csv(
                fit_dir / "outer_training_source_row_indices.csv", index=False
            )
            manifest = pd.DataFrame({
                "source_row_index": np.arange(EXPECTED_OUTER_TRAIN_ROWS + EXPECTED_RESERVED_HOLDOUT_ROWS, dtype=np.int64),
                "partition": (["train"] * EXPECTED_OUTER_TRAIN_ROWS)
                + (["holdout"] * EXPECTED_RESERVED_HOLDOUT_ROWS),
            })
            fit_report = {"training_source_row_indices_sha256": _training_row_index_digest(training_indices)}
            evidence = SimpleNamespace(row_manifest=manifest)
            _verify_training_row_provenance(fit_dir, fit_report, evidence)

            bad_report = {"training_source_row_indices_sha256": "0" * 64}
            with self.assertRaisesRegex(Phase8HoldoutEvaluationError, "digest differs"):
                _verify_training_row_provenance(fit_dir, bad_report, evidence)

    def test_reports_are_rejected_when_dataset_hash_differs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fit_path, fit_dir, _, fit_report, _ = self._synthetic_report_files(root)
            fit_report["dataset_sha256"] = "f" * 64
            fit_path.write_text(json.dumps(fit_report), encoding="utf-8")
            with self.assertRaisesRegex(Phase8HoldoutEvaluationError, "dataset_sha256"):
                _validate_fit_and_selection_reports(root, fit_path, fit_dir, dict(EXPECTED_SHA256))

    def test_fit_report_rejected_if_holdout_was_used_earlier(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fit_path, fit_dir, _, fit_report, _ = self._synthetic_report_files(root)
            fit_report["holdout_scored"] = True
            fit_path.write_text(json.dumps(fit_report), encoding="utf-8")
            with self.assertRaisesRegex(Phase8HoldoutEvaluationError, "holdout_scored"):
                _validate_fit_and_selection_reports(root, fit_path, fit_dir, dict(EXPECTED_SHA256))


if __name__ == "__main__":
    unittest.main()
