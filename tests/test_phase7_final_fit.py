"""Dataset-independent Phase 7 guard/provenance tests. Never fit or predict."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from v2_phase6_evaluation.final_fit import (
    EXPECTED_SELECTED_CANDIDATE,
    EXPECTED_SELECTED_PARAMETERS,
    PHASE7_AUTH_ENV,
    PHASE7_AUTH_TOKEN,
    Phase7FinalFitError,
    _load_and_validate_selection_report,
    _resolve_selection_report,
    run_phase7_final_fit,
)


HASHES = {
    "dataset": "a" * 64,
    "schema": "b" * 64,
    "split_manifest": "c" * 64,
    "inner_cv_assignment": "d" * 64,
}


def valid_selection_report() -> dict:
    return {
        "run_id": "test-selection-run",
        "project_git_commit": "e" * 40,
        "dataset_sha256": HASHES["dataset"],
        "schema_sha256": HASHES["schema"],
        "split_manifest_sha256": HASHES["split_manifest"],
        "inner_cv_assignment_sha256": HASHES["inner_cv_assignment"],
        "outer_training_rows": 8843,
        "reserved_holdout_rows": 2212,
        "inner_cv_folds": 5,
        "primary_selection_metric": "phishing F1; positive label is encoded class 0",
        "selected_candidate_by_inner_cv_only": EXPECTED_SELECTED_CANDIDATE,
        "holdout_scored": False,
        "holdout_predictions_generated": False,
        "final_estimator_fitted": False,
        "model_artifacts_saved": False,
        "candidate_results": {
            "Gradient Boosting": {
                "best_mean_inner_cv_phishing_f1": 0.9438506766903793,
                "best_parameters": dict(EXPECTED_SELECTED_PARAMETERS),
                "metric": "f1_score(pos_label=0, zero_division=0)",
            },
            "Random Forest": {
                "best_mean_inner_cv_phishing_f1": 0.9434236144500581,
                "best_parameters": {"classifier__n_estimators": 128},
                "metric": "f1_score(pos_label=0, zero_division=0)",
            },
        },
    }


class Phase7FinalFitTests(unittest.TestCase):
    def test_runner_refuses_to_fit_without_explicit_authorization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(os.environ, {PHASE7_AUTH_ENV: ""}, clear=False):
                with patch("v2_phase6_evaluation.final_fit.prepare_training_only_selection_plan") as plan_mock:
                    with self.assertRaises(Phase7FinalFitError):
                        run_phase7_final_fit(
                            root / "does-not-exist.json",
                            root,
                            allow_fitting=False,
                            confirmation=PHASE7_AUTH_TOKEN,
                        )
                    plan_mock.assert_not_called()

    def test_selection_report_hashes_and_winner_are_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "selection_report.json"
            path.write_text(json.dumps(valid_selection_report()), encoding="utf-8")
            report, details = _load_and_validate_selection_report(
                path, current_protocol_hashes=HASHES
            )
            self.assertEqual(report["selected_candidate_by_inner_cv_only"], "Gradient Boosting")
            self.assertEqual(details["best_parameters"], EXPECTED_SELECTED_PARAMETERS)

    def test_selection_report_with_holdout_score_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = valid_selection_report()
            report["holdout_scored"] = True
            path = Path(tmp) / "selection_report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(Phase7FinalFitError, "holdout_scored"):
                _load_and_validate_selection_report(path, current_protocol_hashes=HASHES)

    def test_selection_report_with_changed_dataset_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = valid_selection_report()
            report["dataset_sha256"] = "f" * 64
            path = Path(tmp) / "selection_report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(Phase7FinalFitError, "dataset_sha256"):
                _load_and_validate_selection_report(path, current_protocol_hashes=HASHES)

    def test_selection_report_with_wrong_winner_or_parameters_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = valid_selection_report()
            report["candidate_results"]["Random Forest"]["best_mean_inner_cv_phishing_f1"] = 0.99
            path = Path(tmp) / "selection_report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(Phase7FinalFitError, "rank Gradient Boosting first"):
                _load_and_validate_selection_report(path, current_protocol_hashes=HASHES)

            report = valid_selection_report()
            report["candidate_results"]["Gradient Boosting"]["best_parameters"]["classifier__n_estimators"] = 128
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(Phase7FinalFitError, "parameters differ"):
                _load_and_validate_selection_report(path, current_protocol_hashes=HASHES)

    def test_selection_report_path_must_be_inside_evaluation_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "evaluation_artifacts").mkdir()
            outside = root / "selection_report.json"
            outside.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(Phase7FinalFitError, "inside evaluation_artifacts"):
                _resolve_selection_report(root, outside)

    def test_valid_report_path_is_resolved_under_evaluation_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_dir = root / "evaluation_artifacts" / "phase6_cv_selection_test"
            report_dir.mkdir(parents=True)
            report_path = report_dir / "selection_report.json"
            report_path.write_text("{}", encoding="utf-8")
            found = _resolve_selection_report(root, Path("evaluation_artifacts") / "phase6_cv_selection_test" / "selection_report.json")
            self.assertEqual(found, report_path.resolve())


if __name__ == "__main__":
    unittest.main()
