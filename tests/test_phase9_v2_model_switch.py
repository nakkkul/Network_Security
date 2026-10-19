import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app
from network_security.utils.ml_utils.model.estimator import NetworkModel


class FakePreprocessor:
    feature_names_in_ = app.PREDICTION_FEATURES
    n_features_in_ = len(app.PREDICTION_FEATURES)

    def transform(self, frame):
        return frame


class FakeClassifier:
    n_features_in_ = len(app.PREDICTION_FEATURES)
    classes_ = [0, 1]

    def predict(self, frame):
        return [1] * len(frame)


class Phase9ModelSwitchTests(unittest.TestCase):
    def setUp(self):
        app.load_inference_model.cache_clear()

    def tearDown(self):
        app.load_inference_model.cache_clear()

    def test_default_version_remains_legacy(self):
        preprocessor = FakePreprocessor()
        classifier = FakeClassifier()
        env = {key: value for key, value in os.environ.items() if key != app.MODEL_VERSION_ENV}
        with patch.dict(os.environ, env, clear=True):
            with patch.object(app, "load_object", side_effect=[preprocessor, classifier]) as loader:
                model = app.load_inference_model()
        self.assertIsInstance(model, NetworkModel)
        self.assertEqual(loader.call_count, 2)
        self.assertIn("final_model", loader.call_args_list[0].args[0])
        self.assertIn("final_model", loader.call_args_list[1].args[0])

    def _fixture_model_dir(self, model_dir):
        for filename in app.PHASE7_EXPECTED_ARTIFACT_SHA256:
            (model_dir / filename).write_text("fixture", encoding="utf-8")

    def _valid_fit_report(self):
        return {
            "phase": 7,
            "selected_candidate_by_inner_cv_only": "Gradient Boosting",
            "outer_training_rows_fitted": 8843,
            "reserved_holdout_rows": 2212,
            "fitted_estimator_saved": True,
            "holdout_scored": False,
            "holdout_predictions_generated": False,
            "final_model_directory_modified": False,
            "dataset_sha256": "e20bf913b6c2e669f4e875e5d223be2542cc7c79a7b62d284ae9d9627c3d9678",
            "schema_sha256": "0fe172eb670067a0048c62b601d5b63d311be827a5ee81c65f0fbd3dc480ad83",
            "split_manifest_sha256": "3dc82b79e2feb8f7b5fd23045be577be92d73d14aeed0005d3577782749f3ac7",
            "inner_cv_assignment_sha256": "9540bfa60ee0955455e8ba8ae3e039f6c83741f7b2760bac1db9a10fb1291cc9",
            "selection_source_git_commit": "76482641d41c300660be597224208a14090607e5",
            "implementation_git_commit": "27f44fe950e92196ad8fe88a8500d5a6a89fa8ff",
            "artifacts_sha256": {
                name: app.PHASE7_EXPECTED_ARTIFACT_SHA256[name]
                for name in app.PHASE7_EXPECTED_ARTIFACT_SHA256
                if name != "fit_report.json"
            },
        }

    def _load_phase7_with_mocks(self, model_dir, report=None, preprocessor=None, classifier=None, hash_fn=None):
        preprocessor = preprocessor or FakePreprocessor()
        classifier = classifier or FakeClassifier()
        with patch.dict(os.environ, {app.MODEL_VERSION_ENV: "phase7"}, clear=False):
            with patch.object(app, "PHASE7_MODEL_DIR", model_dir):
                with patch.object(
                    app,
                    "_sha256_file",
                    side_effect=hash_fn or (lambda path: app.PHASE7_EXPECTED_ARTIFACT_SHA256[path.name]),
                ):
                    with patch.object(app.json, "load", return_value=report or self._valid_fit_report()):
                        with patch.object(app.joblib, "load", side_effect=[preprocessor, classifier]) as loader:
                            app.load_inference_model.cache_clear()
                            result = app.load_inference_model()
        return result, loader

    def test_phase7_version_loads_hash_pinned_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            self._fixture_model_dir(model_dir)
            model, loader = self._load_phase7_with_mocks(model_dir)
        self.assertIsInstance(model, NetworkModel)
        self.assertEqual(loader.call_count, 2)

    def test_phase7_hash_mismatch_blocks_deserialization(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            self._fixture_model_dir(model_dir)
            with patch.dict(os.environ, {app.MODEL_VERSION_ENV: "phase7"}, clear=False):
                with patch.object(app, "PHASE7_MODEL_DIR", model_dir):
                    with patch.object(app, "_sha256_file", return_value="0" * 64):
                        with patch.object(app.joblib, "load") as loader:
                            app.load_inference_model.cache_clear()
                            with self.assertRaises(Exception):
                                app.load_inference_model()
                            loader.assert_not_called()

    def test_unknown_model_version_is_rejected_instead_of_falling_back(self):
        with patch.dict(os.environ, {app.MODEL_VERSION_ENV: "typo"}, clear=False):
            app.load_inference_model.cache_clear()
            with self.assertRaises(ValueError):
                app.load_inference_model()

    def test_phase7_preprocessor_schema_mismatch_is_rejected(self):
        class WrongPreprocessor:
            feature_names_in_ = ("not_a_feature",)
            n_features_in_ = 1

        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            self._fixture_model_dir(model_dir)
            wrong_preprocessor = WrongPreprocessor()
            # Stub load sequence; validation runs after trusted-artifact hash checks.
            with patch.dict(os.environ, {app.MODEL_VERSION_ENV: "phase7"}, clear=False):
                with patch.object(app, "PHASE7_MODEL_DIR", model_dir):
                    with patch.object(app, "_sha256_file", side_effect=lambda path: app.PHASE7_EXPECTED_ARTIFACT_SHA256[path.name]):
                        with patch.object(app.json, "load", return_value=self._valid_fit_report()):
                            with patch.object(app.joblib, "load", side_effect=[wrong_preprocessor, FakeClassifier()]):
                                app.load_inference_model.cache_clear()
                                with self.assertRaises(Exception):
                                    app.load_inference_model()

    def test_fit_report_dataset_or_split_mismatch_is_rejected_before_loading(self):
        bad_report = self._valid_fit_report()
        bad_report["dataset_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            self._fixture_model_dir(model_dir)
            with patch.dict(os.environ, {app.MODEL_VERSION_ENV: "phase7"}, clear=False):
                with patch.object(app, "PHASE7_MODEL_DIR", model_dir):
                    with patch.object(app, "_sha256_file", side_effect=lambda path: app.PHASE7_EXPECTED_ARTIFACT_SHA256[path.name]):
                        with patch.object(app.json, "load", return_value=bad_report):
                            with patch.object(app.joblib, "load") as loader:
                                app.load_inference_model.cache_clear()
                                with self.assertRaises(Exception):
                                    app.load_inference_model()
                                loader.assert_not_called()

    def test_legacy_loader_is_not_used_when_phase7_selected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            self._fixture_model_dir(model_dir)
            with patch.dict(os.environ, {app.MODEL_VERSION_ENV: "phase7"}, clear=False):
                with patch.object(app, "PHASE7_MODEL_DIR", model_dir):
                    with patch.object(app, "_sha256_file", side_effect=lambda path: app.PHASE7_EXPECTED_ARTIFACT_SHA256[path.name]):
                        with patch.object(app.json, "load", return_value=self._valid_fit_report()):
                            with patch.object(app, "load_object") as legacy_loader:
                                with patch.object(app.joblib, "load", side_effect=[FakePreprocessor(), FakeClassifier()]):
                                    app.load_inference_model.cache_clear()
                                    app.load_inference_model()
                                    legacy_loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
