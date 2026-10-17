import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.responses import FileResponse

import app as app_module


class Phase4AppSafetyTests(unittest.IsolatedAsyncioTestCase):
    def test_training_is_disabled_when_configuration_is_absent(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(app_module.training_endpoint_enabled())

    def test_training_requires_explicit_development_configuration(self):
        with patch.dict(
            os.environ,
            {"APP_ENV": "production", "ENABLE_TRAINING_ENDPOINT": "true"},
            clear=True,
        ):
            self.assertFalse(app_module.training_endpoint_enabled())

        with patch.dict(
            os.environ,
            {"APP_ENV": "development", "ENABLE_TRAINING_ENDPOINT": "false"},
            clear=True,
        ):
            self.assertFalse(app_module.training_endpoint_enabled())

        with patch.dict(os.environ, {"ENABLE_TRAINING_ENDPOINT": "true"}, clear=True):
            self.assertFalse(app_module.training_endpoint_enabled())

        with patch.dict(
            os.environ,
            {"APP_ENV": "development", "ENABLE_TRAINING_ENDPOINT": "true"},
            clear=True,
        ):
            self.assertTrue(app_module.training_endpoint_enabled())

    async def test_disabled_training_route_returns_404_without_importing_pipeline(self):
        import builtins

        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "network_security.pipeline.training_pipeline":
                raise AssertionError("Training pipeline must not be imported while the route is disabled.")
            return original_import(name, *args, **kwargs)

        with patch.dict(os.environ, {}, clear=True):
            with patch("builtins.__import__", side_effect=guarded_import):
                with self.assertRaises(HTTPException) as raised:
                    await app_module.train_route()

        self.assertEqual(raised.exception.status_code, 404)

    def test_prediction_result_ids_are_unique_hex_tokens(self):
        first_id = app_module.create_prediction_result_id()
        second_id = app_module.create_prediction_result_id()

        self.assertRegex(first_id, r"^[0-9a-f]{32}$")
        self.assertRegex(second_id, r"^[0-9a-f]{32}$")
        self.assertNotEqual(first_id, second_id)

    def test_result_path_uses_only_the_generated_id(self):
        result_id = "a" * 32
        with tempfile.TemporaryDirectory() as temporary_directory:
            with patch.object(app_module, "PREDICTION_OUTPUT_DIR", Path(temporary_directory)):
                self.assertEqual(
                    app_module.prediction_result_path(result_id),
                    Path(temporary_directory) / f"prediction_{result_id}.csv",
                )

    def test_result_path_rejects_path_traversal_and_malformed_ids(self):
        invalid_ids = ("../app.py", "..", "a" * 31, "A" * 32, "a" * 33, "")
        for result_id in invalid_ids:
            with self.subTest(result_id=result_id):
                with self.assertRaises(HTTPException) as raised:
                    app_module.prediction_result_path(result_id)
                self.assertEqual(raised.exception.status_code, 404)

    async def test_download_requires_a_request_specific_result_id(self):
        with self.assertRaises(HTTPException) as raised:
            await app_module.download_result()
        self.assertEqual(raised.exception.status_code, 404)

    async def test_download_returns_404_for_missing_result(self):
        result_id = "b" * 32
        with tempfile.TemporaryDirectory() as temporary_directory:
            with patch.object(app_module, "PREDICTION_OUTPUT_DIR", Path(temporary_directory)):
                with self.assertRaises(HTTPException) as raised:
                    await app_module.download_result(result_id=result_id)
        self.assertEqual(raised.exception.status_code, 404)

    async def test_download_serves_only_the_requested_result_file(self):
        result_id = "c" * 32
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            requested_result = output_directory / f"prediction_{result_id}.csv"
            other_result = output_directory / f"prediction_{'d' * 32}.csv"
            requested_result.write_text("feature,predicted_column\n1,0\n", encoding="utf-8")
            other_result.write_text("feature,predicted_column\n2,1\n", encoding="utf-8")

            with patch.object(app_module, "PREDICTION_OUTPUT_DIR", output_directory):
                response = await app_module.download_result(result_id=result_id)

            self.assertIsInstance(response, FileResponse)
            self.assertEqual(Path(response.path), requested_result)
            self.assertNotEqual(Path(response.path), other_result)
            self.assertEqual(response.filename, "phishing_predictions.csv")

    def test_cleanup_removes_expired_prediction_files_only(self):
        now = time.time()
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            expired_result = output_directory / f"prediction_{'e' * 32}.csv"
            recent_result = output_directory / f"prediction_{'f' * 32}.csv"
            legacy_output = output_directory / "output.csv"

            expired_result.write_text("expired", encoding="utf-8")
            recent_result.write_text("recent", encoding="utf-8")
            legacy_output.write_text("legacy", encoding="utf-8")
            old_timestamp = now - 120
            os.utime(expired_result, (old_timestamp, old_timestamp))

            with patch.object(app_module, "PREDICTION_OUTPUT_DIR", output_directory):
                with patch.object(app_module, "PREDICTION_RESULT_RETENTION_SECONDS", 60):
                    app_module.cleanup_expired_prediction_results(now=now)

            self.assertFalse(expired_result.exists())
            self.assertTrue(recent_result.exists())
            self.assertTrue(legacy_output.exists())


if __name__ == "__main__":
    unittest.main()
