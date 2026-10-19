import os
import tempfile
import unittest
from types import SimpleNamespace

import pandas as pd

from network_security.components.data_validation import DataValidation
from network_security.entity.artifact_entity import DataIngestionArtifact
from network_security.exception.exception import NetworkSecurityException


class DataValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir_obj = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir_obj.cleanup)
        self.temp_dir = self.temp_dir_obj.name
        self.expected_columns = ["feature_a", "feature_b", "Result"]
        self.validator = DataValidation.__new__(DataValidation)
        self.validator._schema_config = {
            "columns": [{name: "int64"} for name in self.expected_columns]
        }
        self.validator.data_validation_config = SimpleNamespace(
            drift_report_file_path=os.path.join(self.temp_dir, "reports", "drift.yaml"),
            valid_train_file_path=os.path.join(self.temp_dir, "validated", "train.csv"),
            valid_test_file_path=os.path.join(self.temp_dir, "validated", "test.csv"),
        )

    def test_expected_column_count_names_and_order_are_accepted(self):
        dataframe = pd.DataFrame(columns=self.expected_columns)
        self.assertTrue(self.validator.validate_number_of_columns(dataframe))

    def test_wrong_column_names_are_rejected_even_when_count_matches(self):
        dataframe = pd.DataFrame(columns=["feature_a", "wrong_name", "Result"])
        self.assertFalse(self.validator.validate_number_of_columns(dataframe))

    def test_drift_detector_returns_true_for_identical_distributions(self):
        dataframe = pd.DataFrame(
            {"feature_a": [0, 0, 1, 1], "feature_b": [0, 1, 0, 1], "Result": [0, 0, 1, 1]}
        )
        self.assertTrue(self.validator.detect_dataset_drift(dataframe, dataframe.copy()))
        self.assertTrue(os.path.exists(self.validator.data_validation_config.drift_report_file_path))

    def test_drift_detector_returns_false_for_distribution_shift(self):
        base = pd.DataFrame({"feature_a": [0] * 100, "feature_b": [0] * 100, "Result": [0] * 100})
        current = pd.DataFrame({"feature_a": [1] * 100, "feature_b": [1] * 100, "Result": [1] * 100})
        self.assertFalse(self.validator.detect_dataset_drift(base, current))

    def test_drift_detector_rejects_mismatched_columns(self):
        base = pd.DataFrame({"feature_a": [0], "feature_b": [1]})
        current = pd.DataFrame({"feature_b": [1], "feature_a": [0]})
        with self.assertRaises(NetworkSecurityException):
            self.validator.detect_dataset_drift(base, current)

    def test_validation_artifact_points_to_validated_outputs(self):
        dataframe = pd.DataFrame(
            {"feature_a": [0, 0, 1, 1], "feature_b": [0, 1, 0, 1], "Result": [0, 0, 1, 1]}
        )
        train_input = os.path.join(self.temp_dir, "input", "train.csv")
        test_input = os.path.join(self.temp_dir, "input", "test.csv")
        os.makedirs(os.path.dirname(train_input), exist_ok=True)
        dataframe.to_csv(train_input, index=False)
        dataframe.to_csv(test_input, index=False)
        self.validator.data_ingestion_artifact = DataIngestionArtifact(
            trained_file_path=train_input,
            test_file_path=test_input,
        )

        artifact = self.validator.initiate_data_validation()

        self.assertEqual(artifact.valid_train_file_path, self.validator.data_validation_config.valid_train_file_path)
        self.assertEqual(artifact.valid_test_file_path, self.validator.data_validation_config.valid_test_file_path)
        self.assertTrue(os.path.exists(artifact.valid_train_file_path))
        self.assertTrue(os.path.exists(artifact.valid_test_file_path))
        self.assertEqual(list(pd.read_csv(artifact.valid_train_file_path).columns), self.expected_columns)

    def test_initiate_validation_stops_on_schema_mismatch(self):
        bad_dataframe = pd.DataFrame(
            {"feature_a": [0, 1], "wrong_name": [0, 1], "Result": [0, 1]}
        )
        input_path = os.path.join(self.temp_dir, "input", "bad.csv")
        os.makedirs(os.path.dirname(input_path), exist_ok=True)
        bad_dataframe.to_csv(input_path, index=False)
        self.validator.data_ingestion_artifact = DataIngestionArtifact(
            trained_file_path=input_path,
            test_file_path=input_path,
        )

        with self.assertRaises(NetworkSecurityException):
            self.validator.initiate_data_validation()

        self.assertFalse(os.path.exists(self.validator.data_validation_config.valid_train_file_path))


if __name__ == "__main__":
    unittest.main()
