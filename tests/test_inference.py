import unittest
from pathlib import Path

import pandas as pd

from app import PREDICTION_FEATURES, load_inference_model, validate_prediction_dataframe


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_FILE = ROOT / "sample_data" / "sample_prediction.csv"


class InferenceSmokeTests(unittest.TestCase):
    def test_persisted_artifacts_match_inference_schema(self):
        model = load_inference_model()
        self.assertEqual(tuple(model.preprocessor.feature_names_in_), PREDICTION_FEATURES)
        self.assertEqual(model.model.n_features_in_, len(PREDICTION_FEATURES))

    def test_sample_csv_has_expected_schema(self):
        df = pd.read_csv(SAMPLE_FILE)
        self.assertEqual(tuple(df.columns), PREDICTION_FEATURES)
        self.assertNotIn("Result", df.columns)
        self.assertGreater(len(df), 0)

    def test_sample_csv_can_be_predicted(self):
        df = pd.read_csv(SAMPLE_FILE)
        validated = validate_prediction_dataframe(df)
        model = load_inference_model()
        predictions = model.predict(validated)
        self.assertEqual(len(predictions), len(validated))
        self.assertTrue(set(predictions).issubset({0, 1}))

    def test_feature_columns_are_reordered_to_model_order(self):
        df = pd.read_csv(SAMPLE_FILE)
        shuffled = df[list(reversed(PREDICTION_FEATURES))]
        validated = validate_prediction_dataframe(shuffled)
        self.assertEqual(tuple(validated.columns), PREDICTION_FEATURES)

    def test_target_column_is_rejected(self):
        df = pd.read_csv(SAMPLE_FILE)
        df["Result"] = 1
        with self.assertRaises(ValueError):
            validate_prediction_dataframe(df)

    def test_missing_column_is_rejected(self):
        df = pd.read_csv(SAMPLE_FILE).drop(columns=[PREDICTION_FEATURES[0]])
        with self.assertRaises(ValueError):
            validate_prediction_dataframe(df)

    def test_unexpected_column_is_rejected(self):
        df = pd.read_csv(SAMPLE_FILE)
        df["unexpected_feature"] = 0
        with self.assertRaises(ValueError):
            validate_prediction_dataframe(df)


if __name__ == "__main__":
    unittest.main()
