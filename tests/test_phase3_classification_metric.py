import unittest

from network_security.utils.ml_utils.metric.classification_metric import (
    PHISHING_LABEL,
    get_classification_score,
)


class ClassificationMetricTests(unittest.TestCase):
    def test_phishing_label_zero_is_used_for_precision_recall_and_f1(self):
        y_true = [0, 0, 0, 0, 1, 1, 1, 1]
        y_pred = [0, 0, 1, 1, 0, 1, 1, 1]

        metrics = get_classification_score(y_true, y_pred)

        self.assertEqual(PHISHING_LABEL, 0)
        self.assertAlmostEqual(metrics.precision_score, 2 / 3)
        self.assertAlmostEqual(metrics.recall_score, 1 / 2)
        self.assertAlmostEqual(metrics.f1_score, 4 / 7)

    def test_metrics_return_zero_without_phishing_support_or_predictions(self):
        metrics = get_classification_score([1, 1, 1], [1, 1, 1])

        self.assertEqual(metrics.precision_score, 0.0)
        self.assertEqual(metrics.recall_score, 0.0)
        self.assertEqual(metrics.f1_score, 0.0)


if __name__ == "__main__":
    unittest.main()
