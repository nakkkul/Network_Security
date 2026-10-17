import unittest

import numpy as np
import pandas as pd

# Support both invocation styles: from this folder (`-s tests`) and from the
# project root (`-s v2_phase2_evaluation/tests`).
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation_protocol import (  # noqa: E402
    ProtocolError,
    compute_classification_metrics,
    create_split_plan,
    make_feature_group_ids,
    stable_feature_group_id,
    validate_dataset,
    verify_split_plan,
)


def synthetic_dataset(n_groups=100, seed=9):
    """Generate deterministic groupable data without fitting any model."""
    rng = np.random.default_rng(seed)
    records = []
    # 100 base vectors with two or three repeated rows. Most groups have a
    # consistent label; several deliberately have conflicting labels.
    for gid in range(n_groups):
        a = gid
        b = (gid * 7) % 19
        base_label = -1 if gid % 2 == 0 else 1
        repeats = 2 + (gid % 3 == 0)
        for j in range(repeats):
            label = base_label
            if gid in {2, 11, 27, 48} and j == repeats - 1:
                label = 1 if base_label == -1 else -1
            records.append({"f1": a, "f2": b, "Result": label})
    frame = pd.DataFrame(records)
    # shuffle source order deterministically; source row index remains stable
    return frame.sample(frac=1, random_state=seed).reset_index(drop=True)


class FeatureGroupTests(unittest.TestCase):
    def test_feature_group_id_does_not_include_target(self):
        a = stable_feature_group_id([1, -1, 0])
        b = stable_feature_group_id([1, -1, 0])
        c = stable_feature_group_id([1, -1, 1])
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_feature_group_ids_repeat_for_duplicate_vectors(self):
        features = pd.DataFrame({"f1": [1, 1, 2], "f2": [0, 0, 0]})
        groups = make_feature_group_ids(features)
        self.assertEqual(groups[0], groups[1])
        self.assertNotEqual(groups[0], groups[2])


class SplitProtocolTests(unittest.TestCase):
    def setUp(self):
        self.df = synthetic_dataset()
        self.feature_columns = ("f1", "f2")

    def build(self, df=None, seed=42):
        return create_split_plan(
            self.df if df is None else df,
            self.feature_columns,
            source_sha256="synthetic-test-dataset-sha256",
            seed=seed,
            outer_folds=5,
            inner_folds=5,
        )

    def test_outer_holdout_has_no_shared_feature_groups(self):
        plan = self.build()
        result = verify_split_plan(plan)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["outer_shared_feature_groups"], 0)

    def test_every_row_appears_once_in_outer_partition_manifest(self):
        plan = self.build()
        self.assertEqual(len(plan.row_manifest), len(self.df))
        self.assertEqual(plan.row_manifest.source_row_index.nunique(), len(self.df))
        self.assertEqual(set(plan.row_manifest.partition), {"train", "holdout"})

    def test_inner_cv_is_training_only_and_group_isolated(self):
        plan = self.build()
        result = verify_split_plan(plan)
        train_rows = set(plan.row_manifest.loc[plan.row_manifest.partition == "train", "source_row_index"])
        cv_rows = set(plan.cv_assignment.source_row_index)
        self.assertEqual(cv_rows, train_rows)
        self.assertEqual(result["inner_cv_source_rows_covered_once"], len(train_rows))
        self.assertTrue(all(v == 0 for v in result["inner_cv_shared_feature_groups_by_fold"].values()))
        # Every identical vector gets exactly one validation-fold assignment.
        self.assertTrue((plan.cv_assignment.groupby("feature_group_id").validation_fold.nunique() == 1).all())

    def test_split_plan_is_reproducible_for_same_environment(self):
        first = self.build()
        second = self.build()
        pd.testing.assert_frame_equal(first.row_manifest, second.row_manifest)
        pd.testing.assert_frame_equal(first.cv_assignment, second.cv_assignment)
        self.assertEqual(first.summary, second.summary)

    def test_seed_change_changes_assignments(self):
        first = self.build(seed=42)
        second = self.build(seed=31415)
        self.assertFalse(first.row_manifest.partition.equals(second.row_manifest.partition))

    def test_conflicting_groups_are_reported_not_removed(self):
        plan = self.build()
        self.assertGreater(plan.summary["conflicting_label_policy"]["conflicting_feature_group_count"], 0)
        self.assertEqual(len(plan.row_manifest), len(self.df))
        self.assertTrue(plan.row_manifest.is_conflicting_feature_group.any())

    def test_invalid_target_is_rejected(self):
        bad = self.df.copy()
        bad.loc[0, "Result"] = 99
        with self.assertRaises(ProtocolError):
            self.build(df=bad)

    def test_non_integer_target_is_rejected(self):
        bad = self.df.copy()
        bad["Result"] = bad["Result"].astype(float)
        bad.loc[0, "Result"] = 1.5
        with self.assertRaises(ProtocolError):
            self.build(df=bad)

    def test_allowed_feature_domain_is_enforced_when_requested(self):
        with self.assertRaises(ProtocolError):
            validate_dataset(self.df, self.feature_columns, allowed_feature_values=(-1, 0, 1))


class MetricTests(unittest.TestCase):
    def test_phishing_class_zero_is_explicit_positive_class(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 1, 1]
        score_for_phishing = [0.9, 0.8, 0.2, 0.1]
        result = compute_classification_metrics(y_true, y_pred, phishing_score=score_for_phishing)
        self.assertAlmostEqual(result["phishing_precision"], 1.0)
        self.assertAlmostEqual(result["phishing_recall"], 0.5)
        self.assertAlmostEqual(result["phishing_f1"], 2.0 / 3.0)
        self.assertAlmostEqual(result["legitimate_precision"], 2.0 / 3.0)
        self.assertAlmostEqual(result["legitimate_recall"], 1.0)
        self.assertAlmostEqual(result["roc_auc_phishing"], 1.0)
        self.assertAlmostEqual(result["pr_auc_phishing_average_precision"], 1.0)
        self.assertEqual(result["confusion_matrix"], [[1, 1], [0, 2]])

    def test_non_finite_score_is_rejected(self):
        with self.assertRaises(ProtocolError):
            compute_classification_metrics([0, 1], [0, 1], phishing_score=[np.nan, 0.2])


if __name__ == "__main__":
    unittest.main(verbosity=2)
