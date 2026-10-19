"""Phase 6 tests: protocol/selection-configuration checks only; no model fitting."""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

from v2_phase6_evaluation.protocol import (
    EXPECTED_SHA256,
    PROJECT_ROOT,
    verify_phase2_protocol,
)
from v2_phase6_evaluation.selection import (
    FIT_AUTH_ENV,
    FIT_AUTH_TOKEN,
    Phase6FitDisabledError,
    TrainingOnlySelectionPlan,
    build_candidate_searches,
    prepare_training_only_selection_plan,
    run_training_only_cv_selection,
)


class Phase6FrozenProtocolIntegrationTests(unittest.TestCase):
    """Checks against the actual frozen CSV/manifests when the private/local data is present.

    Network_Data/ is intentionally gitignored in this project. A GitHub Actions
    checkout therefore does not have the canonical CSV, so these integration
    checks are skipped there rather than weakening protocol validation or
    adding the dataset to version control. They still run on the owner's local
    checkout because the CSV is present there.
    """

    @classmethod
    def setUpClass(cls):
        cls.root = PROJECT_ROOT
        dataset_path = cls.root / "Network_Data" / "phisingData.csv"
        if not dataset_path.is_file():
            raise unittest.SkipTest(
                "Canonical dataset is not present. Network_Data/ is intentionally "
                "excluded from Git; run locally with the original CSV to execute "
                "the frozen Phase 2 integration checks. Do not regenerate manifests."
            )
        cls.evidence = verify_phase2_protocol(cls.root)
        cls.plan = prepare_training_only_selection_plan(cls.root)

    def test_read_only_preflight_validates_pinned_hashes_and_split_integrity(self):
        self.assertEqual(self.evidence.summary["status"], "PASS")
        self.assertEqual(self.evidence.hashes, EXPECTED_SHA256)
        self.assertEqual(self.evidence.summary["row_count"], 11055)
        self.assertEqual(self.evidence.summary["feature_count"], 30)
        self.assertEqual(self.evidence.summary["outer_train_rows"], 8843)
        self.assertEqual(self.evidence.summary["reserved_holdout_rows"], 2212)
        self.assertEqual(self.evidence.summary["inner_cv_fold_count"], 5)
        self.assertEqual(self.evidence.summary["outer_shared_feature_groups"], 0)
        self.assertTrue(
            all(
                value == 0
                for value in self.evidence.summary[
                    "inner_cv_shared_feature_groups_by_fold"
                ].values()
            )
        )
        self.assertEqual(self.evidence.summary["conflicting_feature_groups"], 64)
        self.assertEqual(self.evidence.summary["rows_in_conflicting_feature_groups"], 357)
        self.assertFalse(self.evidence.summary["estimator_fitted"])
        self.assertFalse(self.evidence.summary["holdout_scored"])
        self.assertFalse(self.evidence.summary["holdout_predictions_generated"])

    def test_selection_plan_exposes_only_outer_training_rows(self):
        self.assertEqual(len(self.plan.X_train), 8843)
        self.assertEqual(len(self.plan.y_train), 8843)
        self.assertEqual(set(self.plan.y_train.unique().tolist()), {0, 1})
        self.assertEqual(len(self.plan.source_row_indices), 8843)
        self.assertTrue(np.all(np.diff(self.plan.source_row_indices) > 0))
        self.assertEqual(len(self.plan.cv_splits), 5)
        holdout_indices = set(
            self.evidence.row_manifest.loc[
                self.evidence.row_manifest["partition"] == "holdout",
                "source_row_index",
            ].astype(int).tolist()
        )
        self.assertFalse(
            holdout_indices.intersection(
                set(self.plan.source_row_indices.astype(int).tolist())
            )
        )
        self.assertEqual(list(self.plan.X_train.columns), list(self.plan.feature_columns))

    def test_exact_inner_cv_assignments_are_group_disjoint_and_cover_training_once(self):
        seen = np.zeros(len(self.plan.X_train), dtype=np.int64)
        cv = self.evidence.cv_assignment
        source_to_group = dict(
            zip(
                cv["source_row_index"].astype(int),
                cv["feature_group_id"].astype(str),
            )
        )
        for train_positions, validation_positions in self.plan.cv_splits:
            self.assertEqual(
                set(np.unique(self.plan.y_train.iloc[train_positions])), {0, 1}
            )
            self.assertEqual(
                set(np.unique(self.plan.y_train.iloc[validation_positions])), {0, 1}
            )
            train_groups = {
                source_to_group[int(self.plan.source_row_indices[pos])]
                for pos in train_positions
            }
            validation_groups = {
                source_to_group[int(self.plan.source_row_indices[pos])]
                for pos in validation_positions
            }
            self.assertFalse(train_groups.intersection(validation_groups))
            seen[validation_positions] += 1
        np.testing.assert_array_equal(seen, np.ones(len(self.plan.X_train), dtype=np.int64))


class Phase6SelectionConfigurationUnitTests(unittest.TestCase):
    """Dataset-independent tests suitable for CI when Network_Data is gitignored."""

    @staticmethod
    def make_synthetic_training_plan() -> TrainingOnlySelectionPlan:
        # build_candidate_searches validates training row/fold counts but does not
        # inspect or fit the data. This deterministic synthetic plan lets CI verify
        # scorer, pipeline, and refit configuration without the canonical dataset.
        row_count = 8843
        positions = np.arange(row_count, dtype=np.int64)
        validation_folds = np.array_split(positions, 5)
        cv_splits = []
        for validation in validation_folds:
            mask = np.ones(row_count, dtype=bool)
            mask[validation] = False
            cv_splits.append((positions[mask], validation.copy()))

        X_train = pd.DataFrame(
            {"synthetic_feature": (positions % 3).astype(np.int64)}
        )
        y_train = pd.Series((positions % 2).astype(np.int64), name="Result")
        return TrainingOnlySelectionPlan(
            X_train=X_train,
            y_train=y_train,
            source_row_indices=positions.copy(),
            cv_splits=tuple(cv_splits),
            feature_columns=("synthetic_feature",),
            target_column="Result",
            protocol_hashes={},
            protocol_metadata={},
        )

    def test_candidate_searches_are_constructed_but_not_fitted(self):
        plan = self.make_synthetic_training_plan()
        searches = build_candidate_searches(plan)
        self.assertEqual(
            set(searches),
            {
                "Random Forest",
                "Decision Tree",
                "Gradient Boosting",
                "Logistic Regression",
                "AdaBoost",
            },
        )
        for search in searches.values():
            self.assertIsInstance(search, GridSearchCV)
            self.assertFalse(search.refit)
            self.assertEqual(search.scoring._score_func, f1_score)
            self.assertEqual(search.scoring._kwargs.get("pos_label"), 0)
            self.assertEqual(search.scoring._kwargs.get("zero_division"), 0)
            self.assertEqual(len(search.cv), 5)
            self.assertEqual(search.estimator.steps[0][0], "preprocessor")
            self.assertIsInstance(search.estimator.named_steps["preprocessor"], KNNImputer)
            self.assertEqual(search.estimator.named_steps["preprocessor"].n_neighbors, 3)
            self.assertEqual(search.estimator.steps[1][0], "classifier")

    def test_fitting_runner_is_disabled_without_separate_authorization(self):
        with patch.dict(os.environ, {FIT_AUTH_ENV: ""}, clear=False):
            with patch.object(GridSearchCV, "fit", autospec=True) as fit_mock:
                with self.assertRaises(Phase6FitDisabledError):
                    run_training_only_cv_selection(
                        PROJECT_ROOT,
                        allow_fitting=False,
                        confirmation=FIT_AUTH_TOKEN,
                    )
                fit_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
