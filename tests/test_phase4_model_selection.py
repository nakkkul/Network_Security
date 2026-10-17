"""Safe tests for model-selection configuration; no estimator is fitted."""
import ast
import inspect
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from network_security.utils.main_utils import utils


class DummyEstimator:
    def __init__(self):
        self.params = {}

    def set_params(self, **params):
        self.params.update(params)
        return self

    def fit(self, *args, **kwargs):  # pragma: no cover - must never be called by these tests
        raise AssertionError("Tests must not fit an estimator.")


class FakeGridSearchCV:
    """Minimal GridSearchCV double: captures configuration without fitting models."""
    instances = []

    def __init__(self, estimator, param_grid, scoring, cv, refit):
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv
        self.refit = refit
        self.best_score_ = 0.75
        self.best_params_ = {"example_parameter": 7} if param_grid else {}
        self.fit_calls = []
        self.__class__.instances.append(self)

    def fit(self, X, y):
        # Record arguments only. Do not call estimator.fit or any sklearn fitting API.
        self.fit_calls.append((X, y))
        return self


def sample_training_data():
    unique_rows = np.asarray([[i, i % 3, i % 5] for i in range(20)], dtype=float)
    X = np.repeat(unique_rows, repeats=2, axis=0)
    y = np.repeat(np.asarray([0] * 10 + [1] * 10, dtype=int), repeats=2)
    return X, y


class Phase4ModelSelectionTests(unittest.TestCase):
    def setUp(self):
        FakeGridSearchCV.instances.clear()

    def test_duplicate_feature_rows_receive_identical_group_ids(self):
        X, _ = sample_training_data()
        groups = utils.make_feature_group_ids(X)
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(len(np.unique(groups)), len(X) // 2)

    def test_group_split_keeps_duplicate_rows_together_and_covers_each_row_once(self):
        X, y = sample_training_data()
        groups = utils.make_feature_group_ids(X)
        splits = utils.build_group_aware_cv_splits(X, y, n_splits=5, random_state=43)
        self.assertEqual(len(splits), 5)

        validation_counts = np.zeros(len(y), dtype=int)
        for train_indices, validation_indices in splits:
            self.assertFalse(set(groups[train_indices]).intersection(set(groups[validation_indices])))
            self.assertEqual(set(np.unique(y[train_indices])), {0, 1})
            self.assertEqual(set(np.unique(y[validation_indices])), {0, 1})
            validation_counts[validation_indices] += 1
        np.testing.assert_array_equal(validation_counts, np.ones(len(y), dtype=int))

    def test_non_finite_features_are_rejected_before_any_search(self):
        X, _ = sample_training_data()
        X[0, 0] = np.nan
        with self.assertRaises(ValueError):
            utils.make_feature_group_ids(X)
        self.assertEqual(FakeGridSearchCV.instances, [])

    def test_training_labels_must_contain_both_encoded_classes(self):
        X, y = sample_training_data()
        with self.assertRaises(ValueError):
            utils.build_group_aware_cv_splits(X, np.zeros_like(y))

    def test_model_selection_api_does_not_accept_test_or_holdout_data(self):
        parameter_names = set(inspect.signature(utils.evaluate_models).parameters)
        self.assertEqual(parameter_names, {"X_train", "y_train", "models", "param"})
        self.assertNotIn("X_test", parameter_names)
        self.assertNotIn("y_test", parameter_names)

    def test_search_uses_phishing_f1_grouped_cv_and_no_implicit_refit(self):
        X, y = sample_training_data()
        estimator = DummyEstimator()
        with patch.object(utils, "GridSearchCV", FakeGridSearchCV):
            report = utils.evaluate_models(
                X_train=X,
                y_train=y,
                models={"Toy classifier": estimator},
                param={"Toy classifier": {"example_parameter": [1, 2]}},
            )

        self.assertEqual(report, {"Toy classifier": 0.75})
        self.assertEqual(estimator.params, {"example_parameter": 7})
        self.assertEqual(len(FakeGridSearchCV.instances), 1)
        search = FakeGridSearchCV.instances[0]
        self.assertFalse(search.refit)
        self.assertEqual(len(search.cv), 5)
        self.assertEqual(len(search.fit_calls), 1)
        self.assertEqual(search.scoring._kwargs.get("pos_label"), 0)
        self.assertEqual(search.scoring._kwargs.get("zero_division"), 0)
        self.assertTrue(all(len(args) == 2 for args in search.fit_calls))

    def test_model_trainer_calls_selector_with_training_data_only_and_fits_after_selection(self):
        project_root = Path(__file__).resolve().parents[1]
        source_path = project_root / "network_security" / "components" / "model_trainer.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        trainer = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ModelTrainer"
        )
        train_method = next(
            node for node in trainer.body
            if isinstance(node, ast.FunctionDef) and node.name == "train_model"
        )
        calls = [node for node in ast.walk(train_method) if isinstance(node, ast.Call)]
        selection_call = next(
            node for node in calls
            if isinstance(node.func, ast.Name) and node.func.id == "evaluate_models"
        )
        keyword_names = {keyword.arg for keyword in selection_call.keywords}
        self.assertEqual(keyword_names, {"X_train", "y_train", "models", "param"})

        fit_line = min(
            node.lineno for node in calls
            if isinstance(node.func, ast.Attribute)
            and node.func.attr == "fit"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "best_model"
        )
        prediction_line = min(
            node.lineno for node in calls
            if isinstance(node.func, ast.Attribute)
            and node.func.attr == "predict"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "best_model"
        )
        self.assertLess(selection_call.lineno, fit_line)
        self.assertLess(fit_line, prediction_line)


if __name__ == "__main__":
    unittest.main()
