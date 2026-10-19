import yaml
import hashlib
from network_security.exception.exception import NetworkSecurityException
from network_security.logging.logger import logging
import os,sys
import numpy as np
import dill
import pickle
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold



def read_yaml_file(file_path:str) -> dict:
    try:
        with open(file_path,"rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise NetworkSecurityException(e,sys)
    
def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    
def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info("Entered the save_object method of MainUtils class")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Exited the save_object method of MainUtils class")
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
def load_object(file_path: str, ) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path, "rb") as file_obj:
                return pickle.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    


PHISHING_LABEL = 0
PHISHING_F1_SCORER = make_scorer(f1_score, pos_label=PHISHING_LABEL, zero_division=0)


def make_feature_group_ids(features):
    """Create stable IDs for identical finite feature rows (target excluded)."""
    try:
        values = np.array(features, dtype=np.float64, copy=True, order="C")
    except (TypeError, ValueError) as exc:
        raise ValueError("Features must be a finite numeric 2D array.") from exc

    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("Features must be a non-empty 2D array.")
    if not np.isfinite(values).all():
        raise ValueError("Features must not contain NaN or infinite values.")

    # Treat -0.0 and +0.0 as the same feature value before hashing.
    values[values == 0.0] = 0.0
    return np.asarray(
        [hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest() for row in values],
        dtype=object,
    )


def build_group_aware_cv_splits(X_train, y_train, n_splits=5, random_state=43):
    """Build training-only CV splits with identical feature rows kept together.

    This helper only creates indices; it never fits an estimator. These groups
    are derived from the exact matrix supplied to the estimator. The official
    Phase 2 holdout manifest remains a separate protocol for the canonical CSV.
    """
    X = np.asarray(X_train)
    y = np.asarray(y_train).reshape(-1)
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("X_train must be a non-empty 2D array.")
    if len(y) != X.shape[0]:
        raise ValueError("X_train and y_train must contain the same number of rows.")
    if set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("Training labels must contain both encoded classes 0 (phishing) and 1 (legitimate).")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    groups = make_feature_group_ids(X)
    if len(np.unique(groups)) < n_splits:
        raise ValueError("Not enough unique feature groups for the requested CV fold count.")

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = list(splitter.split(X, y, groups=groups))
    validation_counts = np.zeros(len(y), dtype=int)

    for fold_number, (train_indices, validation_indices) in enumerate(splits):
        train_groups = set(groups[train_indices].tolist())
        validation_groups = set(groups[validation_indices].tolist())
        if train_groups.intersection(validation_groups):
            raise ValueError(f"Feature-group leakage detected in CV fold {fold_number}.")
        if set(np.unique(y[train_indices]).tolist()) != {0, 1}:
            raise ValueError(f"CV fold {fold_number} training partition does not contain both classes.")
        if set(np.unique(y[validation_indices]).tolist()) != {0, 1}:
            raise ValueError(f"CV fold {fold_number} validation partition does not contain both classes.")
        validation_counts[validation_indices] += 1

    if not np.all(validation_counts == 1):
        raise ValueError("Every training row must appear in exactly one CV validation fold.")
    return splits


def evaluate_models(X_train, y_train, models, param):
    """Rank candidate classifiers using training-only grouped CV phishing F1.

    Test/holdout arrays are deliberately not accepted by this function, so they
    cannot influence hyperparameter search or candidate model selection.
    """
    try:
        cv_splits = build_group_aware_cv_splits(X_train, y_train)
        report = {}

        for model_name, model in models.items():
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param.get(model_name, {}),
                scoring=PHISHING_F1_SCORER,
                cv=cv_splits,
                refit=False,
            )
            grid_search.fit(X_train, y_train)
            cv_score = float(grid_search.best_score_)
            if not np.isfinite(cv_score):
                raise ValueError(f"Model {model_name!r} produced a non-finite CV score.")
            model.set_params(**grid_search.best_params_)
            report[model_name] = cv_score

        return report
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
