"""Models and cross-validation for parcel-level rice classification."""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PHYSICS_FEATURES = ["vh_flood_min", "vh_peak_max", "vh_rise", "vv_flood_min", "ratio_flood_min", "ratio_peak_mean"]


def grid_features(data: pd.DataFrame) -> list[str]:
    """The fixed-time-grid columns, e.g. `VH_db_05`."""
    return [col for col in data.columns if col.startswith(("VV_db_", "VH_db_", "VH_VV_db_"))]


class PhysicsRule(ClassifierMixin, BaseEstimator):
    """Hand-made rule: rice if the parcel got dark during flooding AND brightened afterwards.

    `vh_flood_min < threshold_db` and `vh_rise > rise_db`. The two thresholds are not set by hand: `fit` picks the
    pair with the best F1 on the training data, so in cross-validation they never see the validation fold.
    """

    def __init__(self, flood_feature: str = "vh_flood_min", rise_feature: str = "vh_rise"):
        self.flood_feature = flood_feature
        self.rise_feature = rise_feature

    def fit(self, X: pd.DataFrame, y: pd.Series) -> PhysicsRule:
        flood, rise, y = X[self.flood_feature].to_numpy(), X[self.rise_feature].to_numpy(), np.asarray(y)
        best = (-1.0, 0.0, 0.0)
        for threshold in np.arange(-30, -15.9, 0.25):
            for rise_min in np.arange(0, 12.1, 0.5):
                score = f1_score(y, (flood < threshold) & (rise > rise_min), zero_division=0)
                if score > best[0]:
                    best = (score, threshold, rise_min)
        _, self.threshold_db_, self.rise_db_ = best
        self.classes_ = np.array([False, True])
        return self

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        """Distance (dB) to the rule's boundary; positive = rice. Used to rank parcels for PR-AUC."""
        return np.minimum(self.threshold_db_ - X[self.flood_feature], X[self.rise_feature] - self.rise_db_).to_numpy()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        score = 1 / (1 + np.exp(-self.decision_function(X)))
        return np.column_stack([1 - score, score])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.decision_function(X) > 0


def _logistic_regression() -> Any:
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))


def _lightgbm() -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=0,
        verbose=-1,
    )


def make_models(grid_columns: list[str]) -> dict[str, tuple[Any, list[str]]]:
    """Models from simplest to most flexible, each with the feature columns it uses."""
    return {
        "majority": (DummyClassifier(strategy="most_frequent"), PHYSICS_FEATURES),
        "physics rule": (PhysicsRule(), ["vh_flood_min", "vh_rise"]),
        "logistic regression": (_logistic_regression(), PHYSICS_FEATURES),
        "lightgbm": (_lightgbm(), PHYSICS_FEATURES + grid_columns),
    }


def make_ablation_models(grid_columns: list[str]) -> dict[str, tuple[Any, list[str]]]:
    """The two missing combinations of model type × inputs, to separate the effect of the model from that of the
    inputs: the linear model with the full time series, and the trees with only the six physics features."""
    return {
        "logistic regression + series": (_logistic_regression(), PHYSICS_FEATURES + grid_columns),
        "lightgbm, 6 features only": (_lightgbm(), PHYSICS_FEATURES),
    }


def scores(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_score),
    }


def cross_validate(
    models: dict[str, tuple[Any, list[str]]],
    data: pd.DataFrame,
    folds: Iterator[tuple[np.ndarray, np.ndarray]] | list[tuple[np.ndarray, np.ndarray]],
    label: str = "is_rice",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fits every model on each training fold and scores it on the held-out fold.

    Returns per-fold scores (one row per model and fold) and out-of-fold predictions (each parcel is predicted by
    the model that did not see it during training).
    """
    y = data[label].astype(bool).to_numpy()
    fold_rows, oof = [], pd.DataFrame(index=data.index)
    for fold, (train_idx, test_idx) in enumerate(folds):
        for name, (estimator, columns) in models.items():
            model = clone(estimator).fit(data.iloc[train_idx][columns], y[train_idx])
            proba = model.predict_proba(data.iloc[test_idx][columns])[:, 1]
            pred = model.predict(data.iloc[test_idx][columns]).astype(bool)
            fold_rows.append({"model": name, "fold": fold, **scores(y[test_idx], pred, proba)})
            oof.loc[data.index[test_idx], f"{name}_proba"] = proba
            oof.loc[data.index[test_idx], f"{name}_pred"] = pred
    return pd.DataFrame(fold_rows), oof


def block_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    blocks: np.ndarray,
    n: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """95 % confidence intervals of the test scores by resampling whole spatial blocks (with replacement).

    Parcels in one block are not independent (same farms, water, soil), so resampling single parcels would give
    intervals that are too narrow.
    """
    y_true, y_pred, y_score, blocks = map(np.asarray, (y_true, y_pred, y_score, blocks))
    unique_blocks = np.unique(blocks)
    members = {block: np.nonzero(blocks == block)[0] for block in unique_blocks}
    rng = np.random.default_rng(seed)

    samples = []
    for _ in range(n):
        idx = np.concatenate([members[block] for block in rng.choice(unique_blocks, size=len(unique_blocks))])
        if y_true[idx].any():
            samples.append(scores(y_true[idx], y_pred[idx], y_score[idx]))
    samples = pd.DataFrame(samples)
    return pd.DataFrame({"ci_low": samples.quantile(0.025), "ci_high": samples.quantile(0.975)})


def summarize(fold_scores: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std over folds per model."""
    stats = fold_scores.drop(columns="fold").groupby("model", sort=False).agg(["mean", "std"])
    return stats.round(3)
