"""Reusable model-evaluation utilities."""

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    cross_validate,
    cross_val_predict,
)


def evaluate_classifier(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    scoring: dict[str, str],
    params: dict[str, object] | None = None,
) -> tuple[dict[str, np.ndarray], pd.Series]:
    """Return CV scores and aligned out-of-fold predictions."""
    scores = cross_validate(
        estimator,
        X,
        y,
        cv=cv_splits,
        scoring=scoring,
        params=params,
        error_score="raise",
    )

    prediction_array = cross_val_predict(
        estimator,
        X,
        y,
        cv=cv_splits,
        params=params,
        method="predict",
    )

    predictions = pd.Series(
        prediction_array,
        index=y.index,
        name="predicted",
    )

    return scores, predictions
