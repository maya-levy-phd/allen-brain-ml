"""Reusable model-evaluation utilities."""
from collections.abc import Callable

import numpy as np
import pandas as pd

from sklearn.base import (
    BaseEstimator,
    clone,
)
from sklearn.metrics import (
    classification_report,
    get_scorer,
)
from sklearn.model_selection import (
    cross_validate,
    cross_val_predict,
)


def evaluate_fold_weighted_classifier(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    cv_splits: list[
        tuple[np.ndarray, np.ndarray]
    ],
    scoring: dict[str, str],
    sample_weight_factory: Callable[[pd.Series], pd.Series],
) -> tuple[dict[str, np.ndarray], pd.Series]:
    """Evaluate a classifier using fold-specific training weights."""
    scorers = {
        score_name: get_scorer(scorer_name)
        for score_name, scorer_name in scoring.items()
    }
    fold_scores = {
        f"test_{score_name}": []
        for score_name in scoring
    }
    prediction_values = np.empty(len(y), dtype=object)

    for training_indices, validation_indices in cv_splits:
        fold_estimator = clone(estimator)

        X_train = X.iloc[training_indices]
        y_train = y.iloc[training_indices]
        groups_train = groups.iloc[training_indices]

        X_validation = X.iloc[validation_indices]
        y_validation = y.iloc[validation_indices]

        training_weights = sample_weight_factory(
            groups_train
        )

        fold_estimator.fit(
            X_train,
            y_train,
            scaler__sample_weight=training_weights,
            classifier__sample_weight=training_weights,
        )

        prediction_values[validation_indices] = (
            fold_estimator.predict(X_validation)
        )

        for score_name, scorer in scorers.items():
            fold_score = scorer(
                fold_estimator,
                X_validation,
                y_validation,
            )
            fold_scores[f"test_{score_name}"].append(
                fold_score
            )

    scores = {
        score_name: np.asarray(score_values)
        for score_name, score_values in fold_scores.items()
    }
    predictions = pd.Series(
        prediction_values,
        index=y.index,
        name="predicted",
    )

    return scores, predictions


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


def make_paired_fold_comparison(
    reference_scores: dict[str, np.ndarray],
    comparison_scores: dict[str, np.ndarray],
    *,
    metric: str,
    reference_name: str,
    comparison_name: str,
) -> pd.DataFrame:
    """Compare two models on the same CV folds."""
    score_key = f"test_{metric}"

    reference_values = reference_scores[score_key]
    comparison_values = comparison_scores[score_key]

    fold_comparison = pd.DataFrame(
        {
            "fold": range(1, len(reference_values) + 1),
            f"{reference_name}_{metric}": reference_values,
            f"{comparison_name}_{metric}": comparison_values,
        }
    )

    fold_comparison[f"{metric}_difference"] = (
            fold_comparison[f"{comparison_name}_{metric}"]
            - fold_comparison[f"{reference_name}_{metric}"]
    )

    return fold_comparison


def make_class_fold_diagnostics(
    y: pd.Series,
    groups: pd.Series,
    predictions_by_model: dict[str, pd.Series],
    *,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    class_label: str,
) -> pd.DataFrame:
    """Return per-fold metrics for one target class."""
    diagnostic_rows = []

    for fold_number, (_, validation_indices) in enumerate(
            cv_splits,
            start=1,
    ):
        actual_fold = y.iloc[validation_indices]
        groups_fold = groups.iloc[validation_indices]

        class_mask = actual_fold.eq(class_label)
        class_cells = class_mask.sum()
        class_groups = (
            groups_fold.loc[class_mask].nunique()
        )

        for model_name, predictions in (
                predictions_by_model.items()
        ):
            predictions_fold = predictions.iloc[
                validation_indices
            ]

            report = classification_report(
                actual_fold,
                predictions_fold,
                labels=[class_label],
                output_dict=True,
                zero_division=0,
            )
            class_metrics = report[class_label]

            diagnostic_rows.append(
                {
                    "fold": fold_number,
                    "model": model_name,
                    "class_cells": class_cells,
                    "class_groups": class_groups,
                    "precision": class_metrics["precision"],
                    "recall": class_metrics["recall"],
                    "f1": class_metrics["f1-score"],
                }
            )

    return pd.DataFrame(diagnostic_rows)

