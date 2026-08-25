import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier

from allen_brain_ml.datasets import make_donor_sample_weights
from allen_brain_ml.evaluation import (
    evaluate_classifier,
    make_paired_fold_comparison,
    make_class_fold_diagnostics,
    evaluate_fold_weighted_classifier,
    evaluate_tree_complexity,
)
from allen_brain_ml.models import make_logistic_regression_pipeline


def test_evaluate_tree_complexity():
    X = pd.DataFrame(
        {
            "feature": [
                0.0,
                1.0,
                2.0,
                3.0,
                10.0,
                11.0,
                12.0,
                13.0,
            ]
        }
    )
    y = pd.Series(
        [
            "non-spiny",
            "non-spiny",
            "non-spiny",
            "non-spiny",
            "spiny",
            "spiny",
            "spiny",
            "spiny",
        ]
    )

    cv_splits = [
        (
            np.array([0, 1, 4, 5]),
            np.array([2, 3, 6, 7]),
        ),
        (
            np.array([2, 3, 6, 7]),
            np.array([0, 1, 4, 5]),
        ),
    ]

    result = evaluate_tree_complexity(
        estimator=DecisionTreeClassifier(
            random_state=42
        ),
        X=X,
        y=y,
        cv_splits=cv_splits,
    )

    assert result["fold"].tolist() == [1, 2]
    assert result["depth"].tolist() == [1, 1]
    assert result["leaves"].tolist() == [2, 2]
    assert result["min_leaf_cells"].tolist() == [2, 2]

    np.testing.assert_allclose(
        result["train_macro_f1"],
        [1.0, 1.0],
    )
    np.testing.assert_allclose(
        result["validation_macro_f1"],
        [1.0, 1.0],
    )
    np.testing.assert_allclose(
        result["generalization_gap"],
        [0.0, 0.0],
    )


def test_evaluate_fold_weighted_classifier_builds_weights_per_fold():
    X = pd.DataFrame(
        {
            "feature": [
                -4.0,
                -3.0,
                -2.0,
                -1.0,
                1.0,
                2.0,
                3.0,
                4.0,
            ]
        },
        index=[10, 20, 30, 40, 50, 60, 70, 80],
    )
    y = pd.Series(
        [
            "aspiny",
            "spiny",
            "aspiny",
            "spiny",
            "aspiny",
            "spiny",
            "aspiny",
            "spiny",
        ],
        index=X.index,
    )
    groups = pd.Series(
        [101, 101, 202, 202, 303, 303, 404, 404],
        index=X.index,
        name="donor__id",
    )
    cv = GroupKFold(n_splits=2)
    cv_splits = list(
        cv.split(
            X,
            y,
            groups,
        )
    )

    weight_factory_inputs = []

    def recording_weight_factory(
        training_groups: pd.Series,
    ) -> pd.Series:
        weight_factory_inputs.append(training_groups.copy())
        return make_donor_sample_weights(training_groups)

    model = make_logistic_regression_pipeline(
        class_weight=None,
    )

    scores, predictions = evaluate_fold_weighted_classifier(
        estimator=model,
        X=X,
        y=y,
        groups=groups,
        cv_splits=cv_splits,
        scoring={"accuracy": "accuracy"},
        sample_weight_factory=recording_weight_factory,
    )

    expected_training_groups = []

    for training_indices, validation_indices in cv_splits:
        fold_training_groups = groups.iloc[training_indices]
        expected_training_groups.append(
            fold_training_groups
        )

    assert len(weight_factory_inputs) == len(cv_splits)
    assert len(expected_training_groups) == len(cv_splits)

    for actual, expected in zip(
        weight_factory_inputs,
        expected_training_groups,
        strict=True,
    ):
        pd.testing.assert_series_equal(actual, expected)

    assert len(predictions) == len(y)
    assert predictions.name == "predicted"
    assert "test_accuracy" in scores


def test_evaluate_classifier_returns_scores_and_predictions():
    X = pd.DataFrame(
        {"feature": [1, 2, 3, 4, 5, 6]},
        index=[10, 20, 30, 40, 50, 60],
    )
    y = pd.Series(
        [
            "aspiny",
            "aspiny",
            "aspiny",
            "spiny",
            "spiny",
            "spiny",
        ],
        index=X.index,
        name="target",
    )

    cv_splits = [
        (
            np.array([1, 2, 4, 5]),
            np.array([0, 3]),
        ),
        (
            np.array([0, 2, 3, 5]),
            np.array([1, 4]),
        ),
        (
            np.array([0, 1, 3, 4]),
            np.array([2, 5]),
        ),
    ]

    scores, predictions = evaluate_classifier(
        DummyClassifier(strategy="most_frequent"),
        X,
        y,
        cv_splits=cv_splits,
        scoring={"accuracy": "accuracy"},
    )

    np.testing.assert_allclose(
        scores["test_accuracy"],
        [0.5, 0.5, 0.5],
    )
    pd.testing.assert_index_equal(
        predictions.index,
        y.index,
    )
    assert predictions.name == "predicted"
    assert len(predictions) == len(y)


def test_make_paired_fold_comparison():
    reference_scores = {
        "test_macro_f1": np.array([0.25, 0.50, 0.75])
    }
    comparison_scores = {
        "test_macro_f1": np.array([0.50, 0.25, 1.00])
    }

    result = make_paired_fold_comparison(
        reference_scores,
        comparison_scores,
        metric="macro_f1",
        reference_name="unweighted",
        comparison_name="weighted",
    )

    expected = pd.DataFrame(
        {
            "fold": [1, 2, 3],
            "unweighted_macro_f1": [0.25, 0.50, 0.75],
            "weighted_macro_f1": [0.50, 0.25, 1.00],
            "macro_f1_difference": [0.25, -0.25, 0.25],
        }
    )

    pd.testing.assert_frame_equal(result, expected)


def test_make_class_fold_diagnostics():
    index = [10, 20, 30, 40, 50, 60]

    y = pd.Series(
        [
            "sparse",
            "aspiny",
            "spiny",
            "sparse",
            "aspiny",
            "spiny",
        ],
        index=index,
    )
    groups = pd.Series(
        [1, 2, 3, 4, 5, 6],
        index=index,
    )

    predictions_by_model = {
        "unweighted": pd.Series(
            [
                "aspiny",
                "aspiny",
                "spiny",
                "sparse",
                "aspiny",
                "aspiny",
            ],
            index=index,
        ),
        "weighted": pd.Series(
            [
                "sparse",
                "sparse",
                "spiny",
                "sparse",
                "sparse",
                "spiny",
            ],
            index=index,
        ),
    }

    cv_splits = [
        (
            np.array([3, 4, 5]),
            np.array([0, 1, 2]),
        ),
        (
            np.array([0, 1, 2]),
            np.array([3, 4, 5]),
        ),
    ]

    result = make_class_fold_diagnostics(
        y,
        groups,
        predictions_by_model,
        cv_splits=cv_splits,
        class_label="sparse",
    )

    expected = pd.DataFrame(
        {
            "fold": [1, 1, 2, 2],
            "model": [
                "unweighted",
                "weighted",
                "unweighted",
                "weighted",
            ],
            "class_cells": [1, 1, 1, 1],
            "class_groups": [1, 1, 1, 1],
            "precision": [0.0, 0.5, 1.0, 0.5],
            "recall": [0.0, 1.0, 1.0, 1.0],
            "f1": [0.0, 2 / 3, 1.0, 2 / 3],
        }
    )

    pd.testing.assert_frame_equal(result, expected)


