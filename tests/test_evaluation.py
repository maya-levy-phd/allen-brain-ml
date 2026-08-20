import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier

from allen_brain_ml.evaluation import evaluate_classifier


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
