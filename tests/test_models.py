from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from allen_brain_ml.models import (
    make_logistic_regression_pipeline,
)


def test_make_logistic_regression_pipeline():
    model = make_logistic_regression_pipeline()

    assert isinstance(model, Pipeline)
    assert list(model.named_steps) == [
        "scaler",
        "classifier",
    ]
    assert isinstance(
        model.named_steps["scaler"],
        StandardScaler,
    )
    assert isinstance(
        model.named_steps["classifier"],
        LogisticRegression,
    )
    assert model.named_steps["classifier"].class_weight is None


def test_make_logistic_regression_pipeline_sets_class_weight():
    model = make_logistic_regression_pipeline(
        class_weight="balanced"
    )

    assert (
        model.named_steps["classifier"].class_weight
        == "balanced"
    )