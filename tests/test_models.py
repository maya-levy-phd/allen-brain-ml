from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    SplineTransformer,
)

from allen_brain_ml.models import (
    make_logistic_regression_pipeline,
    make_spline_logistic_regression_pipeline,
    make_decision_tree_classifier,
)


def test_make_decision_tree_classifier():
    model = make_decision_tree_classifier(
        class_weight="balanced",
        max_depth=4,
        min_samples_leaf=10,
        ccp_alpha=0.01,
    )

    assert isinstance(model, DecisionTreeClassifier)
    assert model.class_weight == "balanced"
    assert model.max_depth == 4
    assert model.min_samples_leaf == 10
    assert model.ccp_alpha == 0.01
    assert model.random_state == 42


def test_make_spline_logistic_regression_pipeline():
    spline_features = [
        "firing_rate",
        "peak_time",
    ]
    linear_features = [
        "vrest",
    ]

    model = make_spline_logistic_regression_pipeline(
        spline_features=spline_features,
        linear_features=linear_features,
        class_weight="balanced",
        n_knots=5,
        degree=3,
    )

    assert isinstance(model, Pipeline)

    preprocessor = model.named_steps["preprocessor"]
    assert isinstance(
        preprocessor,
        ColumnTransformer,
    )

    (
        spline_name,
        spline_pipeline,
        selected_spline_features,
    ) = preprocessor.transformers[0]

    assert spline_name == "spline"
    assert selected_spline_features == spline_features
    assert isinstance(spline_pipeline, Pipeline)

    spline_transformer = (
        spline_pipeline.named_steps["spline"]
    )
    assert isinstance(
        spline_transformer,
        SplineTransformer,
    )
    assert spline_transformer.n_knots == 5
    assert spline_transformer.degree == 3
    assert spline_transformer.knots == "quantile"
    assert spline_transformer.include_bias is False

    assert isinstance(
        spline_pipeline.named_steps["scaler"],
        StandardScaler,
    )

    (
        linear_name,
        linear_transformer,
        selected_linear_features,
    ) = preprocessor.transformers[1]

    assert linear_name == "linear"
    assert selected_linear_features == linear_features
    assert isinstance(
        linear_transformer,
        StandardScaler,
    )

    classifier = model.named_steps["classifier"]
    assert classifier.class_weight == "balanced"


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