"""Model construction for Allen Brain classification."""

from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    SplineTransformer,
)


def make_decision_tree_classifier(
    *,
    class_weight: str | dict | None = None,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    ccp_alpha: float = 0.0,
    random_state: int = 42,
) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(
        class_weight=class_weight,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        ccp_alpha=ccp_alpha,
        random_state=random_state,
    )


def make_logistic_regression_pipeline(
    *,
    class_weight: str | dict[str, float] | None = None,
) -> Pipeline:
    """Construct a scaled logistic regression classifier."""
    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    l1_ratio=0.0,
                    solver="lbfgs",
                    max_iter=1_000,
                    class_weight=class_weight,
                ),
            ),
        ]
    )


def make_spline_logistic_regression_pipeline(
    *,
    spline_features: list[str],
    linear_features: list[str],
    class_weight: str | dict[str, float] | None = None,
    n_knots: int = 5,
    degree: int = 3,
) -> Pipeline:
    """Build logistic regression with spline and linear features."""
    spline_pipeline = Pipeline(
        steps=[
            (
                "spline",
                SplineTransformer(
                    n_knots=n_knots,
                    degree=degree,
                    knots="quantile",
                    include_bias=False,
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "spline",
                spline_pipeline,
                spline_features,
            ),
            (
                "linear",
                StandardScaler(),
                linear_features,
            ),
        ],
        remainder="drop",
    )

    classifier = LogisticRegression(
        C=1.0,
        l1_ratio=0.0,
        solver="lbfgs",
        max_iter=1_000,
        class_weight=class_weight,
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )