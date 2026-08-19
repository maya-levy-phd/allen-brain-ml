"""Model construction for Allen Brain classification."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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