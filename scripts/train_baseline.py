from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedGroupKFold

from allen_brain_ml.data import (
    load_or_fetch_cell_records,
    prepare_cell_data,
)
from allen_brain_ml.datasets import (
    GROUP_COLUMN,
    TARGET_COLUMN,
    build_classification_cohort,
    make_grouped_holdout_split,
)
from allen_brain_ml.models import (
    make_logistic_regression_pipeline,
)
from allen_brain_ml.features import get_ephys_feature_columns
from allen_brain_ml.evaluation import (
    evaluate_classifier,
    make_paired_fold_comparison,
    make_class_fold_diagnostics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = PROJECT_ROOT / "data" / "raw" / "cells.json"
RANDOM_STATE = 42
N_CV_SPLITS = 5

SCORING = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "macro_f1": "f1_macro",
}


def print_split_audit(
    name: str,
    y: pd.Series,
    groups: pd.Series,
) -> None:
    """Print sample, donor, and class distributions for one split."""
    class_summary = pd.DataFrame(
        {
            "count": y.value_counts(),
            "fraction": y.value_counts(normalize=True),
        }
    )

    print(f"\n{name} split")
    print(f"Cells: {len(y)}")
    print(f"Donors: {groups.nunique()}")
    print(class_summary)


def print_cross_validation_results(
    model_name: str,
    results: dict,
) -> None:
    """Print the mean and standard deviation of CV scores."""
    print(f"\n{model_name} cross-validation")

    for metric in SCORING:
        scores = results[f"test_{metric}"]
        print(
            f"{metric}: "
            f"{scores.mean():.3f} +/- {scores.std():.3f}"
        )



def print_classification_diagnostics(
    model_name: str,
    actual_classes: pd.Series,
    predicted_classes: pd.Series,
) -> None:
    """Print per-class metrics and confusion matrices."""
    actual_classes = actual_classes.rename("actual")
    predicted_classes = predicted_classes.rename("predicted")

    print(f"\n{model_name} classification report")
    print(
        classification_report(
            actual_classes,
            predicted_classes,
            digits=3,
            zero_division=0,
        )
    )

    confusion_counts = pd.crosstab(
        actual_classes,
        predicted_classes,
        margins=True,
    )

    print("\nConfusion matrix: cell counts")
    print(confusion_counts)

    confusion_proportions = pd.crosstab(
        actual_classes,
        predicted_classes,
        normalize="index",
    )

    print("\nConfusion matrix: row proportions")
    print(confusion_proportions)


def main() -> None:
    """Load the data and assemble the classification dataset."""
    records = load_or_fetch_cell_records(CACHE_PATH)
    cells = prepare_cell_data(records)

    modeling_cells = build_classification_cohort(cells)
    ephys_columns = get_ephys_feature_columns(modeling_cells)

    X = modeling_cells.loc[:, ephys_columns]
    y = modeling_cells.loc[:, TARGET_COLUMN]
    groups = modeling_cells.loc[:, GROUP_COLUMN]

    print(f"Samples: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Donors: {groups.nunique()}")
    print("\nClass counts:")
    print(y.value_counts())

    development_indices, test_indices = (
        make_grouped_holdout_split(y, groups)
    )

    X_development = X.iloc[development_indices]
    y_development = y.iloc[development_indices]
    groups_development = groups.iloc[development_indices]

    X_test = X.iloc[test_indices]
    y_test = y.iloc[test_indices]
    groups_test = groups.iloc[test_indices]

    print_split_audit(
        "Development",
        y_development,
        groups_development,
    )
    print_split_audit(
        "Test",
        y_test,
        groups_test,
    )

    donor_overlap = set(groups_development).intersection(groups_test)
    print(f"\nDonor overlap: {len(donor_overlap)}")

    cross_validator = StratifiedGroupKFold(
        n_splits=N_CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    dummy_model = DummyClassifier(
        strategy="most_frequent",
    )

    cross_validation_splits = list(
        cross_validator.split(
            X_development,
            y_development,
            groups_development,
        )
    )

    dummy_results, _ = evaluate_classifier(
        dummy_model,
        X_development,
        y_development,
        cv_splits=cross_validation_splits,
        scoring=SCORING,
    )

    logistic_model = make_logistic_regression_pipeline()

    logistic_results, logistic_predicted_classes = (
        evaluate_classifier(
            logistic_model,
            X_development,
            y_development,
            cv_splits=cross_validation_splits,
            scoring=SCORING,
        )
    )

    print_cross_validation_results(
        "Unweighted logistic regression",
        logistic_results,
    )

    print_classification_diagnostics(
        "Unweighted logistic regression",
        y_development,
        logistic_predicted_classes,
    )

    class_weighted_logistic_model = (
        make_logistic_regression_pipeline(
            class_weight="balanced"
        )
    )

    (
        class_weighted_results,
        class_weighted_predicted_classes,
    ) = evaluate_classifier(
        class_weighted_logistic_model,
        X_development,
        y_development,
        cv_splits=cross_validation_splits,
        scoring=SCORING,
    )

    print_cross_validation_results(
        "Class-weighted logistic regression",
        class_weighted_results,
    )

    print_classification_diagnostics(
        "Class-weighted logistic regression",
        y_development,
        class_weighted_predicted_classes,
    )

    fold_comparison = make_paired_fold_comparison(
        logistic_results,
        class_weighted_results,
        metric="macro_f1",
        reference_name="unweighted",
        comparison_name="weighted",
    )

    print("\nPaired macro-F1 comparison")
    print(fold_comparison)

    SPARSE_CLASS = "sparsely spiny"

    fold_diagnostics_df = make_class_fold_diagnostics(
        y_development,
        groups_development,
        {
            "unweighted": logistic_predicted_classes,
            "weighted": class_weighted_predicted_classes,
        },
        cv_splits=cross_validation_splits,
        class_label=SPARSE_CLASS,
    )

    formatted_diagnostics = (
        fold_diagnostics_df
        .rename(
            columns={
                "class_cells": "sparse_cells",
                "class_groups": "sparse_donors",
            }
        )
        .round(3)
    )

    print("\nSparse-class diagnostics by fold")
    print(formatted_diagnostics.to_string(index=False))





if __name__ == "__main__":
    main()
