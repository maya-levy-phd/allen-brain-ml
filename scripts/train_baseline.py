from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
    make_donor_sample_weights,
)
from allen_brain_ml.models import (
    make_logistic_regression_pipeline,
)
from allen_brain_ml.features import get_ephys_feature_columns
from allen_brain_ml.evaluation import (
    evaluate_classifier,
    make_paired_fold_comparison,
    make_class_fold_diagnostics,
    evaluate_fold_weighted_classifier,
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


def run_feature_audit(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    save_figures: bool = False,
) -> None:
    """Print feature audits and optionally save figures."""
    correlation_pairs = make_feature_correlation_table(X)

    print("\nStrongest Spearman feature correlations")
    print(
        correlation_pairs.head(15).to_string(
            index=False
        )
    )

    rate_and_isi = X[
        [
            "ef__avg_firing_rate",
            "ef__avg_isi",
        ]
    ]

    print("\nPearson correlation")
    print(rate_and_isi.corr(method="pearson"))

    rate_isi_product = (
        rate_and_isi["ef__avg_firing_rate"]
        * rate_and_isi["ef__avg_isi"]
    )

    print("\nFiring-rate × ISI product")
    print(rate_isi_product.describe())

    distribution_summary = (
        make_feature_distribution_summary(X)
    )
    distribution_summary["excess_kurtosis"] = (
        X.kurt()
    )

    print("\nEphys feature-distribution summary")
    print(
        distribution_summary
        .round(3)
        .to_string()
    )

    tail_summary = (
        distribution_summary[
            ["skew", "excess_kurtosis"]
        ]
        .sort_values(
            "excess_kurtosis",
            ascending=False,
        )
    )

    print("\nFeature skewness and excess kurtosis")
    print(tail_summary.round(3).to_string())

    grouped_features = X.groupby(
        y,
        observed=True,
    )

    class_medians = grouped_features.median().T
    class_q25 = grouped_features.quantile(0.25)
    class_q75 = grouped_features.quantile(0.75)
    class_iqr = class_q75.sub(class_q25).T

    print("\nMedian ephys features by dendrite class")
    print(class_medians.round(3).to_string())

    print("\nEphys feature IQR by dendrite class")
    print(class_iqr.round(3).to_string())

    if save_figures:
        plot_firing_rate_vs_isi(
            X,
            PROJECT_ROOT
            / "reports"
            / "figures"
            / "firing_rate_vs_isi.png",
        )
        save_feature_histograms(
            X,
            PROJECT_ROOT
            / "reports"
            / "figures"
            / "ephys_feature_distributions.png",
        )
        save_class_feature_boxplots(
            X,
            y,
            PROJECT_ROOT
            / "reports"
            / "figures"
            / "ephys_features_by_class.png",
        )


def save_class_feature_boxplots(
    X: pd.DataFrame,
    y: pd.Series,
    output_path: Path,
) -> None:
    """Save class-specific boxplots of model features."""
    plot_data = X.copy()
    plot_data["dendrite_type"] = y

    class_order = [
        "aspiny",
        "sparsely spiny",
        "spiny",
    ]
    plot_data["dendrite_type"] = pd.Categorical(
        plot_data["dendrite_type"],
        categories=class_order,
        ordered=True,
    )

    figure, axes = plt.subplots(
        3,
        4,
        figsize=(16, 11),
    )
    flat_axes = axes.ravel()

    for ax, feature in zip(
        flat_axes,
        X.columns,
    ):
        plot_data.boxplot(
            column=feature,
            by="dendrite_type",
            ax=ax,
            grid=False,
            showfliers=False,
        )

        ax.set_title(
            feature.removeprefix("ef__")
        )
        ax.set_xlabel("")
        ax.tick_params(
            axis="x",
            labelrotation=20,
        )

    for ax in flat_axes[len(X.columns):]:
        ax.set_visible(False)

    figure.suptitle(
        "Ephys feature distributions by dendrite class",
        fontsize=16,
    )
    figure.tight_layout(
        rect=(0, 0, 1, 0.96)
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_feature_histograms(
    X: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save histograms of all model features."""
    display_features = X.rename(
        columns=lambda column: column.removeprefix("ef__")
    )

    axes = display_features.hist(
        bins=30,
        figsize=(16, 10),
        layout=(3, 4),
        edgecolor="white",
    )

    for ax in axes.ravel():
        if ax.has_data():
            ax.set_ylabel("Cell count")
        else:
            ax.set_visible(False)

    figure = axes.ravel()[0].get_figure()
    figure.suptitle(
        "Development-set ephys feature distributions",
        fontsize=16,
    )
    figure.tight_layout(
        rect=(0, 0, 1, 0.96)
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_firing_rate_vs_isi(
    X: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save a plot of firing rate against average ISI."""
    ax = X.plot.scatter(
        x="ef__avg_isi",
        y="ef__avg_firing_rate",
        alpha=0.35,
        s=20,
        figsize=(7, 5),
    )

    ax.set_title("Average firing rate versus average ISI")
    ax.set_xlabel("Average ISI (ms)")
    ax.set_ylabel("Average firing rate (Hz)")

    figure = ax.get_figure()
    figure.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)


def make_feature_distribution_summary(
    X: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize the distributions of numeric model features."""
    summary = (
        X
        .describe(
            percentiles=[
                0.01,
                0.05,
                0.50,
                0.95,
                0.99,
            ]
        )
        .T
    )

    summary["skew"] = X.skew()

    return summary.loc[
        :,
        [
            "min",
            "1%",
            "5%",
            "50%",
            "95%",
            "99%",
            "max",
            "mean",
            "std",
            "skew",
        ],
    ]


def print_paired_model_comparison(
    title: str,
    reference_results: dict[str, np.ndarray],
    comparison_results: dict[str, np.ndarray],
    *,
    reference_name: str,
    comparison_name: str,
    metric: str = "macro_f1",
) -> None:
    """Print a paired fold-level model comparison."""
    comparison = make_paired_fold_comparison(
        reference_results,
        comparison_results,
        metric=metric,
        reference_name=reference_name,
        comparison_name=comparison_name,
    )

    print(f"\n{title}")
    print(comparison)


def print_donor_weight_audit(
    sample_weights: pd.Series,
    groups: pd.Series,
) -> None:
    """Print summary statistics for cell- and donor-level weights."""
    assert sample_weights.index.equals(groups.index)

    donor_total_weights = sample_weights.groupby(groups).sum()

    print("\nDonor sample-weight audit")
    print(sample_weights.describe())

    print("\nTotal weight per donor")
    print(donor_total_weights.describe())


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


def make_feature_correlation_table(
    X: pd.DataFrame,
) -> pd.DataFrame:
    """Return feature pairs ordered by absolute Spearman correlation."""
    correlation_matrix = X.corr(method="spearman")

    upper_triangle = np.triu(
        np.ones(correlation_matrix.shape, dtype=bool),
        k=1,
    )

    correlation_pairs = (
        correlation_matrix
        .where(upper_triangle)
        .stack()
        .rename("correlation")
        .reset_index()
        .rename(
            columns={
                "level_0": "feature_1",
                "level_1": "feature_2",
            }
        )
    )

    correlation_pairs["absolute_correlation"] = (
        correlation_pairs["correlation"].abs()
    )

    return correlation_pairs.sort_values(
        "absolute_correlation",
        ascending=False,
        ignore_index=True,
    )


def run_classifier_experiment(
    model_name: str,
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: list[
        tuple[np.ndarray, np.ndarray]
    ],
) -> tuple[dict[str, np.ndarray], pd.Series]:
    """Evaluate a classifier and print its diagnostics."""
    results, predictions = evaluate_classifier(
        estimator=estimator,
        X=X,
        y=y,
        cv_splits=cv_splits,
        scoring=SCORING,
    )

    print_cross_validation_results(
        model_name,
        results,
    )
    print_classification_diagnostics(
        model_name,
        y,
        predictions,
    )

    return results, predictions


def run_fold_weighted_classifier_experiment(
    model_name: str,
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    cv_splits: list[
        tuple[np.ndarray, np.ndarray]
    ],
    sample_weight_factory: Callable[
        [pd.Series],
        pd.Series,
    ],
) -> tuple[dict[str, np.ndarray], pd.Series]:
    """Evaluate and report a fold-weighted classifier."""
    results, predictions = (
        evaluate_fold_weighted_classifier(
            estimator=estimator,
            X=X,
            y=y,
            groups=groups,
            cv_splits=cv_splits,
            scoring=SCORING,
            sample_weight_factory=sample_weight_factory,
        )
    )

    print_cross_validation_results(
        model_name,
        results,
    )
    print_classification_diagnostics(
        model_name,
        y,
        predictions,
    )

    return results, predictions


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

    cross_validation_splits = list(
        cross_validator.split(
            X_development,
            y_development,
            groups_development,
        )
    )

    dummy_model = DummyClassifier(
        strategy="most_frequent",
    )

    dummy_results, _ = evaluate_classifier(
        dummy_model,
        X_development,
        y_development,
        cv_splits=cross_validation_splits,
        scoring=SCORING,
    )

    print_cross_validation_results(
        "Most-frequent dummy",
        dummy_results,
    )

    logistic_model = make_logistic_regression_pipeline()

    logistic_results, logistic_predicted_classes = (
        run_classifier_experiment(
            "Unweighted logistic regression",
            logistic_model,
            X_development,
            y_development,
            cross_validation_splits,
        )
    )

    class_weighted_logistic_model = (
        make_logistic_regression_pipeline(
            class_weight="balanced"
        )
    )

    (
        class_weighted_results,
        class_weighted_predicted_classes,
    ) = run_classifier_experiment(
        "Class-weighted logistic regression",
        class_weighted_logistic_model,
        X_development,
        y_development,
        cross_validation_splits,
    )

    print_paired_model_comparison(
        "Paired macro-F1 comparison",
        logistic_results,
        class_weighted_results,
        reference_name="unweighted",
        comparison_name="class-weighted",
    )

    sparse_class = "sparsely spiny"

    fold_diagnostics_df = make_class_fold_diagnostics(
        y_development,
        groups_development,
        {
            "unweighted": logistic_predicted_classes,
            "weighted": class_weighted_predicted_classes,
        },
        cv_splits=cross_validation_splits,
        class_label=sparse_class,
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

    donor_weights = make_donor_sample_weights(groups_development)
    print_donor_weight_audit(donor_weights, groups_development)

    donor_weighted_model = (
        make_logistic_regression_pipeline(
            class_weight=None,
        )
    )

    (
        donor_weighted_results,
        donor_weighted_predictions,
    ) = run_fold_weighted_classifier_experiment(
        model_name="Donor-weighted logistic regression",
        estimator=donor_weighted_model,
        X=X_development,
        y=y_development,
        groups=groups_development,
        cv_splits=cross_validation_splits,
        sample_weight_factory=make_donor_sample_weights,
    )

    print_paired_model_comparison(
        (
            "Paired macro-F1 comparison between "
            "donor-weighted and unweighted"
        ),
        logistic_results,
        donor_weighted_results,
        reference_name="unweighted",
        comparison_name="donor-weighted",
    )

    class_and_donor_weighted_model = (
        make_logistic_regression_pipeline(
            class_weight="balanced",
        )
    )

    (
        class_and_donor_weighted_results,
        class_and_donor_weighted_predictions,
    ) = run_fold_weighted_classifier_experiment(
        model_name=(
            "Combined class and donor weighted "
            "logistic regression"
        ),
        estimator=class_and_donor_weighted_model,
        X=X_development,
        y=y_development,
        groups=groups_development,
        cv_splits=cross_validation_splits,
        sample_weight_factory=make_donor_sample_weights,
    )

    print_paired_model_comparison(
        (
            "Paired macro-F1 comparison between "
            "a combined donor and class weighted and class-weighted"
        ),
        class_weighted_results,
        class_and_donor_weighted_results,
        reference_name="class-weighted",
        comparison_name="combined",
    )

    run_feature_audit(
        X_development,
        y_development,
    )

    X_development_without_isi = (
        X_development.drop(
            columns=["ef__avg_isi"]
        )
    )

    reduced_class_weighted_model = (
        make_logistic_regression_pipeline(
            class_weight="balanced",
        )
    )
    (
        reduced_class_weighted_results,
        reduced_class_weighted_predicted_classes,
    ) = run_classifier_experiment(
        "Reduced class-weighted logistic regression",
        reduced_class_weighted_model,
        X_development_without_isi,
        y_development,
        cross_validation_splits,
    )

    print_paired_model_comparison(
        (
            "Paired macro-F1 comparison between "
            "reduced-features-class-weighted and "
            "full-features-class-weighted"
        ),
        class_weighted_results,
        reduced_class_weighted_results,
        reference_name="class-weighted",
        comparison_name="reduced-class-weighted",
    )


if __name__ == "__main__":
    main()
