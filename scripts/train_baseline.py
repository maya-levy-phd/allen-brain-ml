from collections.abc import Callable
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (
    StratifiedGroupKFold,
    cross_validate,
)

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
    make_spline_logistic_regression_pipeline,
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
SPLINE_FEATURES = (
    "ef__avg_firing_rate",
    "ef__peak_t_ramp",
    "ef__ri",
    "ef__tau",
)
SCORING = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "macro_f1": "f1_macro",
}


def make_vif_table(
    X: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate variance inflation factors for model features."""
    records = []

    for feature in X.columns:
        feature_values = X.loc[:, feature]
        other_features = X.drop(
            columns=[feature]
        )

        auxiliary_model = LinearRegression()
        auxiliary_model.fit(
            other_features,
            feature_values,
        )

        r_squared = auxiliary_model.score(
            other_features,
            feature_values,
        )
        tolerance = 1.0 - r_squared

        vif = (
            np.inf
            if np.isclose(tolerance, 0.0)
            else 1.0 / tolerance
        )

        records.append(
            {
                "feature": feature,
                "r_squared": r_squared,
                "tolerance": tolerance,
                "vif": vif,
            }
        )

    return (
        pd.DataFrame.from_records(records)
        .sort_values(
            "vif",
            ascending=False,
            ignore_index=True,
        )
    )


def run_coefficient_audit(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: list[
        tuple[np.ndarray, np.ndarray]
    ],
) -> pd.DataFrame:
    """Audit standardized coefficients across CV folds."""
    cv_results = cross_validate(
        estimator=estimator,
        X=X,
        y=y,
        cv=cv_splits,
        scoring=SCORING,
        return_estimator=True,
    )

    fold_estimators = cv_results["estimator"]

    for fold_number, fold_estimator in enumerate(
        fold_estimators,
        start=1,
    ):
        classifier = fold_estimator.named_steps[
            "classifier"
        ]

        print(
            f"Fold {fold_number}: "
            f"classes={classifier.classes_}, "
            f"coefficient shape={classifier.coef_.shape}"
        )

    coefficient_table = make_fold_coefficient_table(
        fold_estimators,
        X.columns.to_list(),
    )

    expected_rows = (
            len(fold_estimators)
            * y.nunique()
            * X.shape[1]
    )

    assert len(coefficient_table) == expected_rows

    print("\nFold-specific coefficient table")
    print(
        coefficient_table
        .head(12)
        .round(3)
        .to_string(index=False)
    )

    print(
        f"\nCoefficient rows: "
        f"{len(coefficient_table)}"
    )

    contrast_table = (
        make_pairwise_coefficient_contrasts(
            coefficient_table
        )
    )

    contrast_summary = (
        contrast_table
        .groupby(
            [
                "class_a",
                "class_b",
                "feature",
            ]
        )["coefficient_contrast"]
        .agg(
            mean_contrast="mean",
            std_contrast="std",
            min_contrast="min",
            max_contrast="max",
        )
        .reset_index()
    )

    contrast_summary["sign_stable"] = (
            (contrast_summary["min_contrast"] > 0)
            | (contrast_summary["max_contrast"] < 0)
    )

    contrast_summary["absolute_mean_contrast"] = (
        contrast_summary["mean_contrast"].abs()
    )

    contrast_summary["odds_ratio_a_vs_b"] = np.exp(
        contrast_summary["mean_contrast"]
    )

    ranked_contrasts = contrast_summary.sort_values(
        [
            "class_a",
            "class_b",
            "absolute_mean_contrast",
        ],
        ascending=[True, True, False],
    )

    top_contrasts = (
        ranked_contrasts
        .groupby(
            ["class_a", "class_b"],
            sort=False,
        )
        .head(5)
    )

    print("\nStrongest mean coefficient contrasts")
    print(
        top_contrasts[
            [
                "class_a",
                "class_b",
                "feature",
                "mean_contrast",
                "std_contrast",
                "sign_stable",
                "odds_ratio_a_vs_b",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    stability_summary = (
        contrast_summary
        .groupby(
            ["class_a", "class_b"]
        )["sign_stable"]
        .agg(
            stable_features="sum",
            total_features="size",
        )
        .reset_index()
    )

    stability_summary["stable_fraction"] = (
            stability_summary["stable_features"]
            / stability_summary["total_features"]
    )

    print("\nCoefficient sign stability by class pair")
    print(
        stability_summary
        .round(3)
        .to_string(index=False)
    )

    unstable_contrasts = (
        contrast_summary
        .loc[
            ~contrast_summary["sign_stable"]
        ]
        .sort_values(
            "absolute_mean_contrast",
            ascending=False,
        )
    )

    print("\nCoefficient contrasts with unstable signs")
    print(
        unstable_contrasts[
            [
                "class_a",
                "class_b",
                "feature",
                "mean_contrast",
                "std_contrast",
                "min_contrast",
                "max_contrast",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    sign_consistency = (
        contrast_table
        .groupby(
            [
                "class_a",
                "class_b",
                "feature",
            ]
        )["coefficient_contrast"]
        .agg(
            median_contrast="median",
            positive_folds=lambda values: (
                    values > 0
            ).sum(),
            negative_folds=lambda values: (
                    values < 0
            ).sum(),
        )
        .reset_index()
    )

    sign_consistency["dominant_sign_folds"] = (
        sign_consistency[
            [
                "positive_folds",
                "negative_folds",
            ]
        ]
        .max(axis=1)
    )

    sign_consistency["dominant_sign_fraction"] = (
            sign_consistency["dominant_sign_folds"]
            / len(fold_estimators)
    )

    contrast_summary = contrast_summary.merge(
        sign_consistency,
        on=[
            "class_a",
            "class_b",
            "feature",
        ],
        validate="one_to_one",
    )

    unstable_consistency = (
        contrast_summary
        .loc[
            ~contrast_summary["sign_stable"],
            [
                "class_a",
                "class_b",
                "feature",
                "mean_contrast",
                "median_contrast",
                "positive_folds",
                "negative_folds",
                "dominant_sign_fraction",
            ],
        ]
        .sort_values(
            "dominant_sign_fraction",
            ascending=False,
        )
    )

    print("\nSign consistency of unstable contrasts")
    print(
        unstable_consistency
        .round(3)
        .to_string(index=False)
    )

    return coefficient_table


def make_pairwise_coefficient_contrasts(
    coefficient_table: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate class-pair coefficient differences."""
    coefficient_matrix = coefficient_table.pivot(
        index=["fold", "feature"],
        columns="dendrite_class",
        values="coefficient",
    )

    contrast_tables = []

    for class_a, class_b in combinations(
        coefficient_matrix.columns,
        2,
    ):
        class_contrast = (
            coefficient_matrix[class_a]
            - coefficient_matrix[class_b]
        )

        class_contrast = (
            class_contrast
            .rename("coefficient_contrast")
            .reset_index()
        )

        class_contrast["class_a"] = class_a
        class_contrast["class_b"] = class_b

        contrast_tables.append(class_contrast)

    contrasts = pd.concat(
        contrast_tables,
        ignore_index=True,
    )

    return contrasts.loc[
        :,
        [
            "fold",
            "class_a",
            "class_b",
            "feature",
            "coefficient_contrast",
        ],
    ]


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


def make_fold_coefficient_table(
    fold_estimators: list,
    feature_names: list[str],
) -> pd.DataFrame:
    """Return one row per fold, class, and feature coefficient."""
    records = []

    for fold_number, fold_estimator in enumerate(
        fold_estimators,
        start=1,
    ):
        classifier = fold_estimator.named_steps[
            "classifier"
        ]

        for class_label, class_coefficients in zip(
            classifier.classes_,
            classifier.coef_,
            strict=True,
        ):
            for feature_name, coefficient in zip(
                feature_names,
                class_coefficients,
                strict=True,
            ):
                records.append(
                    {
                        "fold": fold_number,
                        "dendrite_class": class_label,
                        "feature": feature_name,
                        "coefficient": coefficient,
                    }
                )

    return pd.DataFrame.from_records(records)


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

    print()
    coefficient_table = run_coefficient_audit(
        estimator=reduced_class_weighted_model,
        X=X_development_without_isi,
        y=y_development,
        cv_splits=cross_validation_splits,
    )

    vif_table = make_vif_table(
        X_development_without_isi
    )

    print("\nVariance inflation factors")
    print(
        vif_table
        .round(3)
        .to_string(index=False)
    )

    spline_features = list(SPLINE_FEATURES)

    linear_features = [
        feature
        for feature in X_development_without_isi.columns
        if feature not in SPLINE_FEATURES
    ]

    assert set(spline_features).isdisjoint(
        linear_features
    )
    assert (
            len(spline_features)
            + len(linear_features)
            == X_development_without_isi.shape[1]
    )

    spline_logistic_model = (
        make_spline_logistic_regression_pipeline(
            spline_features=spline_features,
            linear_features=linear_features,
            class_weight="balanced",
            n_knots=5,
            degree=3,
        )
    )

    (
        spline_logistic_results,
        spline_logistic_predictions,
    ) = run_classifier_experiment(
        model_name="Spline logistic regression",
        estimator=spline_logistic_model,
        X=X_development_without_isi,
        y=y_development,
        cv_splits=cross_validation_splits,
    )

    print_paired_model_comparison(
        (
            "Paired macro-F1 comparison between "
            "spline and linear logistic regression"
        ),
        reduced_class_weighted_results,
        spline_logistic_results,
        reference_name="linear",
        comparison_name="spline",
    )


if __name__ == "__main__":
    main()
