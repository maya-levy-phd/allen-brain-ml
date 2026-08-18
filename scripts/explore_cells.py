"""Explore Allen cell data."""

from pathlib import Path

import pandas as pd

from allen_brain_ml.data import (
    load_or_fetch_cell_records,
    prepare_cell_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CELL_CACHE_PATH = PROJECT_ROOT / "data" / "raw" / "cells.json"
TARGET_COLUMN = "tag__dendrite_type"
DISTANCE_COLUMN = "nr__max_euclidean_distance"


def print_section(title: str) -> None:
    """Print a readable analysis-section heading."""
    print(f"\n{title}")
    print("=" * len(title))


def print_dataset_overview(
    cell_records: list[dict],
    unique_cells: pd.DataFrame,
) -> None:
    """Print basic dimensions and class counts."""
    print_section("Dataset overview")

    print(f"Raw rows: {len(cell_records)}")
    print(f"Unique rows: {len(unique_cells)}")
    print(
        "Unique specimen IDs: "
        f"{unique_cells['specimen__id'].nunique()}"
    )

    dendrite_counts = pd.crosstab(
        unique_cells["donor__species"],
        unique_cells["tag__dendrite_type"],
        margins=True,
    )

    print("\nDendrite-type counts by species:")
    print(dendrite_counts)


def select_mouse_cells(
    unique_cells: pd.DataFrame,
) -> pd.DataFrame:
    """Return an independent DataFrame containing mouse cells."""
    return unique_cells.loc[
        unique_cells["donor__species"] == "Mus musculus"
    ].copy()


def get_ephys_columns(
    cells: pd.DataFrame,
) -> list[str]:
    """Return the names of electrophysiology columns."""
    return [
        column
        for column in cells.columns
        if column.startswith("ef__")
    ]


def print_ephys_audit(
    mouse_cells: pd.DataFrame,
    ephys_columns: list[str],
) -> None:
    """Print electrophysiology missingness summaries."""
    print_section("Electrophysiology audit")

    missingness = (
        mouse_cells[ephys_columns]
        .isna()
        .agg(["sum", "mean"])
        .T
        .rename(
            columns={
                "sum": "missing_count",
                "mean": "missing_fraction",
            }
        )
        .sort_values(
            "missing_fraction",
            ascending=False,
        )
    )

    missingness["missing_count"] = (
        missingness["missing_count"].astype(int)
    )

    print("\nOverall missingness:")
    print(missingness.to_string())

    missing_by_class = (
        mouse_cells[ephys_columns]
        .isna()
        .groupby(mouse_cells[TARGET_COLUMN])
        .mean()
        .T
    )

    print("\nMissing fractions by dendrite class:")
    print(missing_by_class.to_string())

    complete_ephys = (
        mouse_cells[ephys_columns]
        .notna()
        .all(axis=1)
    )

    complete_case_counts = pd.crosstab(
        mouse_cells[TARGET_COLUMN],
        complete_ephys,
        margins=True,
    )

    complete_case_proportions = pd.crosstab(
        mouse_cells[TARGET_COLUMN],
        complete_ephys,
        normalize="index",
    )

    print("\nComplete-case counts:")
    print(complete_case_counts)

    print("\nComplete-case proportions:")
    print(complete_case_proportions)


def select_classification_cells(
    mouse_cells: pd.DataFrame,
    ephys_columns: list[str],
) -> pd.DataFrame:
    """Return mouse cells with complete classification data."""
    return (
        mouse_cells
        .dropna(
            subset=ephys_columns + [TARGET_COLUMN]
        )
        .copy()
    )


def print_classification_audit(
    classification_cells: pd.DataFrame,
) -> None:
    """Print class and donor summaries for classification."""
    print_section("Classification cohort")

    print(f"Cells: {len(classification_cells)}")
    print(
        "Donors: "
        f"{classification_cells['donor__id'].nunique()}"
    )

    print("\nClass counts:")
    print(
        classification_cells[TARGET_COLUMN]
        .value_counts()
    )

    cells_per_donor = (
        classification_cells
        .groupby("donor__id")
        .size()
    )

    print("\nCells per donor:")
    print(cells_per_donor.describe())

    print("\nDistribution of cells per donor:")
    print(
        cells_per_donor
        .value_counts()
        .sort_index()
    )

    classes_per_donor = (
        classification_cells
        .groupby("donor__id")
        [TARGET_COLUMN]
        .nunique()
    )

    print("\nDistribution of classes per donor:")
    print(
        classes_per_donor
        .value_counts()
        .sort_index()
    )

    donor_summary = (
        classification_cells
        .groupby("donor__id")
        .agg(
            cell_count=("specimen__id", "size"),
            class_count=(TARGET_COLUMN, "nunique"),
        )
    )

    print("\nCell count versus class count per donor:")
    print(
        pd.crosstab(
            donor_summary["cell_count"],
            donor_summary["class_count"],
            margins=True,
        )
    )


def print_apical_status_audit(
    mouse_cells: pd.DataFrame,
) -> None:
    """Print the relationship between apical status and target class."""
    print_section("Apical-status audit")

    apical_status = (
        mouse_cells["tag__apical"]
        .replace("NA", pd.NA)
        .fillna("missing")
    )

    counts = pd.crosstab(
        mouse_cells[TARGET_COLUMN],
        apical_status,
        margins=True,
    )

    proportions = pd.crosstab(
        mouse_cells[TARGET_COLUMN],
        apical_status,
        normalize="index",
    )

    print("\nCounts:")
    print(counts)

    print("\nWithin-class proportions:")
    print(proportions)


def print_morphology_audit(
    mouse_cells: pd.DataFrame,
) -> None:
    """Print reconstruction availability and distance summaries."""
    print_section("Morphology audit")

    distance_available = (
        mouse_cells[DISTANCE_COLUMN].notna()
    )
    reconstruction_id_available = (
        mouse_cells["nrwkf__id"].notna()
    )

    availability = pd.crosstab(
        distance_available,
        reconstruction_id_available,
    )

    print("\nDistance versus reconstruction-ID availability:")
    print(availability)

    inconsistent_availability = (
        distance_available
        != reconstruction_id_available
    )

    print(
        "\nInconsistent availability rows: "
        f"{inconsistent_availability.sum()}"
    )

    reconstructed_mouse = mouse_cells.loc[
        distance_available
    ].copy()

    print(f"\nReconstructed cells: {len(reconstructed_mouse)}")
    print(
        "Reconstructed-cell donors: "
        f"{reconstructed_mouse['donor__id'].nunique()}"
    )

    print("\nMaximum Euclidean distance distribution:")
    print(
        reconstructed_mouse[DISTANCE_COLUMN]
        .describe()
    )

    distance_by_class = (
        reconstructed_mouse
        .groupby(TARGET_COLUMN)
        .agg(
            cells=("specimen__id", "nunique"),
            donors=("donor__id", "nunique"),
            mean_distance=(DISTANCE_COLUMN, "mean"),
            median_distance=(DISTANCE_COLUMN, "median"),
            std_distance=(DISTANCE_COLUMN, "std"),
        )
    )

    print("\nDistance by dendrite class:")
    print(distance_by_class)


def main() -> None:
    """Run exploratory analyses of Allen cell data."""
    cell_records = load_or_fetch_cell_records(CELL_CACHE_PATH)
    unique_cells = prepare_cell_data(cell_records)
    mouse_cells = select_mouse_cells(unique_cells)

    print_dataset_overview(cell_records, unique_cells)

    ephys_columns = get_ephys_columns(mouse_cells)

    print_dataset_overview(cell_records, unique_cells)
    print_ephys_audit(mouse_cells, ephys_columns)

    classification_cells = select_classification_cells(
        mouse_cells,
        ephys_columns,
    )

    print_dataset_overview(cell_records, unique_cells)
    print_ephys_audit(mouse_cells, ephys_columns)
    print_classification_audit(classification_cells)

    print_apical_status_audit(mouse_cells)

    print_morphology_audit(mouse_cells)


if __name__ == "__main__":
    main()