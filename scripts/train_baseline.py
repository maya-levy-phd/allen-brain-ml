from pathlib import Path

import pandas as pd

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
from allen_brain_ml.features import get_ephys_feature_columns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = PROJECT_ROOT / "data" / "raw" / "cells.json"


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


if __name__ == "__main__":
    main()
