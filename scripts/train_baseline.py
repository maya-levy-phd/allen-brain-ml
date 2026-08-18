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
)
from allen_brain_ml.features import get_ephys_feature_columns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = PROJECT_ROOT / "data" / "raw" / "cells.json"


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


if __name__ == "__main__":
    main()
