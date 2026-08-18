"""Construction and partitioning of modeling datasets."""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from allen_brain_ml.features import get_ephys_feature_columns


SPECIES_COLUMN = "donor__species"
TARGET_COLUMN = "tag__dendrite_type"
GROUP_COLUMN = "donor__id"
MOUSE_SPECIES = "Mus musculus"


def build_classification_cohort(
    cells: pd.DataFrame,
) -> pd.DataFrame:
    """Return complete-case mouse cells for classification."""
    ephys_columns = get_ephys_feature_columns(cells)

    if not ephys_columns:
        raise ValueError("No electrophysiology features were found.")

    required_columns = [
        *ephys_columns,
        TARGET_COLUMN,
        GROUP_COLUMN,
    ]

    return (
        cells.loc[cells[SPECIES_COLUMN].eq(MOUSE_SPECIES)]
        .dropna(subset=required_columns)
        .copy()
    )


def make_grouped_holdout_split(
    y: pd.Series,
    groups: pd.Series,
    *,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return development and test indices with disjoint groups."""
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    placeholder_features = np.zeros((len(y), 1))

    development_indices, test_indices = next(
        splitter.split(
            placeholder_features,
            y,
            groups,
        )
    )

    return development_indices, test_indices


