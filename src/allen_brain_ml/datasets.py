"""Construction of modeling datasets."""

import pandas as pd

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

