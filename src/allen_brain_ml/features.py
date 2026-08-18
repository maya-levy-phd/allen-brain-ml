"""Feature selection and engineering for Allen Brain models."""

import pandas as pd


def get_ephys_feature_columns(
    cells: pd.DataFrame,
) -> list[str]:
    """Return electrophysiology columns in their original order."""
    return [
        column
        for column in cells.columns
        if column.startswith("ef__")
    ]



