"""Tests for feature selection and engineering."""

import pandas as pd

from allen_brain_ml.features import get_ephys_feature_columns


def test_get_ephys_feature_columns():
    cells = pd.DataFrame(
        columns=[
            "ef__zeta",
            "specimen__id",
            "ef__alpha",
            "metadata_ef__source",
            "ef__middle",
        ]
    )

    result = get_ephys_feature_columns(cells)

    assert result == [
        "ef__zeta",
        "ef__alpha",
        "ef__middle",
    ]

