import pandas as pd

from allen_brain_ml.datasets import build_classification_cohort


def test_build_classification_cohort_selects_complete_mouse_cells():
    cells = pd.DataFrame(
        {
            "specimen__id": [1, 2, 3, 4, 5],
            "donor__species": [
                "Mus musculus",
                "Homo Sapiens",
                "Mus musculus",
                "Mus musculus",
                "Mus musculus",
            ],
            "donor__id": [101, 102, 103, 104, pd.NA],
            "tag__dendrite_type": [
                "aspiny",
                "spiny",
                "spiny",
                pd.NA,
                "aspiny",
            ],
            "ef__ri": [100.0, 110.0, pd.NA, 130.0, 140.0],
            "ef__tau": [10.0, 11.0, 12.0, 13.0, 14.0],
        }
    )

    result = build_classification_cohort(cells)

    assert result["specimen__id"].tolist() == [1]
