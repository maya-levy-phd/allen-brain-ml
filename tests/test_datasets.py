import pandas as pd
import numpy as np

from allen_brain_ml.datasets import (
    build_classification_cohort,
    make_grouped_holdout_split,
    make_donor_sample_weights,
)


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


def test_make_grouped_holdout_split_is_group_disjoint():
    groups = pd.Series(np.repeat(np.arange(30), 2))
    y = pd.Series(
        np.repeat(
            ["aspiny", "sparsely spiny", "spiny"],
            20,
        )
    )

    development_indices, test_indices = (
        make_grouped_holdout_split(y, groups)
    )

    development_groups = set(groups.iloc[development_indices])
    test_groups = set(groups.iloc[test_indices])

    assert development_groups.isdisjoint(test_groups)

    all_indices = np.concatenate(
        [development_indices, test_indices]
    )
    assert sorted(all_indices.tolist()) == list(range(len(y)))

    assert set(y.iloc[test_indices]) == set(y)


def test_make_donor_sample_weights_equalizes_donors():
    groups = pd.Series(
        [101, 202, 202, 303, 303, 303],
        index=[10, 20, 30, 40, 50, 60],
        name="donor__id",
    )

    result = make_donor_sample_weights(groups)

    expected = pd.Series(
        [2.0, 1.0, 1.0, 2 / 3, 2 / 3, 2 / 3],
        index=groups.index,
        name="sample_weight",
    )

    pd.testing.assert_series_equal(result, expected)

    donor_total_weights = result.groupby(groups).sum()

    np.testing.assert_allclose(
        donor_total_weights.to_numpy(),
        [2.0, 2.0, 2.0],
    )
    np.testing.assert_allclose(
        result.mean(),
        1.0,
    )
