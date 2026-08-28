from pathlib import Path

import pandas as pd

from allen_brain_ml.data import (
    load_or_fetch_cell_records,
    prepare_cell_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = PROJECT_ROOT / "data" / "raw" / "cells.json"

MOUSE_SPECIES = "Mus musculus"


def main() -> None:
    """Explore mouse cells with neuronal reconstructions."""
    records = load_or_fetch_cell_records(CACHE_PATH)
    cells = prepare_cell_data(records)

    mouse_cells = cells.loc[
        cells["donor__species"].eq(MOUSE_SPECIES)
    ].copy()

    reconstructed_mouse_cells = mouse_cells.loc[
        mouse_cells["nrwkf__id"].notna()
    ].copy()

    reconstruction_columns = sorted(
        column
        for column in reconstructed_mouse_cells.columns
        if column.startswith(
            (
                "nr__",
                "nrwkf__",
            )
        )
    )

    print("\nReconstructed mouse cells")
    print(f"Cells: {len(reconstructed_mouse_cells)}")
    print(
        "Donors: "
        f"{reconstructed_mouse_cells['donor__id'].nunique()}"
    )

    print("\nReconstruction-related columns")
    for column in reconstruction_columns:
        print(column)

    print(
        "\nUnique specimen IDs: "
        f"{reconstructed_mouse_cells['specimen__id'].nunique()}"
    )
    print(
        "Unique reconstruction file IDs: "
        f"{reconstructed_mouse_cells['nrwkf__id'].nunique()}"
    )

    print("\nDendrite-type counts")
    print(
        reconstructed_mouse_cells[
            "tag__dendrite_type"
        ].value_counts(
            dropna=False
        )
    )

    print("\nUnique donors by dendrite type")
    print(
        reconstructed_mouse_cells
        .groupby(
            "tag__dendrite_type",
            dropna=False,
        )["donor__id"]
        .nunique()
    )

    print("\nReconstruction-type counts")
    print(
        reconstructed_mouse_cells[
            "nr__reconstruction_type"
        ].value_counts(
            dropna=False
        )
    )

    morphology_feature_columns = [
        column
        for column in reconstruction_columns
        if (
                column.startswith("nr__")
                and column != "nr__reconstruction_type"
        )
    ]

    print("\nPrecomputed morphology-feature missingness")
    print(
        reconstructed_mouse_cells[
            morphology_feature_columns
        ]
        .isna()
        .sum()
    )

    reconstruction_types = (
        reconstructed_mouse_cells[
            "nr__reconstruction_type"
        ]
        .fillna("missing")
        .rename("reconstruction_type")
    )

    print("\nDendrite type by reconstruction type")
    print(
        pd.crosstab(
            reconstructed_mouse_cells[
                "tag__dendrite_type"
            ],
            reconstruction_types,
            margins=True,
        )
    )


if __name__ == "__main__":
    main()