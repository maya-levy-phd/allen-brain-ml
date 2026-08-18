"""Explore Allen cell data."""

from pathlib import Path

import pandas as pd

from allen_brain_ml.data import load_or_fetch_cell_records


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CELL_CACHE_PATH = PROJECT_ROOT / "data" / "raw" / "cells.json"


def main() -> None:
    """Load cell data and report basic dataset dimensions."""
    cell_records = load_or_fetch_cell_records(CELL_CACHE_PATH)
    raw_cells = pd.DataFrame(cell_records)

    unique_cells = (
        raw_cells
        .drop_duplicates()
        .reset_index(drop=True)
    )

    print(f"Raw rows: {len(raw_cells)}")
    print(f"Unique rows: {len(unique_cells)}")
    print(
        "Unique specimen IDs: "
        f"{unique_cells['specimen__id'].nunique()}"
    )


if __name__ == "__main__":
    main()