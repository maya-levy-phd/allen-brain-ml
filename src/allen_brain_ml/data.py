"""Functions for loading and cleaning Allen Brain data."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from allen_brain_ml.allen_api import get_cells



def prepare_cell_data(
    records: list[dict],
) -> pd.DataFrame:
    """Create a deduplicated cell DataFrame from raw API records."""
    return (
        pd.DataFrame(records)
        .drop_duplicates()
        .reset_index(drop=True)
    )


def load_or_fetch_cell_records(
    cache_path: Path,
    *,
    limit: int = 10_000,
    page_size: int = 100,
) -> list[dict]:
    """Load cached cell records or fetch and cache them."""
    if cache_path.is_file():
        return load_cell_records(cache_path)

    records = get_cells(
        limit=limit,
        page_size=page_size,
    )
    save_cell_records(records, cache_path)

    return records


def save_cell_records(
    records: list[dict],
    cache_path: Path,
) -> None:
    """Save raw cell records to a JSON cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    with cache_path.open("w", encoding="utf-8") as cache_file:
        json.dump(records, cache_file)


def load_cell_records(cache_path: Path) -> list[dict]:
    """Load raw cell records from a JSON cache."""
    with cache_path.open(encoding="utf-8") as cache_file:
        return json.load(cache_file)


def clean_session_name(name: str) -> str:
    """Return a session name with surrounding whitespace removed."""
    return name.strip()


def normalize_session_names(names: list[str]) -> list[str]:
    """Clean whitespace from a collection of session names."""
    return [name.strip() for name in names]


def validate_duration(
    duration_seconds: float | np.ndarray,
) -> None:
    """Validate that all durations are strictly positive."""
    if np.any(duration_seconds <= 0):
        raise ValueError("Duration must be greater than zero.")
