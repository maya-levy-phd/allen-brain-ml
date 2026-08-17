"""Functions for interacting with the Allen Brain Observatory API."""

import requests

BASE_URL = "https://api.brain-map.org/api/v2/data/query.json"
SPECIMEN_CRITERIA = "model::Specimen"
EXPERIMENT_CRITERIA = "model::Experiment"
CELL_CRITERIA = "model::Cell"

def get_cells(limit: int = 100, page_size: int = 100) -> list[dict]:
    """Retrieve up to limit cells records from the Allen API."""
    return _get_paginated(
        url=BASE_URL,
        params={"criteria": CELL_CRITERIA},
        limit=limit,
        page_size=page_size,
    )


def get_experiments(limit: int = 100, page_size: int = 100) -> list[dict]:
    """Retrieve up to limit experiments records from the Allen API."""
    return _get_paginated(
        url=BASE_URL,
        params={"criteria": EXPERIMENT_CRITERIA},
        limit=limit,
        page_size=page_size,
    )


def get_specimens(limit: int = 100, page_size: int = 100) -> list[dict]:
    """Retrieve up to limit specimen records from the Allen API."""
    return _get_paginated(
        url=BASE_URL,
        params={"criteria": SPECIMEN_CRITERIA},
        limit=limit,
        page_size=page_size,
    )


def _get_json(url: str, params: dict) -> dict:
    """Make a GET request and return the JSON response."""
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()

    data = response.json()

    if not data["success"]:
        raise RuntimeError(data["msg"])

    return data


def _get_paginated(
    url: str,
    params: dict,
    limit: int,
    page_size: int,
) -> list[dict]:
    """Retrieve up to limit records from a paginated Allen API endpoint."""
    if limit <= 0:
        raise ValueError("limit must be positive")

    if page_size <= 0:
        raise ValueError("page_size must be positive")

    records = []
    start_row = 0

    while len(records) < limit:
        remaining = limit - len(records)
        num_rows = min(page_size, remaining)

        page_params = {
            **params,
            "num_rows": num_rows,
            "start_row": start_row,
        }

        data = _get_json(url, params=page_params)

        batch = data["msg"]
        records.extend(batch)

        if len(records) >= data["total_rows"]:
            break

        if len(batch) == 0:
            break

        start_row += len(batch)

    return records[:limit]
