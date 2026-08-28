import json
from unittest.mock import patch

import pandas as pd
from pandas.testing import assert_frame_equal

from allen_brain_ml.data import (
    load_cell_records,
    load_or_fetch_cell_records,
    save_cell_records,
    prepare_cell_data,
    get_or_download_reconstruction,
)


def test_get_or_download_reconstruction_downloads_when_missing(
    tmp_path,
):
    cache_directory = tmp_path / "reconstructions"
    expected_path = (
        cache_directory
        / "specimen_556968207"
        / "reconstruction.swc"
    )

    with patch(
        "allen_brain_ml.data.download_well_known_file"
    ) as mock_download:
        mock_download.return_value = expected_path

        result = get_or_download_reconstruction(
            specimen_id=556968207,
            well_known_file_id=759943447,
            cache_directory=cache_directory,
        )

    assert result == expected_path

    mock_download.assert_called_once_with(
        file_id=759943447,
        destination=expected_path,
    )


def test_get_or_download_reconstruction_reuses_cached_file(
    tmp_path,
):
    cache_directory = tmp_path / "reconstructions"
    cached_path = (
        cache_directory
        / "specimen_556968207"
        / "reconstruction.swc"
    )

    cached_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    cached_path.write_bytes(
        b"# Existing reconstruction\n"
    )

    with patch(
        "allen_brain_ml.data.download_well_known_file"
    ) as mock_download:
        result = get_or_download_reconstruction(
            specimen_id=556968207,
            well_known_file_id=759943447,
            cache_directory=cache_directory,
        )

    assert result == cached_path
    assert (
        cached_path.read_bytes()
        == b"# Existing reconstruction\n"
    )
    mock_download.assert_not_called()


def test_prepare_cell_data_removes_exact_duplicates():
    records = [
        {"specimen__id": 1, "donor__species": "Mus musculus"},
        {"specimen__id": 1, "donor__species": "Mus musculus"},
        {"specimen__id": 2, "donor__species": "Homo Sapiens"},
    ]

    expected = pd.DataFrame([
        {"specimen__id": 1, "donor__species": "Mus musculus"},
        {"specimen__id": 2, "donor__species": "Homo Sapiens"},
    ])

    result = prepare_cell_data(records)

    assert_frame_equal(result, expected)


def test_load_or_fetch_cell_records_fetches_when_cache_missing(tmp_path):
    fetched_records = [{"specimen__id": 1}]
    cache_path = tmp_path / "raw" / "cells.json"

    with patch(
        "allen_brain_ml.data.get_cells",
        return_value=fetched_records,
    ) as mock_get_cells:
        result = load_or_fetch_cell_records(cache_path)

    assert result == fetched_records
    assert json.loads(
        cache_path.read_text(encoding="utf-8")
    ) == fetched_records

    mock_get_cells.assert_called_once()


def test_load_or_fetch_cell_records_uses_cache(tmp_path):
    cached_records = [{"specimen__id": 1}]
    cache_path = tmp_path / "cells.json"
    cache_path.write_text(
        json.dumps(cached_records),
        encoding="utf-8",
    )

    with patch("allen_brain_ml.data.get_cells") as mock_get_cells:
        result = load_or_fetch_cell_records(cache_path)

    assert result == cached_records
    mock_get_cells.assert_not_called()


def test_save_cell_records(tmp_path):
    records = [
        {"specimen__id": 1},
        {"specimen__id": 2},
    ]
    cache_path = tmp_path / "raw" / "cells.json"

    save_cell_records(records, cache_path)

    saved_records = json.loads(
        cache_path.read_text(encoding="utf-8")
    )

    assert saved_records == records


def test_load_cell_records_from_cache(tmp_path):
    cached_records = [
        {"specimen__id": 1, "donor__species": "Homo Sapiens"},
        {"specimen__id": 2, "donor__species": "Mus musculus"},
    ]
    cache_path = tmp_path / "cells.json"
    cache_path.write_text(
        json.dumps(cached_records),
        encoding="utf-8",
    )

    result = load_cell_records(cache_path)

    assert result == cached_records
