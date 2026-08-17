import pytest
import requests
from unittest.mock import Mock, call, patch
from allen_brain_ml.allen_api import get_specimens


def test_get_specimens():
    fake_response = {
        "success": True,
        "start_row": 0,
        "num_rows": 2,
        "total_rows": 265937,
        "msg": [
            {"id": 41, "name": "343-0935"},
            {"id": 42, "name": "343-0935-56.1"},
        ],
    }

    with patch("allen_brain_ml.allen_api.requests.get") as mock_get:
        mock_get.return_value.json.return_value = fake_response

        result = get_specimens(limit=2)

        mock_get.assert_called_once_with(
            "https://api.brain-map.org/api/v2/data/query.json",
            params={
                "criteria": "model::Specimen",
                "num_rows": 2,
                "start_row": 0,
            },
            timeout=10,
        )

    assert result == fake_response["msg"]


def test_get_specimens_paginates():
    first_response = {
        "success": True,
        "total_rows": 4,
        "msg": [
            {"id": 1, "name": "specimen_001"},
            {"id": 2, "name": "specimen_002"},
        ],
    }

    second_response = {
        "success": True,
        "total_rows": 4,
        "msg": [
            {"id": 3, "name": "specimen_003"},
            {"id": 4, "name": "specimen_004"},
        ],
    }

    with patch("allen_brain_ml.allen_api.requests.get") as mock_get:
        mock_get.side_effect = [
            Mock(json=lambda: first_response),
            Mock(json=lambda: second_response),
        ]

        result = get_specimens(limit=4, page_size=2)

    assert result == [
        {"id": 1, "name": "specimen_001"},
        {"id": 2, "name": "specimen_002"},
        {"id": 3, "name": "specimen_003"},
        {"id": 4, "name": "specimen_004"},
    ]

    assert mock_get.call_args_list == [
        call(
            "https://api.brain-map.org/api/v2/data/query.json",
            params={
                "criteria": "model::Specimen",
                "num_rows": 2,
                "start_row": 0,
            },
            timeout=10,
        ),
        call(
            "https://api.brain-map.org/api/v2/data/query.json",
            params={
                "criteria": "model::Specimen",
                "num_rows": 2,
                "start_row": 2,
            },
            timeout=10,
        ),
    ]


def test_get_specimens_rejects_nonpositive_limit():
    with pytest.raises(ValueError, match="limit must be positive"):
        get_specimens(limit=0)


def test_get_specimens_rejects_nonpositive_page_size():
    with pytest.raises(ValueError, match="page_size must be positive"):
        get_specimens(page_size=0)


def test_get_specimens_stops_when_no_more_records():
    fake_response = {
        "success": True,
        "start_row": 0,
        "num_rows": 2,
        "total_rows": 2,
        "msg": [
            {"id": 1, "name": "specimen_001"},
            {"id": 2, "name": "specimen_002"},
        ],
    }

    with patch("allen_brain_ml.allen_api.requests.get") as mock_get:
        mock_get.return_value.json.return_value = fake_response

        result = get_specimens(limit=10, page_size=2)

    assert result == fake_response["msg"]
    mock_get.assert_called_once()





