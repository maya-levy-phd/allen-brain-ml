from unittest.mock import Mock, call, patch
import pytest
from allen_brain_ml.allen_api import (
    BASE_URL,
    CELL_CRITERIA,
    EXPERIMENT_CRITERIA,
    SPECIMEN_CRITERIA,
    get_cells,
    get_experiments,
    get_specimens,
    download_well_known_file,
)


def test_download_well_known_file(
    tmp_path,
):
    file_content = (
        b"# Example SWC\n"
        b"1 1 0.0 0.0 0.0 1.0 -1\n"
    )

    response = Mock()
    response.content = file_content

    destination = (
        tmp_path
        / "specimen_556968207"
        / "reconstruction.swc"
    )

    with patch(
        "allen_brain_ml.allen_api.requests.get",
        return_value=response,
    ) as mock_get:
        result = download_well_known_file(
            file_id=759943447,
            destination=destination,
        )

    assert result == destination
    assert destination.read_bytes() == file_content

    mock_get.assert_called_once_with(
        (
            "https://api.brain-map.org/api/v2/"
            "well_known_file_download/759943447"
        ),
        timeout=30,
    )
    response.raise_for_status.assert_called_once_with()


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
            BASE_URL,
            params={
                "criteria": "model::Specimen",
                "num_rows": 2,
                "start_row": 0,
            },
            timeout=10,
        )

    assert result == fake_response["msg"]


def test_get_experiments():
    fake_response = {
        "success": True,
        "total_rows": 2,
        "msg": [
            {"id": 1, "name": "experiment_001"},
            {"id": 2, "name": "experiment_002"},
        ],
    }

    with patch("allen_brain_ml.allen_api.requests.get") as mock_get:
        mock_get.return_value.json.return_value = fake_response

        result = get_experiments(limit=2)

        mock_get.assert_called_once_with(
            BASE_URL,
            params={
                "criteria": "model::Experiment",
                "num_rows": 2,
                "start_row": 0,
            },
            timeout=10,
        )

    assert result == fake_response["msg"]


def test_get_cells():
    fake_response = {
        "success": True,
        "total_rows": 2,
        "msg": [
            {"id": 1, "name": "cell_001"},
            {"id": 2, "name": "cell_002"},
        ],
    }

    with patch("allen_brain_ml.allen_api.requests.get") as mock_get:
        mock_get.return_value.json.return_value = fake_response

        result = get_cells(limit=2)

        mock_get.assert_called_once_with(
            BASE_URL,
            params={
                "criteria": "model::ApiCellTypesSpecimenDetail",
                "num_rows": 2,
                "start_row": 0,
            },
            timeout=10,
        )

    assert result == fake_response["msg"]


@pytest.mark.parametrize(
    "get_records, criteria",
    [
        (get_specimens, SPECIMEN_CRITERIA),
        (get_experiments, EXPERIMENT_CRITERIA),
        (get_cells, CELL_CRITERIA),
    ],
)
def test_paginates(get_records, criteria):
    first_response = {
        "success": True,
        "total_rows": 4,
        "msg": [
            {"id": 1},
            {"id": 2},
        ],
    }

    second_response = {
        "success": True,
        "total_rows": 4,
        "msg": [
            {"id": 3},
            {"id": 4},
        ],
    }

    with patch("allen_brain_ml.allen_api.requests.get") as mock_get:
        mock_get.side_effect = [
            Mock(json=lambda: first_response),
            Mock(json=lambda: second_response),
        ]

        result = get_records(limit=4, page_size=2)

    assert result == [
        {"id": 1},
        {"id": 2},
        {"id": 3},
        {"id": 4},
    ]

    assert mock_get.call_args_list == [
        call(
            BASE_URL,
            params={
                "criteria": criteria,
                "num_rows": 2,
                "start_row": 0,
            },
            timeout=10,
        ),
        call(
            BASE_URL,
            params={
                "criteria": criteria,
                "num_rows": 2,
                "start_row": 2,
            },
            timeout=10,
        ),
    ]


@pytest.mark.parametrize(
    "get_records",
    [get_specimens, get_experiments, get_cells],
)
def test_rejects_nonpositive_limit(get_records):
    with pytest.raises(ValueError, match="limit must be positive"):
        get_records(limit=0)


@pytest.mark.parametrize(
    "get_records",
    [get_specimens, get_experiments, get_cells],
)
def test_rejects_nonpositive_page_size(get_records):
    with pytest.raises(ValueError, match="page_size must be positive"):
        get_records(page_size=0)


@pytest.mark.parametrize(
    "get_records",
    [get_specimens, get_experiments, get_cells],
)
def test_stops_when_no_more_records(get_records):
    fake_response = {
        "success": True,
        "start_row": 0,
        "num_rows": 2,
        "total_rows": 2,
        "msg": [
            {"id": 1},
            {"id": 2},
        ],
    }

    with patch("allen_brain_ml.allen_api.requests.get") as mock_get:
        mock_get.return_value.json.return_value = fake_response

        result = get_records(limit=10, page_size=2)

    assert result == fake_response["msg"]
    mock_get.assert_called_once()





