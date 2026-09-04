import pytest

import pandas as pd

from allen_brain_ml.morphology import (
    read_swc,
    validate_swc,
)


def test_read_swc(tmp_path):
    swc_path = tmp_path / "reconstruction.swc"
    swc_path.write_text(
        (
            "# Example reconstruction\n"
            "# id,type,x,y,z,r,pid\n"   
            "\n"
            "1 1 0.0 0.0 0.0 5.0 -1\n"
            "2 3 1.0 2.0 3.0 0.5 1\n"
            "3 4 2.0 3.0 4.0 0.4 2\n"
        ),
        encoding="utf-8",
    )

    result = read_swc(swc_path)

    expected = pd.DataFrame(
        {
            "node_id": [1, 2, 3],
            "compartment_type": [1, 3, 4],
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 2.0, 3.0],
            "z": [0.0, 3.0, 4.0],
            "radius": [5.0, 0.5, 0.4],
            "parent_id": [-1, 1, 2],
        }
    )

    pd.testing.assert_frame_equal(
        result,
        expected,
    )


def test_validate_swc_rejects_duplicate_node_ids():
    nodes = pd.DataFrame(
        {
            "node_id": [1, 2, 2],
            "compartment_type": [1, 3, 3],
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 0.0, 0.0],
            "z": [0.0, 0.0, 0.0],
            "radius": [5.0, 0.5, 0.4],
            "parent_id": [-1, 1, 1],
        }
    )

    with pytest.raises(
        ValueError,
        match=r"Duplicate SWC node IDs: \[2\]",
    ):
        validate_swc(nodes)


def test_validate_swc_rejects_missing_parent_ids():
    nodes = pd.DataFrame(
        {
            "node_id": [1, 2, 3],
            "compartment_type": [1, 3, 3],
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 0.0, 0.0],
            "z": [0.0, 0.0, 0.0],
            "radius": [5.0, 0.5, 0.4],
            "parent_id": [-1, 1, 99],
        }
    )

    with pytest.raises(
        ValueError,
        match=r"Missing SWC parent IDs: \[99\]",
    ):
        validate_swc(nodes)


def test_validate_swc_rejects_self_parenting_nodes():
    nodes = pd.DataFrame(
        {
            "node_id": [1, 2, 3],
            "compartment_type": [1, 3, 3],
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 0.0, 0.0],
            "z": [0.0, 0.0, 0.0],
            "radius": [5.0, 0.5, 0.4],
            "parent_id": [-1, 2, 1],
        }
    )

    with pytest.raises(
        ValueError,
        match=(
            r"Self-parenting SWC node IDs: "
            r"\[2\]"
        ),
    ):
        validate_swc(nodes)


def test_validate_swc_rejects_parent_cycle():
    nodes = pd.DataFrame(
        {
            "node_id": [1, 2, 3, 4],
            "compartment_type": [1, 3, 3, 3],
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [0.0, 0.0, 0.0, 0.0],
            "z": [0.0, 0.0, 0.0, 0.0],
            "radius": [5.0, 0.5, 0.4, 0.3],
            "parent_id": [-1, 3, 2, 1],
        }
    )

    with pytest.raises(
        ValueError,
        match=(
            r"Cycle detected in SWC parent "
            r"relationships: \[2, 3\]"
        ),
    ):
        validate_swc(nodes)


@pytest.mark.parametrize(
    "invalid_node_id",
    [0, -2],
)
def test_validate_swc_rejects_nonpositive_node_ids(
    invalid_node_id,
):
    nodes = pd.DataFrame(
        {
            "node_id": [invalid_node_id, 2],
            "compartment_type": [1, 3],
            "x": [0.0, 1.0],
            "y": [0.0, 0.0],
            "z": [0.0, 0.0],
            "radius": [5.0, 0.5],
            "parent_id": [-1, invalid_node_id],
        }
    )

    with pytest.raises(
        ValueError,
        match="SWC node IDs must be positive",
    ):
        validate_swc(nodes)


@pytest.mark.parametrize(
    "parent_ids",
    [
        pytest.param(
            [],
            id="no-root",
        ),
        pytest.param(
            [-1, -1],
            id="multiple-roots",
        ),
    ],
)
def test_validate_swc_requires_exactly_one_root(
    parent_ids,
):
    node_count = len(parent_ids)

    nodes = pd.DataFrame(
        {
            "node_id": list(
                range(1, node_count + 1)
            ),
            "compartment_type": (
                [3] * node_count
            ),
            "x": [0.0] * node_count,
            "y": [0.0] * node_count,
            "z": [0.0] * node_count,
            "radius": [0.5] * node_count,
            "parent_id": parent_ids,
        }
    )

    with pytest.raises(
        ValueError,
        match=(
            "SWC reconstruction must contain "
            "exactly one root"
        ),
    ):
        validate_swc(nodes)


@pytest.mark.parametrize(
    "invalid_radius",
    [
        pytest.param(0.0, id="zero"),
        pytest.param(-0.5, id="negative"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="positive-infinity"),
        pytest.param(float("-inf"), id="negative-infinity"),
    ],
)
def test_validate_swc_rejects_invalid_radius(
    invalid_radius,
):
    nodes = pd.DataFrame(
        {
            "node_id": [1, 2],
            "compartment_type": [1, 3],
            "x": [0.0, 1.0],
            "y": [0.0, 0.0],
            "z": [0.0, 0.0],
            "radius": [5.0, invalid_radius],
            "parent_id": [-1, 1],
        }
    )

    with pytest.raises(
        ValueError,
        match=(
            "SWC radii must be finite "
            "and positive"
        ),
    ):
        validate_swc(nodes)




