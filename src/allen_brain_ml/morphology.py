from pathlib import Path

import numpy as np
import pandas as pd


SWC_COLUMNS = (
    "node_id",
    "compartment_type",
    "x",
    "y",
    "z",
    "radius",
    "parent_id",
)


def read_swc(path: Path) -> pd.DataFrame:
    """Read an SWC reconstruction into a node table."""
    records = []

    with path.open(encoding="utf-8") as swc_file:
        for line_number, line in enumerate(
            swc_file,
            start=1,
        ):
            stripped_line = line.strip()

            if (
                not stripped_line
                or stripped_line.startswith("#")
            ):
                continue

            fields = stripped_line.split()

            if len(fields) != len(SWC_COLUMNS):
                raise ValueError(
                    f"{path}: line {line_number} contains "
                    f"{len(fields)} fields; expected "
                    f"{len(SWC_COLUMNS)}"
                )

            try:
                record = {
                    "node_id": int(fields[0]),
                    "compartment_type": int(fields[1]),
                    "x": float(fields[2]),
                    "y": float(fields[3]),
                    "z": float(fields[4]),
                    "radius": float(fields[5]),
                    "parent_id": int(fields[6]),
                }
            except ValueError as error:
                raise ValueError(
                    f"{path}: line {line_number} "
                    "contains invalid numeric values"
                ) from error

            records.append(record)

    return pd.DataFrame.from_records(
        records,
        columns=SWC_COLUMNS,
    )


def validate_swc(nodes: pd.DataFrame) -> None:
    """Validate the structure of an SWC node table."""
    nonpositive_node_ids = (
        nodes.loc[
            nodes["node_id"].le(0),
            "node_id",
        ]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    if nonpositive_node_ids:
        raise ValueError(
            "SWC node IDs must be positive: "
            f"{nonpositive_node_ids}"
        )

    duplicate_node_ids = (
        nodes.loc[
            nodes["node_id"].duplicated(keep=False),
            "node_id",
        ]
        .drop_duplicates()
        .tolist()
    )

    if duplicate_node_ids:
        raise ValueError(
            "Duplicate SWC node IDs: "
            f"{duplicate_node_ids}"
        )

    missing_parent_ids = (
        nodes.loc[
            (
                nodes["parent_id"].ne(-1)
                & ~nodes["parent_id"].isin(
                    nodes["node_id"]
                )
            ),
            "parent_id",
        ]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    if missing_parent_ids:
        raise ValueError(
            "Missing SWC parent IDs: "
            f"{missing_parent_ids}"
        )

    self_parenting_node_ids = (
        nodes.loc[
            nodes["node_id"].eq(
                nodes["parent_id"]
            ),
            "node_id",
        ]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    if self_parenting_node_ids:
        raise ValueError(
            "Self-parenting SWC node IDs: "
            f"{self_parenting_node_ids}"
        )

    root_count = int(
        nodes["parent_id"].eq(-1).sum()
    )

    if root_count != 1:
        raise ValueError(
            "SWC reconstruction must contain "
            "exactly one root; "
            f"found {root_count}"
        )

    parent_by_node = (
        nodes
        .set_index("node_id")["parent_id"]
        .to_dict()
    )

    cycle_node_ids = _find_parent_cycle(
        parent_by_node
    )

    if cycle_node_ids is not None:
        raise ValueError(
            "Cycle detected in SWC parent "
            f"relationships: {cycle_node_ids}"
        )

    valid_radius_mask = (
            np.isfinite(nodes["radius"])
            & nodes["radius"].gt(0)
    )

    if not valid_radius_mask.all():
        raise ValueError(
            "SWC radii must be finite and positive"
        )


def _find_parent_cycle(
    parent_by_node: dict[int, int],
) -> list[int] | None:
    """Return node IDs in a parent cycle, if one exists."""
    fully_checked = set()

    for starting_node_id in parent_by_node:
        path = []
        path_positions = {}
        current_node_id = starting_node_id

        while (
            current_node_id != -1
            and current_node_id not in fully_checked
        ):
            if current_node_id in path_positions:
                cycle_start = path_positions[
                    current_node_id
                ]
                return sorted(path[cycle_start:])

            path_positions[current_node_id] = len(path)
            path.append(current_node_id)

            current_node_id = parent_by_node[
                current_node_id
            ]

        fully_checked.update(path)

    return None


