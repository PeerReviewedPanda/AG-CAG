"""Recalculate pointwise cross-unit proxy metrics without protected measurements."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

from verify_metrics import score


COLUMNS = (
    "direction", "axis", "thermal_segment", "sample_index",
    "target_proxy", "agcag",
)
DIRECTIONS = ("A_to_B", "B_to_A")
AXES = ("X", "Y", "Z")
SEGMENTS = ("dynamic", "steady", "intermediate")


def read_transfer(path: Path) -> dict[tuple[str, str], list[dict[str, object]]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or set(reader.fieldnames) != set(COLUMNS):
            raise ValueError(f"Expected exactly these columns: {', '.join(COLUMNS)}")
        for row_number, row in enumerate(reader, start=2):
            direction, axis, segment = (
                row["direction"], row["axis"], row["thermal_segment"]
            )
            if direction not in DIRECTIONS or axis not in AXES or segment not in SEGMENTS:
                raise ValueError(f"Invalid group label on CSV row {row_number}")
            try:
                index = int(row["sample_index"])
                if str(index) != row["sample_index"].strip():
                    raise ValueError("non-integer sample_index")
                target = float(row["target_proxy"])
                prediction = float(row["agcag"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid numeric field on CSV row {row_number}") from exc
            if not (math.isfinite(target) and math.isfinite(prediction)):
                raise ValueError(f"Nonfinite value on CSV row {row_number}")
            key = (direction, axis)
            if index != len(groups[key]):
                raise ValueError(f"Nonconsecutive sample_index on CSV row {row_number}")
            groups[key].append(
                {"segment": segment, "target": target, "prediction": prediction}
            )
    if not groups:
        raise ValueError("No held-out rows found")
    return dict(groups)


def calculate(
    groups: dict[tuple[str, str], list[dict[str, object]]]
) -> list[tuple[str, str, str, int, dict[str, float]]]:
    rows = []
    for direction in DIRECTIONS:
        for axis in AXES:
            samples = groups.get((direction, axis), [])
            if not samples:
                continue
            for subset in ("all", "dynamic", "steady"):
                selected = (
                    samples if subset == "all" else
                    [item for item in samples if item["segment"] == subset]
                )
                if not selected:
                    continue
                result = score(
                    [float(item["target"]) for item in selected],
                    [float(item["prediction"]) for item in selected],
                )
                rows.append((direction, axis, subset, len(selected), result))
    for direction in DIRECTIONS:
        for subset in ("all", "dynamic", "steady"):
            axis_rows = [
                item for item in rows
                if item[0] == direction and item[2] == subset and item[1] in AXES
            ]
            if len(axis_rows) != len(AXES):
                continue
            means = {
                metric: math.fsum(item[4][metric] for item in axis_rows) / len(AXES)
                for metric in ("rmse_deg_per_h", "mae_deg_per_h", "r2")
            }
            rows.append((direction, "mean_of_axes", subset,
                         sum(item[3] for item in axis_rows), means))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="Approved pointwise transfer CSV")
    args = parser.parse_args()
    print("direction,axis,subset,n,rmse_deg_per_h,mae_deg_per_h,r2")
    for direction, axis, subset, count, result in calculate(read_transfer(args.csv_path)):
        print(
            f"{direction},{axis},{subset},{count},"
            f"{result['rmse_deg_per_h']:.6f},{result['mae_deg_per_h']:.6f},"
            f"{result['r2']:.6f}"
        )


if __name__ == "__main__":
    main()
