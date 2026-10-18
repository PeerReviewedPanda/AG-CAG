"""Recalculate held-out proxy-target metrics without model or hardware access."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


METHODS = (
    "svr", "random_forest", "standard_lstm",
    "patch_transformer", "tsmixer", "agcag",
)
COLUMNS = ("sample_index", "target_proxy", *METHODS)


def read_predictions(path: Path) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {key: [] for key in COLUMNS}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or set(reader.fieldnames) != set(COLUMNS):
            raise ValueError(f"Expected exactly these columns: {', '.join(COLUMNS)}")
        for row_number, row in enumerate(reader, start=2):
            try:
                index = int(row["sample_index"])
                if str(index) != row["sample_index"].strip():
                    raise ValueError("non-integer sample_index")
                numbers = {key: float(row[key]) for key in COLUMNS[1:]}
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid numeric field on CSV row {row_number}") from exc
            if index != len(values["sample_index"]):
                raise ValueError(f"Nonconsecutive sample_index on CSV row {row_number}")
            if not all(math.isfinite(item) for item in numbers.values()):
                raise ValueError(f"Nonfinite value on CSV row {row_number}")
            values["sample_index"].append(index)
            for key, value in numbers.items():
                values[key].append(value)
    if len(values["sample_index"]) < 2:
        raise ValueError("At least two held-out samples are required")
    return values


def score(target: list[float], prediction: list[float]) -> dict[str, float]:
    if len(target) != len(prediction) or len(target) < 2:
        raise ValueError("Target and prediction lengths must agree and be at least two")
    if not all(math.isfinite(v) for v in [*target, *prediction]):
        raise ValueError("Scores require finite inputs")
    n = len(target)
    mean = math.fsum(target) / n
    residual_sq = math.fsum((a - b) ** 2 for a, b in zip(target, prediction))
    target_sq = math.fsum((a - mean) ** 2 for a in target)
    if target_sq == 0:
        raise ValueError("R² is undefined for a constant target")
    return {
        "rmse_deg_per_h": math.sqrt(residual_sq / n),
        "mae_deg_per_h": math.fsum(abs(a - b) for a, b in zip(target, prediction)) / n,
        "r2": 1.0 - residual_sq / target_sq,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="Approved held-out prediction CSV")
    args = parser.parse_args()
    values = read_predictions(args.csv_path)
    print(f"{args.csv_path.name}: {len(values['target_proxy'])} held-out rows")
    print("method,rmse_deg_per_h,mae_deg_per_h,r2")
    for method in METHODS:
        result = score(values["target_proxy"], values[method])
        print(
            f"{method},{result['rmse_deg_per_h']:.6f},"
            f"{result['mae_deg_per_h']:.6f},{result['r2']:.6f}"
        )


if __name__ == "__main__":
    main()
