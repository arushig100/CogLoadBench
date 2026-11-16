"""Check saved paired steering scores against the ELS-guided column of the paper's steering table (Table 5b)."""
# data/steering holds, for each dataset, model and steered axis, the baseline and
# ELS-steered F1 of the same 300 test examples (scores.csv), the fixed train,
# alpha-validation and test keys (splits/), and the paper's ELS gain and paired SE
# of every cell (reference.csv). Gains are recomputed from the paired scores and,
# rounded to two decimals as printed, must equal the references.
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from evaluation.common import run
from tools.datadir import DATA_ROOT, MISSING_DATA

# Saved steering data, and the datasets, models and steered axes of the table.
STEERING = DATA_ROOT / "steering"
DATASETS = ("cuad", "qasper")
MODELS = ("Qwen-Qwen3-14B", "Qwen-Qwen3-32B")
AXES = ("quantity", "similarity", "interference")
# Columns that identify one steering cell.
KEY_FIELDS = ("dataset", "model", "axis")
# Keys in each dataset and model's train, alpha-validation and test splits.
N_TRAIN, N_VALIDATION, N_TEST = 150, 200, 300
# The paper's mean ELS steering gain over the 12 cells, in percentage points.
MEAN_GAIN = 0.42


def read_csv(path: Path) -> list[dict[str, str]]:
    """Rows of a CSV file as dictionaries of strings."""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close(actual: float, expected: float, *, tolerance: float = 1e-12) -> None:
    """Require two finite values within an absolute tolerance."""
    if not (math.isfinite(actual) and math.isfinite(expected)) or not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"{actual!r} != {expected!r} (abs tolerance {tolerance})")


def cell_key(row: dict[str, Any]) -> tuple[str, str, str]:
    """(dataset, model, axis) of a score or reference row."""
    return tuple(str(row[field]) for field in KEY_FIELDS)  # type: ignore[return-value]


def check_splits(package: Path) -> dict[tuple[str, str], set[str]]:
    """Check the fixed train, validation and test keys of each dataset and model; return the test keys.

    Each split must hold the expected number of unique, nonempty keys, and no key may
    appear in two splits.
    """
    split_root = package / "splits"
    test_keys: dict[tuple[str, str], set[str]] = {}
    for dataset in DATASETS:
        for model in MODELS:
            prefix = f"{dataset}_{model}"
            paths = {
                "train": split_root / f"{prefix}_train_keys.json",
                "validation": split_root / f"{prefix}_alpha_val_keys.json",
                "test": split_root / f"{prefix}_test_keys.json",
            }
            values = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
            for name, count in (("train", N_TRAIN), ("validation", N_VALIDATION), ("test", N_TEST)):
                if (not isinstance(values[name], list)
                        or any(not isinstance(key, str) or not key.strip() for key in values[name])
                        or len(values[name]) != count or len(set(values[name])) != count):
                    raise ValueError(f"{prefix}: the {name} split must hold {count} unique nonempty string keys")
            for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
                if set(values[left]) & set(values[right]):
                    raise ValueError(f"{prefix}: {left}/{right} overlap")
            test_keys[(dataset, model)] = set(values["test"])
    return test_keys


def recompute_cells(
    package: Path,
    expected: dict[tuple[str, str, str], dict[str, str]],
    test_keys: dict[tuple[str, str], set[str]],
) -> list[dict[str, Any]]:
    """Recompute each cell's paired F1 gain and SE from saved scores and check the references."""
    # Every score row must belong to a reference cell, and every reference cell needs rows.
    rows = read_csv(package / "scores.csv")
    n_expected = N_TEST * len(DATASETS) * len(MODELS) * len(AXES)
    if len(rows) != n_expected:
        raise ValueError(f"Expected {n_expected:,} paired score rows, found {len(rows)}")
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = cell_key(row)
        if key not in expected:
            raise ValueError(f"Unexpected paired-score cell: {key}")
        grouped[key].append(row)
    if set(grouped) != set(expected):
        raise ValueError("Paired-score cells do not equal the reference cells")

    computed = []
    for key, cell_rows in grouped.items():
        ref = expected[key]
        if (not math.isfinite(float(ref["alpha"])) or int(ref["n"]) != N_TEST
                or not math.isfinite(float(ref["gain"]))
                or not math.isfinite(float(ref["paired_se"])) or float(ref["paired_se"]) < 0):
            raise ValueError(f"{key}: invalid steering reference values")
        # Each cell scores exactly the fixed test examples once, at the reference alpha.
        ids = [row["id"] for row in cell_rows]
        if len(ids) != N_TEST or len(set(ids)) != N_TEST:
            raise ValueError(f"{key}: expected {N_TEST} unique paired IDs")
        if set(ids) != test_keys[(key[0], key[1])]:
            raise ValueError(f"{key}: paired IDs differ from the fixed test keys")
        alphas = {float(row["alpha"]) for row in cell_rows}
        if alphas != {float(ref["alpha"])}:
            raise ValueError(f"{key}: alpha mismatch {alphas}")
        # Paired change in F1 (percentage points) for each example; the gain is their
        # mean and the paired SE their standard error.
        # statistics.stdev is the sample standard deviation (n - 1 denominator).
        changes = []
        for row in cell_rows:
            baseline = float(row["baseline_f1"])
            steered = float(row["steered_f1"])
            if not (math.isfinite(baseline) and math.isfinite(steered)):
                raise ValueError(f"{key}: nonfinite score")
            if not (0.0 <= baseline <= 1.0 and 0.0 <= steered <= 1.0):
                raise ValueError(f"{key}: F1 outside [0,1]")
            changes.append(100.0 * (steered - baseline))
        result = {
            "dataset": key[0],
            "model": key[1],
            "axis": key[2],
            "alpha": float(ref["alpha"]),
            "n": len(changes),
            "gain": statistics.mean(changes),
            "paired_se": statistics.stdev(changes) / math.sqrt(len(changes)),
        }
        close(round(result["gain"], 2), float(ref["gain"]))
        close(round(result["paired_se"], 2), float(ref["paired_se"]))
        computed.append(result)
    # Return the cells in table order: dataset, then model, then axis.
    computed.sort(key=lambda row: (DATASETS.index(row["dataset"]), MODELS.index(row["model"]), AXES.index(row["axis"])))
    return computed


def main(argv=None) -> int:
    """Check the reference table, the splits and every steering cell; return 0 on success."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    if not (STEERING / "reference.csv").is_file():
        parser.error(f"Missing {STEERING}. {MISSING_DATA}")
    reference = read_csv(STEERING / "reference.csv")
    expected = {cell_key(row): row for row in reference}
    required = {(dataset, model, axis) for dataset in DATASETS for model in MODELS for axis in AXES}
    if len(reference) != len(required) or set(expected) != required:
        raise ValueError(f"Expected {len(required)} unique dataset/model/axis reference cells")
    rows = recompute_cells(STEERING, expected, check_splits(STEERING))
    gain = statistics.mean(row["gain"] for row in rows)
    close(round(gain, 2), MEAN_GAIN)
    print(f"Mean paired steering gain: {gain:+.2f} percentage points")
    print(f"PASS: {len(rows)} steering cells ({sum(row['n'] for row in rows):,} paired scores) match the paper.")
    return 0


if __name__ == "__main__":
    run(main)
