"""Check the saved benchmark records against the paper's accuracy values and standard errors."""
# The paper's values live in data/benchmark/reference_values.json, laid out like
# values.json from evaluation.benchmark.compute: macro[axis][task][condition] and
# per_model[axis][task][model][condition], each an {"acc": ..., "se": ...} cell.
# Each recomputed accuracy and SE, formatted to one decimal place as in the paper's
# tables, must equal its reference value.
from __future__ import annotations

import argparse
import json
import math

from evaluation.benchmark.compute import (BENCHMARK_DATA, RESULTS, build_values, count_cells, reference_coordinates,
                                          validate_benchmark_inputs, variant_macro_cells)
from evaluation.common import run, unique_object
from tools.datadir import MISSING_DATA

REFERENCE_VALUES = BENCHMARK_DATA / "reference_values.json"
STATS = ("acc", "se")  # Values of each cell, as printed in the paper's tables.


def validate_expected(expected: dict) -> None:
    """Check the reference cells hold an accuracy and SE in [0, 100] and cover exactly the expected cells."""
    if not isinstance(expected, dict) or set(expected) != {"macro", "per_model"}:
        raise ValueError("Invalid reference sections")
    cells = set()
    required = reference_coordinates()
    # Every partial path of a required cell is allowed, so unexpected keys fail at the first wrong level.
    prefixes = {key[:length] for key in required for length in range(1, len(key) + 1)}

    def visit(value, path):
        """Walk the nested reference, recording each cell by its key path."""
        if path not in prefixes:
            raise ValueError(f'Unexpected reference: {"/".join(path)}')
        # type() rather than isinstance() so that booleans are not accepted as numbers.
        if path in required:
            if not isinstance(value, dict) or set(value) != set(STATS) or not all(
                    type(value[stat]) in (int, float) and math.isfinite(value[stat]) and 0 <= value[stat] <= 100
                    for stat in STATS):
                raise ValueError(f'Invalid reference cell: {"/".join(path)}')
            cells.add(path)
        elif isinstance(value, dict):
            for key, child in value.items():
                visit(child, (*path, key))
        else:
            raise ValueError(f'Invalid reference branch: {"/".join(path)}')

    for section in ("macro", "per_model"):
        visit(expected[section], (section,))
    if cells != required:
        raise ValueError(f"Reference coverage mismatch: missing={len(required - cells)}, "
                         f"extra={len(cells - required)}; expected {len(required)} cells")


def main(argv=None) -> int:
    """Compare every recomputed accuracy and SE with the reference values; return 0 on a match."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    if not REFERENCE_VALUES.is_file():
        parser.error(f"Missing {REFERENCE_VALUES}. {MISSING_DATA}")
    expected = json.loads(REFERENCE_VALUES.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    validate_expected(expected)
    validate_benchmark_inputs(BENCHMARK_DATA)
    computed = build_values(count_cells(RESULTS), variant_macro_cells(RESULTS))

    # Look up each required path in the computed values; a missing cell counts as a mismatch.
    mismatches = []
    for path in sorted(reference_coordinates()):
        node = computed
        for key in path:
            node = node.get(key, {}) if isinstance(node, dict) else {}
        reference = expected
        for key in path:
            reference = reference[key]
        # Compare each value after formatting it to one decimal place.
        for stat in STATS:
            got = node.get(stat)
            if type(got) not in (int, float) or not math.isfinite(got) or float(f"{got:.1f}") != reference[stat]:
                mismatches.append(f"{'/'.join(path)} {stat}: expected {reference[stat]}, computed {got}")
    total = len(reference_coordinates())
    if mismatches:
        print(f"FAIL: {len(mismatches)} benchmark values in {total} cells differ from the paper:")
        for line in mismatches:
            print("  ", line)
        return 1
    print(f"PASS: {total}/{total} benchmark accuracies and SEs match the paper.")
    return 0


if __name__ == "__main__":
    run(main)
