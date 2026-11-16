"""Validate benchmark records and aggregate them into CSV tables and values.json."""
# The saved benchmark results hold one JSONL file of graded records per task, axis
# and model, plus results/counts.tsv with each cell's correct count k and size n.
# validate_benchmark_inputs checks that both are complete and agree; the remaining
# functions compute per-model and macro accuracies with standard errors and write
# the benchmark tables. evaluation.benchmark.verify compares the same values with
# the paper.
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import stdev

from benchmark.shared.axes import (DISCONTINUITY_CONDITIONS, INTERFERENCE_CONDITIONS, QUANTITY_K, SIMILARITY_K,
                                   SIMILARITY_SWEEP_K, UNDERSPECIFICATION_CONDITIONS)
from benchmark.shared.common import ITEM_FIELDS
from evaluation.common import run
from tools.datadir import CODE_ROOT, DATA_ROOT, MISSING_DATA

# Saved benchmark data, its graded records and counts, and the default table directory.
BENCHMARK_DATA = DATA_ROOT / "benchmark"
RESULTS = BENCHMARK_DATA / "results"
DEFAULT_OUTPUT = CODE_ROOT / "outputs/benchmark"

# Tasks in table row order (math, context, analogy), and the evaluated models.
TASKS = list(QUANTITY_K)
TASK_LABEL = {"math": "Math", "context": "Context", "analogy": "Analogy"}
MODELS = ["Qwen3-4B", "Qwen3-8B", "Qwen3-14B", "Gemma3-12B", "Llama3.1-8B"]
VARIANTS = 200  # Prompt variants per task, axis, model and condition.

# K values of the quantity columns, over all tasks.
QUANTITY_COLUMNS = sorted(set().union(*QUANTITY_K.values()))
# Axis -> ordered (condition, column header); the first condition is the baseline.
AXES = {
    "quantity": [(f"k{k}", f"K={k}") for k in QUANTITY_COLUMNS],
    "similarity": [("low", "Low similarity"), ("high", "High similarity")],
    **{axis: [(name, name.replace("_", " ").capitalize()) for name in names]
       for axis, names in (("interference", INTERFERENCE_CONDITIONS),
                           ("discontinuity", DISCONTINUITY_CONDITIONS),
                           ("underspecification", UNDERSPECIFICATION_CONDITIONS))},
}

# Columns of results/counts.tsv, and keys of every benchmark record; Math and Context
# records also have the target IDs (ITEM_FIELDS).
COUNTS_FIELDS = ["task", "axis", "model", "condition", "n", "k", "acc"]
RECORD_FIELDS = ["condition", "variant", "prompt", "raw_model_text", "parsed", "pred", "gold", "correct"]


def record_fields(task: str) -> set[str]:
    """Keys of a benchmark record of the task."""
    return set(RECORD_FIELDS) | (set() if task == "analogy" else set(ITEM_FIELDS))


def conditions_for(task: str, axis: str, *, include_curve: bool = False) -> set[str]:
    """Conditions run for one task and axis; include_curve adds the extra similarity-curve conditions.

    The curve's extra conditions are high_k<K> for each K in SIMILARITY_SWEEP_K; its K=1
    point comes from quantity and the task's similarity K from the 'high' condition.
    """
    if axis == "quantity":
        return {f"k{k}" for k in QUANTITY_K[task]}
    conditions = {name for name, _ in AXES[axis]}
    if include_curve and axis == "similarity":
        conditions.update(f"high_k{k}" for k in SIMILARITY_SWEEP_K[task])
    return conditions


def benchmark_coordinates() -> set[tuple[str, str, str, str]]:
    """Every saved (task, axis, model, condition) cell, including the similarity-curve cells."""
    return {(task, axis, model, condition)
            for task in TASKS for axis in AXES for model in MODELS
            for condition in conditions_for(task, axis, include_curve=True)}


def reference_coordinates() -> set[tuple[str, ...]]:
    """Paths of every accuracy compared with the paper: macro and per-model cells, without the curve cells."""
    keys = set()
    for task in TASKS:
        for axis in AXES:
            for condition in conditions_for(task, axis):
                keys.add(("macro", axis, task, condition))
                keys.update(("per_model", axis, task, model, condition) for model in MODELS)
    return keys


def validate_benchmark_inputs(root: Path) -> None:
    """Check that the count table and the records cover every expected cell and agree."""
    # Count table: one well-formed row per expected cell, with VARIANTS prompts and acc = 100 k / n.
    with (root / "results/counts.tsv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != COUNTS_FIELDS:
            raise ValueError("Invalid benchmark count table columns")
        rows = list(reader)
    expected = benchmark_coordinates()
    by_coordinate = {}
    # DictReader stores surplus fields under the key None and missing fields as None values.
    for row in rows:
        if None in row or not all(row.values()):
            raise ValueError("Malformed benchmark count table row")
        key = tuple(row[field] for field in COUNTS_FIELDS[:4])
        if key in by_coordinate:
            raise ValueError(f"Duplicate benchmark coordinate: {key}")
        if key not in expected:
            raise ValueError(f"Unexpected benchmark coordinate: {key}")
        if not row["n"].isdigit() or not row["k"].isdigit():
            raise ValueError(f"Invalid benchmark counts: {key}")
        n, k, accuracy = int(row["n"]), int(row["k"]), float(row["acc"])
        if n != VARIANTS or not 0 <= k <= n or not math.isfinite(accuracy):
            raise ValueError(f"Invalid benchmark counts/accuracy: {key}")
        if not math.isclose(accuracy, 100 * k / n, abs_tol=1e-10, rel_tol=0):
            raise ValueError(f"Benchmark count table accuracy disagrees with k/n: {key}")
        by_coordinate[key] = row
    if set(by_coordinate) != expected:
        raise ValueError(f"Incomplete benchmark count table: expected {len(expected)} cells, "
                         f"found {len(by_coordinate)}")

    # Records: exactly the expected files, valid records, and variant IDs 0 to VARIANTS - 1
    # once each per cell, with correct counts equal to the count table's k.
    results = root / "results"
    paths = {results / task / axis / f"{model}.jsonl" for task, axis, model, _ in expected}
    if set(results.rglob("*.jsonl")) != paths:
        raise ValueError("Benchmark record files differ from the expected cells")
    observed = {}
    correct_counts = {}
    for path in sorted(paths):
        task, axis, model = path.parent.parent.name, path.parent.name, path.stem
        fields = record_fields(task)
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                record = json.loads(line)
                if not isinstance(record, dict) or set(record) != fields:
                    raise ValueError(f"Invalid benchmark record: {path}:{line_number}")
                key = (task, axis, model, record["condition"])
                if key not in expected:
                    raise ValueError(f"Unexpected benchmark cell: {path}:{line_number}")
                variant, correct = record["variant"], record["correct"]
                if type(variant) is not int or not 0 <= variant < VARIANTS:
                    raise ValueError(f"Invalid variant ID: {path}:{line_number}")
                if type(correct) is not int or correct not in (0, 1):
                    raise ValueError(f"Invalid correctness value: {path}:{line_number}")
                ids = observed.setdefault(key, set())
                if variant in ids:
                    raise ValueError(f"Duplicate variant ID {variant}: {key}")
                ids.add(variant)
                correct_counts[key] = correct_counts.get(key, 0) + correct
    for key, row in by_coordinate.items():
        if observed.get(key, set()) != set(range(VARIANTS)) or correct_counts.get(key, 0) != int(row["k"]):
            raise ValueError(f"Record counts or IDs disagree with the count table: {key}")


def records(results: Path):
    """Yield (task, axis, model, record) for every benchmark record under results."""
    # Record files are laid out as <results>/<task>/<axis>/<model>.jsonl.
    for path in sorted(results.rglob("*.jsonl")):
        task, axis, model = path.parts[-3], path.parts[-2], path.stem
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                yield task, axis, model, json.loads(line)


def count_cells(results: Path) -> dict:
    """counts[task][axis][model][condition] = (k, n)."""
    counts: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for task, axis, model, record in records(results):
        k, n = counts[task][axis][model].get(record["condition"], (0, 0))
        counts[task][axis][model][record["condition"]] = (k + int(record["correct"]), n + 1)
    return counts


def variant_macro_cells(results: Path) -> dict:
    """Macro accuracy and SE over variants, each variant averaged across models first."""
    # rows[(task, axis, condition)][variant] collects the correctness of every model.
    rows: dict = defaultdict(lambda: defaultdict(list))
    for task, axis, _, record in records(results):
        rows[task, axis, record["condition"]][int(record["variant"])].append(float(record["correct"]))
    stats = {}
    for key, by_variant in rows.items():
        # The SE is the sample standard deviation of the variant means over sqrt(variants).
        variant_means = [sum(values) / len(values) for values in by_variant.values()]
        n_variants = len(variant_means)
        mean = 100 * sum(variant_means) / n_variants
        se = 100 * stdev(variant_means) / math.sqrt(n_variants) if n_variants > 1 else 0.0
        stats[key] = {"acc": mean, "se": se, "n_variants": n_variants}
    return stats


def acc_se(k: int, n: int) -> tuple[float, float]:
    """Accuracy and binomial standard error, in percent."""
    p = k / n
    return 100 * p, 100 * math.sqrt(p * (1 - p) / n)


def macro_cell(counts: dict, variant_stats: dict, task: str, axis: str, condition: str) -> dict | None:
    """Variant-averaged macro accuracy and SE with the pooled counts of one cell.

    k and n are summed over the models that ran the cell; returns None if none did.
    """
    cells = [counts.get(task, {}).get(axis, {}).get(model, {}).get(condition) for model in MODELS]
    cells = [kn for kn in cells if kn]
    if not cells:
        return None
    stats = variant_stats[(task, axis, condition)]
    return {"acc": stats["acc"], "se": stats["se"], "k": sum(k for k, _ in cells),
            "n": sum(n for _, n in cells), "n_variants": stats["n_variants"]}


def build_values(counts: dict, variant_stats: dict) -> dict:
    """Per-model and macro accuracy and SE for every cell."""
    values = {"per_model": {}, "macro": {}}
    for axis, conditions in AXES.items():
        per_model: dict = {}
        macro: dict = {}
        for task in TASKS:
            axis_counts = counts.get(task, {}).get(axis)
            if not axis_counts:
                continue
            per_model[task] = {}
            macro[task] = {}
            for condition, _ in conditions:
                for model in MODELS:
                    kn = axis_counts.get(model, {}).get(condition)
                    if kn is not None:
                        a, se = acc_se(*kn)
                        per_model[task].setdefault(model, {})[condition] = {"acc": a, "se": se, "k": kn[0], "n": kn[1]}
                cell = macro_cell(counts, variant_stats, task, axis, condition)
                if cell:
                    macro[task][condition] = cell
        values["per_model"][axis] = per_model
        values["macro"][axis] = macro
    return values


def format_csv_cell(cell: dict | None, baseline: dict | None = None) -> str:
    """Format accuracy, SE, and the change from baseline, e.g. '61.5±1.2 (↓3.0)'.

    A missing cell is '---'; the change is omitted for the baseline itself and when it
    rounds to 0.0.
    """
    if not cell:
        return "---"
    s = f"{cell['acc']:.1f}±{cell['se']:.1f}"
    if baseline is not None and baseline is not cell:
        delta = cell["acc"] - baseline["acc"]
        if round(abs(delta), 1) > 0:
            s += f" ({'↑' if delta > 0 else '↓'}{abs(delta):.1f})"
    return s


def write_macro_table(output: Path, name: str, axis: str, values: dict) -> None:
    """Write one row per task and one column per condition."""
    conditions = AXES[axis]
    macro = values["macro"][axis]
    rows = [[TASK_LABEL[task]] + [format_csv_cell(macro[task].get(c), macro[task].get(conditions[0][0]))
                                  for c, _ in conditions]
            for task in TASKS if task in macro]
    _write_table(output, name, ["Task"] + [h for _, h in conditions], rows)


def write_per_model_table(output: Path, name: str, axis: str, values: dict) -> None:
    """Write one row per task and model and one column per condition."""
    conditions = AXES[axis]
    per_model = values["per_model"][axis]
    rows = []
    for task in TASKS:
        if task not in per_model:
            continue
        for i, model in enumerate(MODELS):
            cells = per_model[task].get(model, {})
            base = cells.get(conditions[0][0])
            label = TASK_LABEL[task] if i == 0 else ""
            rows.append([label, model] + [format_csv_cell(cells.get(c), base) for c, _ in conditions])
    _write_table(output, name, ["Task", "Model"] + [h for _, h in conditions], rows)


def _write_table(output: Path, name: str, header: list, rows: list) -> None:
    """Write one CSV table to the output directory."""
    output.mkdir(parents=True, exist_ok=True)
    with open(output / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def write_similarity_curve(output: Path, counts: dict, variant_stats: dict) -> None:
    """Write macro accuracy against K for low- and high-similarity distractors.

    Low uses quantity k{K}; high uses similarity high_k{K}, or 'high' at the task's
    similarity K (marked '*'). K=1 is shared by both curves."""
    rows = []
    for task in TASKS:
        anchor = SIMILARITY_K[task]
        low = {K: macro_cell(counts, variant_stats, task, "quantity", f"k{K}") for K in QUANTITY_COLUMNS}
        high = {K: macro_cell(counts, variant_stats, task, "similarity", "high" if K == anchor else f"high_k{K}")
                for K in QUANTITY_COLUMNS}
        # K=1 shows only the target problem, so both curves use the quantity k1 cell.
        high[1] = low[1]
        for i, (curve, cells) in enumerate((("low", low), ("high", high))):
            label = TASK_LABEL[task] if i == 0 else ""
            rows.append([label, f"{curve.capitalize()} similarity"]
                        + [format_csv_cell(cells[K]) + ("*" if curve == "high" and K == anchor else "")
                           for K in QUANTITY_COLUMNS])
    _write_table(output, "quantity_similarity_curve", ["Task", "Curve"] + [f"K={K}" for K in QUANTITY_COLUMNS], rows)


# Output name, axis, and aggregation level for each CSV.
TABLES = [
    ("interference", "interference", "macro"),
    ("discontinuity", "discontinuity", "macro"),
    ("underspecification", "underspecification", "macro"),
    ("quantity", "quantity", "macro"),
    ("similarity", "similarity", "macro"),
    ("quantity_per_model", "quantity", "per_model"),
    ("similarity_per_model", "similarity", "per_model"),
    ("interference_per_model", "interference", "per_model"),
    ("discontinuity_per_model", "discontinuity", "per_model"),
    ("underspecification_per_model", "underspecification", "per_model"),
]


def main(argv=None) -> int:
    """Validate the saved records and write values.json and the benchmark tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help="Directory for the tables (default: outputs/benchmark)")
    args = parser.parse_args(argv)
    if not (RESULTS / "counts.tsv").is_file():
        parser.error(f"Missing {RESULTS}. {MISSING_DATA}")
    output = args.output_dir.resolve()
    validate_benchmark_inputs(BENCHMARK_DATA)
    counts = count_cells(RESULTS)
    variant_stats = variant_macro_cells(RESULTS)
    values = build_values(counts, variant_stats)
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "values.json", "w", encoding="utf-8") as f:
        json.dump(values, f, indent=1)
    for name, axis, kind in TABLES:
        (write_macro_table if kind == "macro" else write_per_model_table)(output, name, axis, values)
    write_similarity_curve(output, counts, variant_stats)
    print(f"Benchmark tables written to {output}")
    return 0


if __name__ == "__main__":
    run(main)
