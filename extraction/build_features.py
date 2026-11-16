"""Build labeled ELS features from records stored as <input-dir>/<model>/**/*.jsonl."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark.shared.axes import (AXES, DISCONTINUITY_CONDITIONS, INTERFERENCE_CONDITIONS, QUANTITY_K,
                                   UNDERSPECIFICATION_CONDITIONS)
from benchmark.shared.common import ITEM_FIELDS

# Benchmark tasks a record may carry.
TASKS = tuple(QUANTITY_K)
# Ordered load levels of each non-quantity axis, lowest first; quantity levels come
# from the benchmark definition (quantity_levels).
LEVELS = {
    "similarity": ("low", "high"),
    "interference": INTERFERENCE_CONDITIONS,
    "discontinuity": DISCONTINUITY_CONDITIONS,
    "underspecification": UNDERSPECIFICATION_CONDITIONS,
}
# Prompt regions, and the per-layer magnitude and layer-to-layer dynamics read from each record.
REGIONS = {"ctx": "context", "q": "question", "full": "full"}
DYNAMICS = {"act_mag": "activation_magnitude", "drift": "drift_to_next_layer",
            "angle": "angle_to_next_layer_deg", "M_to_next": "M_to_next_layer"}


def get(d, *keys):
    """Return the nested value at keys, or None if any key is missing."""
    for key in keys:
        if not isinstance(d, dict) or key not in d:
            return None
        d = d[key]
    return d


def number(value) -> float:
    """Return value as a float, or NaN if it is missing."""
    return float(value) if value is not None else np.nan


def pair_ids(record: dict) -> dict:
    """Group key and target of a prompt: the target's base problem and ID for Math and
    Context, and the variant for Analogy, whose targets are generated per variant."""
    # Analogy keys are zero-padded variant numbers, e.g. 'v00042'.
    if record["task"] == "analogy":
        return {"pair_key": f"v{record['variant']:05d}", "target_id": None}
    return {"pair_key": record["target_base_id"], "target_id": record["target_id"]}


def feature_row(record: dict, model: str) -> dict:
    """Convert one feature record into a table row."""
    # Identifiers and label, then the feature columns.
    return {
        "task": record["task"], "model": model, "axis": record["axis"], "condition": record["condition"],
        "variant": record["variant"], **pair_ids(record),
        "correct": number(record.get("correct")),
        **feature_columns(record),
    }


def feature_columns(record: dict) -> dict:
    """The feature columns of one feature record."""
    prompt_time = get(record, "activation_metrics", "prompt_time") or {}
    summary = get(record, "activation_metrics", "generation_trace", "summary") or {}
    dynamics = record.get("neural_dynamics") or {}
    layers = prompt_time.get("layers") or {}
    counts = prompt_time.get("token_counts") or {}
    # Layer keys are layer numbers (strings after a JSON round trip); sort them numerically.
    ordered = [layers[key] for key in sorted(layers, key=int)]

    def per_layer(*keys):
        """One value per layer, in layer order, read at the nested keys (NaN if missing)."""
        return [number(get(layer, *keys)) for layer in ordered]

    eff_rank_ctx, eff_rank_q = per_layer("context", "effective_rank_pr"), per_layer("question", "effective_rank_pr")
    gini_ctx, gini_q = per_layer("context", "gini"), per_layer("question", "gini")
    ctx_q_cos = per_layer("context_question_cos")
    # Token counts, the entropy and margin baselines, and the raw per-layer arrays
    # (_per_layer_*) that later steps summarize by layer position.
    row = {
        "prompt_tok_count": int(counts.get("full", 0)),
        "ctx_tok_count": int(counts.get("context", 0)),
        "q_tok_count": int(counts.get("question", 0)),
        "mean_entropy": number(summary.get("mean_entropy")),
        "mean_logit_margin": number(summary.get("mean_logit_margin_top1_top2")),
        "_per_layer_eff_rank_ctx": eff_rank_ctx,
        "_per_layer_gini_ctx": gini_ctx,
        "_per_layer_ctx_q_cos": ctx_q_cos,
        "_per_layer_eff_rank_q": eff_rank_q,
        "_per_layer_gini_q": gini_q,
        "_per_layer_token_norm_ctx": per_layer("context", "token_norm", "mean"),
        "_per_layer_token_norm_q": per_layer("question", "token_norm", "mean"),
    }
    # Dynamics arrays for every region, e.g. _per_layer_drift_ctx.
    for name, key in DYNAMICS.items():
        for suffix, region in REGIONS.items():
            row[f"_per_layer_{name}_{suffix}"] = [number(x) for x in get(dynamics, "per_layer", region, key) or []]
    return row


def read_features(input_dir: Path) -> pd.DataFrame:
    """Read every feature record under input_dir into one row per prompt."""
    files = sorted(Path(input_dir).glob("*/**/*.jsonl"))
    if not files:
        raise ValueError(f"No feature files in {input_dir}; expected <input-dir>/<model>/**/*.jsonl")
    rows, seen = [], set()
    for path in files:
        # The first directory below input_dir names the model.
        model = path.relative_to(input_dir).parts[0]
        with open(path, encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                location = f"{path}:{line_number}"
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{location}: invalid JSON") from error
                if record.get("task") not in TASKS or record.get("axis") not in AXES:
                    raise ValueError(f"{location}: expected a task in {TASKS} and an axis in {AXES}")
                if record["task"] != "analogy" and not all(isinstance(record.get(field), str) and record[field]
                                                           for field in ITEM_FIELDS):
                    raise ValueError(f"{location}: Math and Context records need {' and '.join(ITEM_FIELDS)}, "
                                     "which cogloadbench items and benchmark run records provide")
                key = (model, record["task"], record["axis"], record["condition"], record["variant"])
                if key in seen:
                    raise ValueError(f"{location}: duplicate prompt {key}")
                seen.add(key)
                rows.append(feature_row(record, model))
    return pd.DataFrame(rows)


def quantity_levels() -> dict:
    """Each task's quantity conditions k<K> in the benchmark's order of K, whether or not a run includes them all."""
    return {task: tuple(f"k{k}" for k in counts) for task, counts in QUANTITY_K.items()}


def level(row, quantity: dict):
    """Return the index of the row's condition among its axis levels (None if absent) and the level count."""
    levels = quantity.get(row.task, ()) if row.axis == "quantity" else LEVELS.get(row.axis, ())
    return levels.index(row.condition) if row.condition in levels else None, len(levels)


def add_load_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep rows with a load level; load_label is 0 at the lowest level, 1 at the highest, missing otherwise."""
    quantity = quantity_levels()
    levels = [level(row, quantity) for row in frame[["task", "axis", "condition"]].itertuples()]
    out = frame.assign(level_index=pd.array([index for index, _ in levels], dtype="Int64"),
                       load_label=pd.array([0 if index == 0 else 1 if index == count - 1 else None
                                            for index, count in levels], dtype="Int64"))
    # Rows whose condition is not a level of its axis (e.g. similarity high_k<K>) are dropped;
    # middle levels keep their level_index with no load_label.
    out = out[out.level_index.notna()].reset_index(drop=True)
    # Place the two new columns right after target_id.
    position = out.columns.get_loc("target_id") + 1
    columns = list(frame.columns)
    return out[columns[:position] + ["level_index", "load_label"] + columns[position:]]


def main(argv=None):
    """Read the feature records, add load labels, and write the parquet file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Directory of feature records, one subdirectory per model")
    parser.add_argument("--output", type=Path, required=True, help="Labeled parquet file for ELS --features")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output file")
    args = parser.parse_args(argv)
    if not args.input_dir.is_dir():
        parser.error(f"Not a directory: {args.input_dir}")
    destination = args.output.resolve()
    if destination == args.input_dir.resolve() or any(
            path.resolve() == destination for path in args.input_dir.glob("*/**/*.jsonl")):
        parser.error("--output must differ from the feature inputs")
    if args.output.exists() and not args.force:
        parser.error(f"{args.output} exists; pass --force to overwrite")
    frame = add_load_labels(read_features(args.input_dir))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output)
    print(f"Wrote {len(frame):,} labeled rows to {args.output}")


if __name__ == "__main__":
    main()
