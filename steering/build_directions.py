#!/usr/bin/env python3
"""Build behavior-calibrated ELS steering directions from CUAD and QASPER extraction runs.

Train task-local ELS on binary low/high-load pairs, calibrate its feature
representation to training-split behavioral drops, and fit layerwise ridge
directions from hidden states to calibrated severity.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

STEERING_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(STEERING_ROOT.parent))

from els.scoring import fit_mlp  # noqa: E402
from els.features import ELS_FEATURES, aggregated  # noqa: E402
from extraction.build_features import DYNAMICS, REGIONS, feature_columns  # noqa: E402
from steering.direction_utils import (  # noqa: E402
    alpha_validation_keys,
    cuad_candidate_keys,
    extraction_paths,
    fit_ridge_directions,
    load_activation,
    qasper_candidate_keys,
    stable_shuffle,
)


# --- Datasets, models, and conditions ---
# Record field that identifies a target question in each runner's output.
ID_FIELD = {"cuad": "qa_id", "qasper": "question_id"}
# Supported model directories (runtime.model_slug of the model) and their Hugging Face IDs.
HF_MODEL = {
    "Qwen-Qwen3-14B": "Qwen/Qwen3-14B",
    "Qwen-Qwen3-32B": "Qwen/Qwen3-32B",
}
# Fixed middle-to-late steering layers used in the paper, as 0-based decoder layer indices.
STEERING_LAYERS = {
    "Qwen-Qwen3-14B": [16, 20, 24],
    "Qwen-Qwen3-32B": [24, 32, 40],
}

# Settings a question needs in the extraction run to be used, with their load family.
SETTINGS: dict[str, list[dict[str, str]]] = {
    "cuad": [
        {"setting": "k1", "family": "quantity"},
        {"setting": "k5", "family": "quantity"},
        {"setting": "k20", "family": "quantity"},
        {"setting": "k50", "family": "quantity"},
        {"setting": "high_k5", "family": "similarity"},
        {"setting": "high_k20", "family": "similarity"},
        {"setting": "high_k50", "family": "similarity"},
        {"setting": "none", "family": "interference"},
        {"setting": "summary", "family": "interference"},
        {"setting": "extra_questions", "family": "interference"},
    ],
    "qasper": [
        {"setting": "k1", "family": "quantity"},
        {"setting": "k3", "family": "quantity"},
        {"setting": "k5", "family": "quantity"},
        {"setting": "k7", "family": "quantity"},
        {"setting": "high_k3", "family": "similarity"},
        {"setting": "high_k5", "family": "similarity"},
        {"setting": "none", "family": "interference"},
        {"setting": "extra_questions", "family": "interference"},
    ],
}

# Binary low/high-load contrasts for the raw ELS training stage.
ELS_PAIRS: dict[str, list[dict[str, str]]] = {
    "cuad": [
        {"low": "k1", "high": "k5", "family": "quantity"},
        {"low": "k1", "high": "k20", "family": "quantity"},
        {"low": "k1", "high": "k50", "family": "quantity"},
        {"low": "k5", "high": "high_k5", "family": "similarity"},
        {"low": "k20", "high": "high_k20", "family": "similarity"},
        {"low": "k50", "high": "high_k50", "family": "similarity"},
        {"low": "none", "high": "summary", "family": "interference"},
        {"low": "none", "high": "extra_questions", "family": "interference"},
    ],
    "qasper": [
        {"low": "k1", "high": "k3", "family": "quantity"},
        {"low": "k1", "high": "k5", "family": "quantity"},
        {"low": "k1", "high": "k7", "family": "quantity"},
        {"low": "k3", "high": "high_k3", "family": "similarity"},
        {"low": "k5", "high": "high_k5", "family": "similarity"},
        {"low": "none", "high": "extra_questions", "family": "interference"},
    ],
}

BASELINE_SETTING = "k1"  # Severity is the training-split F1 drop from this setting.
# Complete examples drawn after the training split and kept out of the test and alpha-validation splits.
N_HELD_BACK = 50


# --- Reading and indexing extraction records ---
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read extraction records, rejecting malformed lines."""
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number}: invalid extraction JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected an extraction object")
            rows.append(row)
    return rows


def build_index(rows: list[dict[str, Any]], dataset: str) -> dict[str, dict[str, dict[str, Any]]]:
    """Index records by question and setting, rejecting duplicates and conflicting variant indices."""
    id_field = ID_FIELD[dataset]
    out: dict[str, dict[str, dict[str, Any]]] = {}
    variants = {}
    for row in rows:
        setting = row.get("setting")
        key = row.get(id_field)
        if not isinstance(setting, str) or not setting.strip() or not isinstance(key, str) or not key.strip():
            raise ValueError(f"Each {dataset} extraction needs nonempty setting and {id_field} strings")
        # Activation files are named by setting and variant index alone
        # (els__<setting>__v<variant>.pt), so each (setting, variant) pair must
        # belong to one question.
        variant = row.get("variant_index")
        if type(variant) is not int or variant < 0:
            raise ValueError(f"{dataset}/{key}/{setting}: invalid variant_index")
        by_setting = out.setdefault(key, {})
        if setting in by_setting:
            raise ValueError(f"Duplicate extraction identity: {dataset}/{key}/{setting}")
        coordinate = (setting, variant)
        if coordinate in variants and variants[coordinate] != key:
            raise ValueError(f"Conflicting extraction variant: {dataset}/{setting}/{variant}")
        variants[coordinate] = key
        by_setting[setting] = row
    return out


def complete_keys(index: dict[str, dict[str, dict[str, Any]]], dataset: str) -> list[str]:
    """Sorted question ids that have a record for every setting in SETTINGS."""
    needed = [str(spec["setting"]) for spec in SETTINGS[dataset]]
    return sorted([key for key, by_setting in index.items() if all(s in by_setting for s in needed)])


def row_score(row: dict[str, Any]) -> float:
    """The record's score as a float; NaN if missing or non-finite."""
    try:
        value = float(row["score"])
    except (KeyError, TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


# --- Feature sources and feature rows ---
def missing_els_sources(row: dict[str, Any]) -> list[str]:
    """ELS features whose source field is absent from one extraction record.

    Feature names are <metric>_<region>__<layer position>, e.g. gini_q__mid. Layer
    statistics must be present in every layer of activation_metrics.prompt_time.layers,
    and layer-to-layer dynamics must be nonempty arrays in neural_dynamics.per_layer. A
    present null or NaN value counts as present.
    """
    def has_path(value: Any, path: tuple[str, ...]) -> bool:
        """True if the nested keys in path exist in value."""
        for key in path:
            if not isinstance(value, dict) or key not in value:
                return False
            value = value[key]
        return True

    prompt_time = (row.get("activation_metrics") or {}).get("prompt_time") or {}
    layers = list((prompt_time.get("layers") or {}).values())
    per_layer = (row.get("neural_dynamics") or {}).get("per_layer") or {}
    metric_paths = {"eff_rank": ("effective_rank_pr",), "gini": ("gini",), "token_norm": ("token_norm", "mean")}
    missing = []
    # Map each feature to its source: ctx_q_cos is a layer statistic of its own; other
    # bases split at the last underscore into a metric and a region suffix (gini_q ->
    # gini, q), and REGIONS maps the suffix to the record's region name.
    for feature in ELS_FEATURES:
        base = feature.split("__", 1)[0]
        if base == "ctx_q_cos":
            available = bool(layers) and all(has_path(layer, ("context_question_cos",)) for layer in layers)
        else:
            metric, suffix = base.rsplit("_", 1)
            region = REGIONS[suffix]
            if metric in DYNAMICS:
                array = (per_layer.get(region) or {}).get(DYNAMICS[metric])
                available = isinstance(array, (list, tuple)) and len(array) > 0
            elif metric in metric_paths:
                available = bool(layers) and all(has_path(layer, (region, *metric_paths[metric]))
                                                 for layer in layers)
            else:
                raise RuntimeError(f"No source-schema check for ELS feature {feature}")
        if not available:
            missing.append(feature)
    return missing


def require_els_feature_sources(rows: list[dict[str, Any]], context: str) -> None:
    """Raise if any record lacks the source field of any ELS feature."""
    for row in rows:
        missing = missing_els_sources(row)
        if missing:
            raise RuntimeError(
                f"{context}: {row.get('setting')} v{row.get('variant_index')}: absent ELS feature sources "
                f"({len(missing)}/{len(ELS_FEATURES)}): {missing}. Rerun extraction with --els_mode."
            )


def feature_record(
    row: dict[str, Any],
    dataset: str,
    model_dir: str,
    family: str,
    key: str,
) -> dict[str, Any]:
    """One table row for a record: identifiers, its task score, and its feature columns (as in build_features)."""
    if not ((row.get("activation_metrics") or {}).get("prompt_time") or {}).get("layers"):
        raise ValueError(f"{dataset}/{key}/{row['setting']}: extraction record has no layer activations")
    return {
        "model_dir": model_dir,
        "dataset": dataset,
        "family": family,
        "base_key": key,
        "setting": str(row["setting"]),
        "variant_index": int(row["variant_index"]),
        "score": row_score(row),
        **feature_columns(row),
    }


def write_json(path: Path, value: object) -> None:
    """Write value as indented JSON, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


# --- Severity targets and calibration ---
def impute_train_test(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Replace non-finite values in both arrays with training-column medians (0 where a column has none)."""
    x_train = np.asarray(x_train, dtype=np.float64)
    x_test = np.asarray(x_test, dtype=np.float64)
    med = np.nanmedian(x_train, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    x_train = np.where(np.isfinite(x_train), x_train, med)
    x_test = np.where(np.isfinite(x_test), x_test, med)
    return x_train, x_test


def severity_by_setting(train_unique: pd.DataFrame, dataset: str) -> dict[str, float]:
    """Per setting, the training-split mean score at BASELINE_SETTING minus the setting's mean score.

    A positive severity means the setting scores lower than the baseline; a setting
    with no finite mean gets NaN.
    """
    means = train_unique.groupby("setting")["score"].mean().to_dict()
    if BASELINE_SETTING not in means or not np.isfinite(means[BASELINE_SETTING]):
        raise RuntimeError(f"Missing finite baseline score for {BASELINE_SETTING}")
    base_score = float(means[BASELINE_SETTING])
    out: dict[str, float] = {}
    for spec in SETTINGS[dataset]:
        setting = str(spec["setting"])
        value = float(means.get(setting, float("nan")))
        out[setting] = base_score - value if np.isfinite(value) else float("nan")
    return out


def fit_behavior_calibrator(
    train_unique: pd.DataFrame,
    all_unique: pd.DataFrame,
    feature_cols: list[str],
    ridge_alpha: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Ridge-fit ELS features and score to training severity; return calibrated scores, fit stats.

    Each prompt's target is its setting's severity. The fit uses training rows only and
    predicts a calibrated ELS score for every row.
    """
    # Inputs are the ELS features plus the raw ELS score, imputed and standardized with
    # training statistics only.
    cal_cols = feature_cols + ["raw_els_score"]
    x_train, x_all = impute_train_test(
        train_unique[cal_cols].astype(float).values,
        all_unique[cal_cols].astype(float).values,
    )
    y_train = train_unique["category_severity"].astype(float).values
    ok = np.isfinite(y_train)
    if ok.sum() < 20:
        raise RuntimeError(f"Only {ok.sum()} finite rows for behavior calibration")
    scaler = StandardScaler().fit(x_train[ok])
    reg = Ridge(alpha=float(ridge_alpha)).fit(scaler.transform(x_train[ok]), y_train[ok])
    pred = reg.predict(scaler.transform(x_all))
    # Fit statistics; train_pred_corr is the in-sample correlation, NaN for a constant target.
    meta = {
        "intercept": float(reg.intercept_),
        "coef_norm": float(np.linalg.norm(reg.coef_)),
        "train_target_mean": float(np.mean(y_train[ok])),
        "train_target_std": float(np.std(y_train[ok])),
        "train_pred_corr": float(np.corrcoef(reg.predict(scaler.transform(x_train[ok])), y_train[ok])[0, 1])
        if np.std(y_train[ok]) > 1e-8
        else float("nan"),
    }
    return pred.astype(float), meta


def build_one(args: argparse.Namespace, dataset: str, model_dir: str) -> dict[str, Any]:
    """Build and save one dataset/model direction, split keys, and ELS scores; return a summary."""
    # Read the extraction run and keep questions recorded under every setting.
    source, els_dir = extraction_paths(args.realworld_root, dataset, model_dir)
    rows = read_jsonl(source)
    index = build_index(rows, dataset)
    keys_all = complete_keys(index, dataset)
    # Examples drawn: training, then held back, then test.
    n_needed = args.n_train + N_HELD_BACK + args.n_test
    if len(keys_all) < n_needed:
        raise RuntimeError(f"{dataset}/{model_dir} has only {len(keys_all)} complete examples; need {n_needed}")

    # Split a seeded shuffle of the complete questions into train, held-back, and test.
    # The shuffle is seeded by dataset, model, and --seed, so each cell has its own split.
    keys = stable_shuffle(keys_all, f"behcal::{dataset}::{model_dir}::{args.seed}")[:n_needed]
    train_keys = keys[: args.n_train]
    held_back_keys = keys[args.n_train : args.n_train + N_HELD_BACK]
    test_keys = keys[args.n_train + N_HELD_BACK :]

    # One feature row per question and setting, after checking every record's feature sources.
    records: list[dict[str, Any]] = []
    for split_name, split_keys in [("train", train_keys), ("held_back", held_back_keys), ("test", test_keys)]:
        require_els_feature_sources(
            [index[key][str(spec["setting"])] for key in split_keys for spec in SETTINGS[dataset]],
            f"{dataset}/{model_dir}/{split_name}",
        )
        for key in split_keys:
            by_setting = index[key]
            for spec in SETTINGS[dataset]:
                rec = feature_record(
                    by_setting[str(spec["setting"])],
                    dataset,
                    model_dir,
                    str(spec["family"]),
                    key,
                )
                rec["split"] = split_name
                records.append(rec)

    # Summarize each per-layer array into early/mid/late/last/pooled feature columns,
    # as the benchmark ELS features are.
    df_raw = pd.DataFrame(records)
    df = aggregated(df_raw).reset_index(drop=True)
    feats = list(ELS_FEATURES)
    missing = [feat for feat in feats if feat not in df.columns]
    if missing:
        raise RuntimeError(f"{dataset}/{model_dir} missing ELS features: {missing}")

    # Train ELS on binary low/high-load pairs from the training split, balanced across
    # load families, and score every row (raw_els_score).
    # A setting in several pairs (e.g. k1) contributes one row per pair. fit_mlp weights
    # rows by their pair's load family so each family carries equal total weight.
    train_records = []
    for key in train_keys:
        by_setting = index[key]
        for pair in ELS_PAIRS[dataset]:
            for setting, label in [(str(pair["low"]), 0.0), (str(pair["high"]), 1.0)]:
                rec = feature_record(by_setting[setting], dataset, model_dir, str(pair["family"]), key)
                rec["load_label"] = label
                rec["pair_family"] = str(pair["family"])
                train_records.append(rec)
    pair_df = aggregated(pd.DataFrame(train_records)).reset_index(drop=True)
    x_pair = pair_df[feats].astype(float).values
    y_pair = pair_df["load_label"].astype(float).values
    balance = pair_df["pair_family"].astype(str).values
    x_all = df[feats].astype(float).values
    raw_els = fit_mlp(
        x_pair,
        y_pair,
        x_all,
        balance_keys=balance,
        seed=int(args.seed),
    )
    df["raw_els_score"] = raw_els.astype(float)

    # Calibrate: regress features and raw score onto each setting's training score drop.
    # Each (setting, variant_index) is one row, as build_index guarantees.
    unique_cols = ["setting", "variant_index", "split", "base_key", "family", "score", "raw_els_score", *feats]
    all_unique = df[unique_cols].copy()
    train_unique = all_unique[all_unique["split"] == "train"].reset_index(drop=True)

    severities = severity_by_setting(train_unique, dataset)
    all_unique["category_severity"] = all_unique["setting"].map(severities).astype(float)
    train_unique["category_severity"] = train_unique["setting"].map(severities).astype(float)
    calibrated_score, calibrator = fit_behavior_calibrator(
        train_unique,
        all_unique,
        feats,
        args.calibration_ridge_alpha,
    )
    all_unique["calibrated_els_score"] = calibrated_score

    # Fit the hidden-state direction on training prompts' final-token states: per layer,
    # a ridge regression from the state to the calibrated score, normalized to unit length.
    train_rows = all_unique[all_unique["split"] == "train"]
    h = np.stack([load_activation(els_dir, str(row["setting"]), int(row["variant_index"]))
                  for _, row in train_rows.iterrows()])
    y = train_rows["calibrated_els_score"].to_numpy(dtype=np.float32)
    directions = fit_ridge_directions(h, y, args.ridge_alpha)
    steering_layers = STEERING_LAYERS[model_dir]
    if directions.shape[0] <= max(steering_layers) or not all(np.linalg.norm(directions[L]) > 0 for L in steering_layers):
        raise RuntimeError(f"{dataset}/{model_dir}: no direction at steering layers {steering_layers}")

    # Save the direction archive read by activations.install_steering_hooks, the split
    # keys, and every row's ELS scores.
    safe = f"{dataset}_{model_dir}"
    npz_path = args.out_dir / "directions" / f"{safe}.npz"
    np.savez(
        npz_path,
        directions=directions,
        steering_layers=np.asarray(steering_layers, dtype=np.int64),
        n_train=args.n_train,
        n_test=len(test_keys),
        ridge_alpha=args.ridge_alpha,
        calibration_ridge_alpha=args.calibration_ridge_alpha,
        setting_specs=np.asarray([json.dumps(spec, sort_keys=True) for spec in SETTINGS[dataset]]),
        els_pairs=np.asarray([json.dumps(pair, sort_keys=True) for pair in ELS_PAIRS[dataset]]),
        severity_by_setting=np.asarray([json.dumps(severities, sort_keys=True)]),
    )

    key_dir = args.out_dir / "splits"
    write_json(key_dir / f"{safe}_train_keys.json", train_keys)
    write_json(key_dir / f"{safe}_test_keys.json", test_keys)
    # Alpha is selected on further examples outside all three draws, taken from the
    # runner's eligible targets rather than from the extraction run.
    candidates = (cuad_candidate_keys(args.cuad_qas) if dataset == "cuad"
                  else qasper_candidate_keys(args.qasper_test))
    alpha_val_keys = alpha_validation_keys(candidates, set(keys), args.n_alpha_val)
    write_json(key_dir / f"{safe}_alpha_val_keys.json", alpha_val_keys)

    scores_path = args.out_dir / "scores" / f"{safe}.jsonl"
    with scores_path.open("w", encoding="utf-8") as handle:
        for _, row in all_unique.iterrows():
            handle.write(json.dumps({
                "dataset": dataset,
                "model_dir": model_dir,
                "split": str(row["split"]),
                "family": str(row["family"]),
                "base_key": str(row["base_key"]),
                "setting": str(row["setting"]),
                "variant_index": int(row["variant_index"]),
                "score": float(row["score"]),
                "raw_els_score": float(row["raw_els_score"]),
                "category_severity": float(row["category_severity"]),
                "calibrated_els_score": float(row["calibrated_els_score"]),
            }) + "\n")

    return {
        "dataset": dataset,
        "model_dir": model_dir,
        "hf_model": HF_MODEL[model_dir],
        "source_jsonl": str(source),
        "els_dir": str(els_dir),
        "npz": str(npz_path),
        "scores_jsonl": str(scores_path),
        "n_complete_available": len(keys_all),
        "steering_layers": steering_layers,
        "severity_by_setting": severities,
        "behavior_calibrator": calibrator,
        "splits": {"train": train_keys, "alpha_val": alpha_val_keys, "test": test_keys},
        "settings": SETTINGS[dataset],
        "els_pairs": ELS_PAIRS[dataset],
    }


# --- Command line ---
def main() -> None:
    """Validate options, build each requested dataset/model cell, and write manifest.json."""
    ap = argparse.ArgumentParser(prog="run.py directions",
                                 description="Build steering directions, split keys, and ELS scores "
                                             "from CUAD/QASPER extraction runs (run.py steer --els_mode).")
    ap.add_argument("--realworld-root", type=Path, required=True,
                    help="Directory holding each runner's outputs under <dataset>/results/")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="New or empty directory for directions/, splits/, scores/, and manifest.json")
    ap.add_argument("--cells", nargs="+", required=True,
                    help="dataset:model_dir pairs to build, e.g. cuad:Qwen-Qwen3-14B qasper:Qwen-Qwen3-32B")
    ap.add_argument("--n-train", type=int, default=150,
                    help="Complete examples used to fit ELS, the calibration, and the direction (default: 150)")
    ap.add_argument("--n-test", type=int, default=300,
                    help="Complete examples held out as the steering test split (default: 300)")
    ap.add_argument("--n-alpha-val", type=int, default=200,
                    help="Examples outside the training, held-back and test splits, used to select alpha (default: 200)")
    ap.add_argument("--cuad-qas", type=Path,
                    help="Prepared cuad_qas_real.jsonl; required for CUAD cells")
    ap.add_argument("--qasper-test", type=Path,
                    help="Official QASPER test JSON; required for QASPER cells")
    ap.add_argument("--ridge-alpha", type=float, default=1.0,
                    help="Ridge penalty for the per-layer hidden-state direction fit (default: 1.0)")
    ap.add_argument("--calibration-ridge-alpha", type=float, default=1.0,
                    help="Ridge penalty for calibrating ELS scores to per-setting score drops on the training split (default: 1.0)")
    ap.add_argument("--seed", type=int, default=20260506,
                    help="Seed for the split shuffle and the ELS fit (default: 20260506)")
    args = ap.parse_args()
    for cell in args.cells:
        dataset, separator, model_dir = cell.partition(":")
        if not separator or dataset not in ID_FIELD or model_dir not in HF_MODEL:
            ap.error(f"Unsupported cell {cell!r}; use cuad or qasper and a model from {list(HF_MODEL)}")

    for name in ('n_train', 'n_test', 'n_alpha_val'):
        if getattr(args, name) <= 0:
            ap.error(f"--{name.replace('_', '-')} must be positive")
    for name in ('ridge_alpha', 'calibration_ridge_alpha'):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            ap.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
    if len(args.cells) != len(set(args.cells)):
        ap.error('--cells must not contain duplicates')
    for dataset, name in (('cuad', 'cuad_qas'), ('qasper', 'qasper_test')):
        path = getattr(args, name)
        if any(cell.startswith(f'{dataset}:') for cell in args.cells):
            if path is None or not path.expanduser().is_file():
                ap.error(f"--{name.replace('_', '-')} must name an existing file for {dataset} cells")
            setattr(args, name, path.expanduser().resolve())
    args.realworld_root = args.realworld_root.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    if not args.realworld_root.is_dir():
        ap.error('--realworld-root must be an existing extraction directory')
    if args.out_dir.exists() and (not args.out_dir.is_dir() or any(args.out_dir.iterdir())):
        ap.error('--out-dir must be new or empty; existing fitted directions are not overwritten')
    # Check every cell's inputs before building any cell.
    for cell in args.cells:
        dataset, model_dir = cell.split(":", 1)
        extraction_paths(args.realworld_root, dataset, model_dir)
    for sub in ["directions", "splits", "scores"]:
        (args.out_dir / sub).mkdir(parents=True, exist_ok=True)

    # Run-wide settings, then one summary per cell from build_one.
    manifest: dict[str, Any] = {
        "description": "ELS trained on low/high-load pairs, linear behavior calibration, ridge hidden-state direction",
        "n_train": args.n_train,
        "n_held_back": N_HELD_BACK,
        "n_test": args.n_test,
        "n_alpha_val": args.n_alpha_val,
        "seed": args.seed,
        "ridge_alpha": args.ridge_alpha,
        "calibration_ridge_alpha": args.calibration_ridge_alpha,
        "steer_mode": "projective",
        "steer_scope": "prefill_except_last_token",
        "layer_policy": "fixed_middle_to_late_layers",
        "cells": {},
    }
    for cell in args.cells:
        dataset, model_dir = cell.split(":", 1)
        print(f"\n=== Building {dataset}/{model_dir} ===")
        info = build_one(args, dataset, model_dir)
        manifest["cells"][f"{dataset}/{model_dir}"] = info
        print(f"  complete={info['n_complete_available']} test={len(info['splits']['test'])}")
        print(f"  npz={info['npz']}")
        print(f"  steering_layers={info['steering_layers']}")
        print(f"  severity_by_setting={json.dumps(info['severity_by_setting'], sort_keys=True)}")
        print(f"  behavior_calibrator={json.dumps(info['behavior_calibrator'], sort_keys=True)}")

    write_json(args.out_dir / "manifest.json", manifest)
    print(f"\nWrote {args.out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
