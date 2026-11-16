"""Activation loading, split-key, and ridge-direction helpers for build_directions.py."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch


def load_pt(p: Path):
    """Load a saved activation payload onto the CPU."""
    return torch.load(p, map_location="cpu", weights_only=True)


def stack_layers(payload) -> np.ndarray:
    """Stack per-layer query_last vectors into a [n_layers, hidden_dim] array."""
    # payload["layers"] maps 0-based decoder layer index -> {"query_last": [hidden] tensor};
    # every index up to the largest must be present.
    L = payload.get("layers", {})
    if not L:
        raise ValueError("Activation file has no layers")
    n = max(L.keys()) + 1
    try:
        return torch.stack([L[i]["query_last"] for i in range(n)]).float().numpy()
    except KeyError as error:
        raise ValueError(f"Activation file lacks query_last for layer {error}") from error


def stable_shuffle(items: list[str], seed_text: str) -> list[str]:
    """Return a shuffled copy of items, seeded from an MD5 of seed_text so it is the same on every run."""
    # The first 8 hex digits of the digest give a 32-bit seed.
    seed = int(hashlib.md5(seed_text.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(items))
    return [items[int(i)] for i in order]


def cuad_candidate_keys(qas_path: Path) -> list[str]:
    """CUAD steering targets in file order: answerable, with exactly one gold chunk."""
    keys: list[str] = []
    with Path(qas_path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("is_impossible") or len(row.get("gold_chunks_global") or []) != 1:
                continue
            if row.get("qa_id") is not None:
                keys.append(str(row["qa_id"]))
    return keys


def qasper_candidate_keys(test_path: Path) -> list[str]:
    """QASPER question IDs whose most common answer type is extractive, sorted.

    Answers are typed as in run_qasper.get_qa_info, including the MD5 fallback for a
    missing question_id, so the IDs match the runner's question_id field.
    """
    with Path(test_path).open(encoding="utf-8") as handle:
        papers = json.load(handle)
    keys: list[str] = []
    for paper in papers.values():
        for qa in paper["qas"]:
            types = []
            for annotation in qa["answers"]:
                answer = annotation["answer"]
                if answer.get("unanswerable", False):
                    types.append("unanswerable")
                elif answer.get("extractive_spans"):
                    types.append("extractive")
                elif answer.get("yes_no") is not None:
                    types.append("yes_no")
                elif answer.get("free_form_answer", "").strip():
                    types.append("free_form")
            if types and Counter(types).most_common(1)[0][0] == "extractive":
                keys.append(str(qa.get("question_id", hashlib.md5(qa["question"].encode()).hexdigest())))
    return sorted(keys)


def alpha_validation_keys(candidates: list[str], excluded: set[str], n: int) -> list[str]:
    """First n candidates outside the direction splits, used to select alpha.

    candidates come in file order for CUAD and sorted order for QASPER (see the
    *_candidate_keys helpers above); excluded holds every key drawn for the splits.
    """
    if len(set(candidates)) != len(candidates):
        raise ValueError("Duplicate candidate keys")
    selected = [key for key in candidates if key not in excluded][:n]
    if len(selected) != n:
        raise RuntimeError(f"Only {len(selected)} candidate keys remain outside the splits; need {n}")
    return selected


def extraction_paths(root: Path, dataset: str, model_dir: str) -> tuple[Path, Path]:
    """Return <root>/<dataset>/results/<model>/<dataset>_predictions_<model>.jsonl and results/els_vectors/<model>, raising if either is missing."""
    # model_dir must be one path component (e.g. Qwen-Qwen3-14B), never a path.
    if (dataset not in {"cuad", "qasper"} or not model_dir or model_dir in {".", ".."}
            or Path(model_dir).name != model_dir):
        raise ValueError("Expected a supported dataset and a single model directory name")
    results = (root / dataset / "results").resolve()
    source = results / model_dir / f"{dataset}_predictions_{model_dir}.jsonl"
    vectors = results / "els_vectors" / model_dir
    if not source.is_file():
        raise FileNotFoundError(f"Missing extraction records: {source}")
    if not vectors.is_dir():
        raise FileNotFoundError(f"Missing extraction vectors: {vectors}")
    # Resolving symlinks must not lead outside results/.
    if not all(path.resolve().is_relative_to(results) for path in (source, vectors)):
        raise ValueError("Extraction records and vectors must stay within the same results directory")
    return source, vectors


def load_activation(els_dir: Path, setting: str, variant_index: int) -> np.ndarray:
    """Load one prompt's saved [layers, hidden] final-token states."""
    # File name written by runtime.save_vectors, e.g. els__k50__v0007.pt.
    path = els_dir / f"els__{setting}__v{int(variant_index):04d}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing activation file: {path}")
    try:
        return stack_layers(load_pt(path))
    except Exception as error:
        raise ValueError(f"Unreadable activation file {path}: {error}") from error


def fit_ridge_directions(h: np.ndarray, y: np.ndarray, ridge_alpha: float) -> np.ndarray:
    """Fit a ridge direction per layer; return unit directions, zero where a layer has non-finite states.

    h is [n_prompts, n_layers, hidden] and y the per-prompt target. Each layer solves the
    ridge problem in its dual form, w = X^T (X X^T + alpha I)^-1 y on centered X and y,
    which is cheaper than the primal form when prompts are fewer than hidden units.
    Each direction points toward higher y.
    """
    y = y.astype(np.float32)
    y_c = y - float(np.nanmean(y))
    n, n_layers, hidden_dim = h.shape
    directions = np.zeros((n_layers, hidden_dim), dtype=np.float32)
    eye_n = np.eye(n, dtype=np.float32)

    for layer in range(n_layers):
        x = h[:, layer, :].astype(np.float32)
        if not np.isfinite(x).all():
            continue
        # x_c is [n, hidden], gram [n, n], coef [n], and w [hidden].
        x_c = x - x.mean(axis=0, keepdims=True)
        gram = x_c @ x_c.T
        try:
            coef = np.linalg.solve(gram + ridge_alpha * eye_n, y_c)
            w = x_c.T @ coef
        except np.linalg.LinAlgError:
            # A singular system falls back to the covariance direction X^T y.
            w = x_c.T @ y_c
        nrm = float(np.linalg.norm(w))
        if nrm > 1e-8:
            directions[layer] = (w / nrm).astype(np.float32)
    return directions
