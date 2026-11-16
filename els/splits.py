"""Row groups, the fixed calibration test split, and calibration pair orders.

els.fit and els.calibrate build their splits with these functions, and
els.validation rebuilds the same splits to check the saved outputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Calibration pair counts, pair-sampling seeds, and the seed of the fixed test split.
KS = [0, 4, 8, 16, 32, 64, 128]
SEEDS = [0, 1, 2]
SPLIT_SEED = 20261003


def identity_groups(frame: pd.DataFrame) -> np.ndarray:
    """Merge rows sharing a task/base label or a nonmissing task/target ID.

    Union-find over task-scoped keys: rows with the same base label are one group,
    and rows linked through a shared target ID join the same group, so prompts from
    one problem cannot be split across folds. Returns one group key per row.
    """
    # Union-find forest: parent maps each key to its parent; a group's root maps to itself.
    parent = {}

    def root(key):
        """Return the representative key of key's group, compressing the path."""
        parent.setdefault(key, key)
        # Path halving: point each visited key at its grandparent while walking up.
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    bases = []
    for row in frame.itertuples():
        if pd.isna(row.pair_key) or not str(row.pair_key).strip():
            raise ValueError('Cannot isolate a row without a base label')
        base = f'{row.task}::base::{row.pair_key}'
        bases.append(base)
        root(base)
        if pd.notna(row.target_id):
            target = f'{row.task}::target::{row.target_id}'
            # Link the base and target groups under the smaller key.
            a, b = root(base), root(target)
            parent[max(a, b)] = min(a, b)
    return np.array([root(base) for base in bases])


def fixed_test_groups(frame):
    """Select 20% of base groups per task as test groups, identically for every model."""
    # One generator over tasks in sorted order and groups in sorted order, so the
    # selection depends only on SPLIT_SEED and the set of groups.
    rng = np.random.default_rng(SPLIT_SEED)
    selected = set()
    for _, task in frame.groupby('task', sort=True):
        groups = np.array(sorted(task.group.unique()))
        # 20% of the task's groups, rounded up, and at least one.
        selected.update(rng.permutation(groups)[:max(1, int(np.ceil(.2 * len(groups))))])
    return selected


def complete_pairs(held):
    """Low/high pairs sharing task and variant; low and high are row index labels of held.

    A (task, variant) cell without exactly one low and one high row is skipped; a pair
    whose two rows differ in group or base label is an error.
    """
    # Within one axis, a (task, variant) cell holds the low-load and high-load prompts
    # built from the same variant.
    pairs = []
    for (task, variant), rows in held.groupby(['task', 'variant'], sort=True):
        if len(rows) != 2 or set(rows.load_label) != {0, 1}:
            continue
        if rows.group.nunique() != 1 or rows.pair_key.nunique() != 1:
            raise ValueError(f'Unmatched calibration pair: {task}/{variant}')
        pairs.append(dict(task=task, variant=int(variant), group=rows.group.iloc[0],
                          low=int(rows[rows.load_label == 0].index[0]),
                          high=int(rows[rows.load_label == 1].index[0])))
    return pairs


def pair_order(pairs, seed):
    """Sample one pair per base group, interleaving shuffled task queues."""
    rng = np.random.default_rng(seed)
    # Pick one candidate pair per (task, base group).
    by_group = {}
    for pair in pairs:
        by_group.setdefault((pair['task'], pair['group']), []).append(pair)
    by_task = {}
    for (task, _), choices in sorted(by_group.items()):
        by_task.setdefault(task, []).append(choices[int(rng.integers(len(choices)))])
    # Shuffle each task's pairs, then take one pair per task in turn (in a shuffled
    # task order) so every prefix of the result is balanced across tasks until a task's
    # pairs run out.
    queues = {task: [items[i] for i in rng.permutation(len(items))] for task, items in by_task.items()}
    tasks = list(rng.permutation(sorted(queues)))
    ordered = []
    while any(queues.values()):
        for task in tasks:
            if queues[task]:
                ordered.append(queues[task].pop())
    return ordered
