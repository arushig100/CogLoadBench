"""Leave-one-task-out and held-out-level generalization of ELS (Table 4)."""
# Task panel: each labeled prompt gets an all-task score (five-fold ELS) and a
# held-out-task score (ELS trained on the other two tasks); AUCs per model, task and
# axis are averaged per task. Level panel: on three axes with ordered load levels,
# ELS learns the lowest training level against the higher ones and is tested on
# levels it never saw in training.
# main fits and saves the scores; evaluate checks their coverage and folds and
# recomputes the table from them.
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from els.features import ELS_FEATURES, aggregated, balance_keys, feature_matrix, load_features
from els.paths import GZIP, new_output, parser_with_output, require_inputs
from els.scoring import AXES, TASK_NAMES, fit_mlp, group_folds

# Tasks in the row order of the task panel.
TASKS = ['context', 'math', 'analogy']
# Axes with three or more ordered load levels, used in the level panel.
LEVEL_AXES = ['quantity', 'discontinuity', 'interference']
LEVEL_NAMES = {'quantity': 'Quantity', 'discontinuity': 'Discontinuity', 'interference': 'Interference'}


def task_scores(data):
    """Score each labeled prompt with all-task 5-fold ELS and with ELS trained on the other two tasks."""
    parts = []
    for model in sorted(data.model.dropna().unique()):
        sub = data[(data.model == model) & data.load_label.notna()]
        x, y, balance = feature_matrix(sub, ELS_FEATURES), sub.load_label.astype(int).to_numpy(), balance_keys(sub)
        # All-task folds keep each task's base problem in one fold across axes; training keeps the five axes only.
        groups = (sub.task.astype(str) + '::' + sub.pair_key.astype(str)).to_numpy()
        trainable = sub.axis.isin(AXES).to_numpy()
        in_family, held_out = np.full(len(sub), np.nan), np.full(len(sub), np.nan)
        for train, test in group_folds(len(x), groups):
            train = train[trainable[train]]
            in_family[test] = fit_mlp(x[train], y[train], x[test], balance_keys=balance[train])
        # Held-out task: train on the other two tasks, score every prompt of this one.
        for task in TASKS:
            train = np.flatnonzero((sub.task.to_numpy() != task) & trainable)
            test = np.flatnonzero(sub.task.to_numpy() == task)
            held_out[test] = fit_mlp(x[train], y[train], x[test], balance_keys=balance[train])
        # Only rows of the five axes are saved and evaluated.
        parts.append(pd.DataFrame(dict(row=sub.index, in_family=in_family.astype(np.float32),
                                       held_out=held_out.astype(np.float32)))[trainable])
        print(f'Task transfer complete: {model}', flush=True)
    return pd.concat(parts, ignore_index=True)


def level_frame(data, axis, model):
    """Rows of one model and axis with an ordinal level."""
    sub = data[(data.axis == axis) & data.level_index.notna() & (data.model == model)]
    return sub.assign(level_index=sub.level_index.astype(int))


def holdout_splits(axis, levels):
    """Return (held-out label, train levels, [(bridge, test levels)]) for each split; level 0 is the baseline.

    Quantity holds out every pair of loaded levels and interference every single loaded
    level; both test the baseline against the held-out levels. Discontinuity has three
    levels and holds out each one in turn; when the middle level is held out, it is
    tested separately against the low and the high level (the two bridges).
    """
    # Each configuration is (train levels, [(bridge, test levels)]).
    base, rest = levels[0], levels[1:]
    if axis == 'quantity':
        configs = [([base, *[l for l in rest if l not in held]], [('baseline', [base, *held])])
                   for held in combinations(rest, 2)]
    elif axis == 'interference':
        configs = [([base, *[l for l in rest if l != held]], [('baseline', [base, held])]) for held in rest]
    else:
        low, mid, high = levels
        configs = [([low, high], [(f'{low}-{mid}', [low, mid]), (f'{mid}-{high}', [mid, high])]),
                   ([mid, high], [('baseline', [low, mid])]),
                   ([low, mid], [('baseline', [mid, high])])]
    # The held-out label lists the tested levels absent from training, e.g. '1,3'.
    splits = []
    for train, tests in configs:
        tested = {l for _, test in tests for l in test}
        splits.append((','.join(str(l) for l in sorted(tested - set(train))), train, tests))
    return splits


def level_scores(data):
    """Score prompts on held-out base problems, both with all levels and with some levels held out."""
    parts = []
    for axis in LEVEL_AXES:
        for model in sorted(data.model.dropna().unique()):
            sub = level_frame(data, axis, model)
            x, level, balance = feature_matrix(sub, ELS_FEATURES), sub.level_index.to_numpy(), balance_keys(sub)
            # Base-problem folds shared by the all-level and held-out-level fits.
            folds = list(group_folds(len(x), sub.pair_key.astype(str).to_numpy()))
            # All levels: label every loaded level against the baseline.
            y = (level != level.min()).astype(int)
            for fold, (train, test) in enumerate(folds):
                score = fit_mlp(x[train], y[train], x[test], balance_keys=balance[train])
                parts.append(pd.DataFrame(dict(hold_out='all', bridge='all', fold=fold, row=sub.index[test], score=score)))
            # Held-out levels: train on the training levels only (lowest one is the
            # negative class), then score the test fold's rows at each bridge's levels.
            for held, train_levels, tests in holdout_splits(axis, sorted(set(level))):
                y = (level != min(train_levels)).astype(int)
                for fold, (train, test) in enumerate(folds):
                    train = train[np.isin(level[train], train_levels)]
                    # One fit per fold scores the rows of every bridge.
                    scored = test[np.isin(level[test], [l for _, test_levels in tests for l in test_levels])]
                    score = pd.Series(fit_mlp(x[train], y[train], x[scored], balance_keys=balance[train]), index=scored)
                    for bridge, test_levels in tests:
                        rows = test[np.isin(level[test], test_levels)]
                        parts.append(pd.DataFrame(dict(hold_out=held, bridge=bridge, fold=fold,
                                                       row=sub.index[rows], score=score.loc[rows].to_numpy())))
            print(f'Level transfer complete: {model} {axis}', flush=True)
    return pd.concat(parts, ignore_index=True)


def level_cells(data, scores):
    """ROC AUC per task: pooled over folds for all levels, per fold for held-out levels."""
    rows = []
    frame = scores.join(data[['axis', 'model', 'task', 'level_index']], on='row')
    for (axis, model, hold_out, bridge), part in frame.groupby(['axis', 'model', 'hold_out', 'bridge'], sort=True):
        # Within each scored set, the lowest level present is the negative class.
        level = part.level_index.astype(int)
        part = part.assign(label=(level != level.min()).astype(int))
        # All-level cells pool every fold (reported with fold -1); held-out cells are per fold.
        keys = 'task' if hold_out == 'all' else ['fold', 'task']
        for _, cell in part.groupby(keys, sort=True):
            # A cell needs five prompts and both labels for a defined AUC.
            if len(cell) >= 5 and cell.label.nunique() == 2:
                rows.append(dict(axis=axis, model=model, hold_out=hold_out, bridge=bridge, task=cell.task.iloc[0],
                                 fold=-1 if hold_out == 'all' else cell.fold.iloc[0],
                                 auc=roc_auc_score(cell.label, cell.score)))
    return pd.DataFrame(rows)


def task_cells(data, scores):
    """ROC AUC per model, task and axis for all-task and held-out-task scores."""
    frame = scores.join(data[['model', 'task', 'axis', 'load_label']], on='row')
    rows = []
    for (model, task, axis), part in frame.groupby(['model', 'task', 'axis'], sort=True):
        label = part.load_label.astype(int)
        rows.append(dict(model=model, task=task, axis=axis, n=len(part),
                         in_family=roc_auc_score(label, part.in_family), held_out=roc_auc_score(label, part.held_out)))
    return pd.DataFrame(rows)


def summarize(tasks, levels):
    """Average cell AUCs into the two panels of Table 4, in percent.

    Columns: 'all' is the all-task (or all-level) score and 'held_out' the held-out task
    (or held-out level) score.
    """
    rows = [dict(panel='task', name=TASK_NAMES[t], all=100 * tasks[tasks.task == t].in_family.mean(),
                 held_out=100 * tasks[tasks.task == t].held_out.mean()) for t in TASKS]
    for axis in LEVEL_AXES:
        part = levels[levels.axis == axis]
        rows.append(dict(panel='level', name=LEVEL_NAMES[axis], all=100 * part[part.hold_out == 'all'].auc.mean(),
                         held_out=100 * part[part.hold_out != 'all'].auc.mean()))
    return pd.DataFrame(rows).set_index(['panel', 'name'])


def read_scores(output):
    """Read saved task and level scores at their saved float32 precision."""
    tasks = pd.read_csv(output / 'task_scores.csv.gz')
    levels = pd.read_csv(output / 'level_scores.csv.gz', dtype={'hold_out': str})
    tasks[['in_family', 'held_out']] = tasks[['in_family', 'held_out']].astype(np.float32)
    levels['score'] = levels.score.astype(np.float32)
    return tasks, levels


def evaluate(output, data):
    """Recompute cell AUCs and the summary from saved scores, after checking rows and folds."""
    # Task scores must cover exactly the labeled five-axis rows.
    tasks, levels = read_scores(output)
    expected = data.index[data.load_label.notna() & data.axis.isin(AXES)]
    if (tasks.row.duplicated().any() or set(tasks.row) != set(expected)
            or not np.isfinite(tasks[['in_family', 'held_out']].to_numpy()).all()):
        raise ValueError('Task-transfer scores do not cover the labeled prompts')
    # Rebuild each model and axis's level folds and require the saved level scores to
    # cover exactly the expected rows for every split, with matching fold numbers.
    checked = 0
    for axis in LEVEL_AXES:
        for model in sorted(data.model.dropna().unique()):
            sub = level_frame(data, axis, model)
            fold = pd.Series(-1, index=sub.index)
            for number, (_, test) in enumerate(group_folds(len(sub), sub.pair_key.astype(str).to_numpy())):
                fold.iloc[test] = number
            splits = [('all', 'all', None)] + [(held, bridge, test) for held, _, tests in
                                               holdout_splits(axis, sorted(set(sub.level_index))) for bridge, test in tests]
            for held, bridge, test_levels in splits:
                part = levels[(levels.hold_out == held) & (levels.bridge == bridge) & levels.row.isin(sub.index)]
                wanted = sub.index if test_levels is None else sub.index[sub.level_index.isin(test_levels)]
                if (part.row.duplicated().any() or set(part.row) != set(wanted)
                        or (fold.loc[part.row].to_numpy() != part.fold.to_numpy()).any()
                        or not np.isfinite(part.score).all()):
                    raise ValueError(f'{model}/{axis}/{held}/{bridge}: incomplete level scores or mismatched folds')
                checked += len(part)
    # Every saved level score must belong to one of the expected splits.
    if checked != len(levels):
        raise ValueError('Level scores contain rows outside the expected splits')
    task_detail, level_detail = task_cells(data, tasks), level_cells(data, levels)
    return task_detail, level_detail, summarize(task_detail, level_detail)


def main():
    """Fit and save task and level scores, then write the cell tables and the summary."""
    parser = parser_with_output(__doc__)
    args = parser.parse_args()
    require_inputs(parser, args.features)
    output = new_output(args.output_dir)
    torch.set_num_threads(1)
    data = aggregated(load_features(args.features))
    task_scores(data).to_csv(output / 'task_scores.csv.gz', index=False, compression=GZIP)
    level_scores(data).to_csv(output / 'level_scores.csv.gz', index=False, compression=GZIP)
    tasks, levels, summary = evaluate(output, data)
    tasks.to_csv(output / 'task_cells.csv', index=False)
    levels.to_csv(output / 'level_cells.csv', index=False)
    summary.to_csv(output / 'summary.csv')
    print(summary.round(1).to_string(), flush=True)


if __name__ == '__main__':
    main()
