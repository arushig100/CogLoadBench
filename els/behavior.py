"""Behavior-calibrated ELS: rank load conditions by their accuracy drop (Table D2)."""
# Steps: (1) score every prompt with out-of-fold ELS; (2) split base problems into a
# calibration half and a held-out half; (3) for each condition of the four graded
# axes, compute its accuracy drop (severity) and ELS change relative to the axis
# baseline in each half; (4) predict each condition's severity from held-out prompts,
# either with the raw ELS change or with a ridge head fitted on calibration prompts;
# (5) score each prediction by how often it orders pairs of conditions as their
# calibration-half severities do.
# main fits and saves the ELS scores; evaluate recomputes everything else from them.
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from els.features import ELS_FEATURES, aggregated, balance_keys, feature_matrix, load_features
from els.scoring import AXES, TASK_NAMES, fit_mlp, group_folds, impute
from els.paths import GZIP, new_output, parser_with_output, require_inputs

# Tasks in the row order of the summary table, and the baseline condition of each graded axis.
TASKS = ['math', 'context', 'analogy']
BASELINES = {'quantity': 'k1', 'similarity': 'low', 'interference': 'none', 'discontinuity': 'ordered'}
# Seed of the calibration/held-out split of base problems.
SPLIT_SEED = 0
# Training seed of the fit on all labeled pairs that scores prompts outside every fold;
# the fold fits use their fold numbers as seeds.
ALL_PAIRS_SEED = 99
# Minimum prompts in a condition cell (and in its axis baseline), conditions per split
# for a task and model, and prompts per split for fitting the ridge head.
MIN_CELL_PROMPTS = 5
MIN_CONDITIONS = 4
MIN_HEAD_PROMPTS = 20


def categories(data):
    """Label prompts of the four graded axes with axis:condition and whether it is the axis baseline."""
    graded = data.axis.isin(BASELINES).to_numpy()
    category = np.where(graded, (data.axis + ':' + data.condition).to_numpy(dtype=object), None)
    baseline = graded & (data.condition == data.axis.map(BASELINES)).to_numpy()
    return data.assign(category=category, is_baseline=baseline)


def els_scores(data):
    """Score every prompt with ELS trained on labeled pairs from other base problems (task::pair_key folds)."""
    # One score per feature-table row (every prompt, labeled or not), indexed like data.
    scores = pd.Series(np.nan, index=data.index)
    for model in sorted(data.model.dropna().unique()):
        pairs = data[(data.model == model) & data.axis.isin(AXES) & data.load_label.notna()]
        prompts = data[data.model == model]
        x, x_all = feature_matrix(pairs, ELS_FEATURES), feature_matrix(prompts, ELS_FEATURES)
        y = pairs.load_label.astype(int).to_numpy()
        balance = balance_keys(pairs)
        # Folds group labeled pairs by task::pair_key. Each fold's model scores every
        # prompt (labeled or not) whose base problem is in that fold's test set.
        groups = (pairs.task.astype(str) + '::' + pairs.pair_key.astype(str)).to_numpy()
        prompt_groups = (prompts.task.astype(str) + '::' + prompts.pair_key.astype(str)).to_numpy()
        scored = np.zeros(len(prompts), dtype=bool)
        for fold, (train, test) in enumerate(group_folds(len(x), groups)):
            held = np.isin(prompt_groups, groups[test])
            scores[prompts.index[held]] = fit_mlp(x[train], y[train], x_all[held], balance_keys=balance[train], seed=fold)
            scored |= held
        # Prompts whose base problem has no labeled pair use a model fit on all pairs.
        if not scored.all():
            scores[prompts.index[~scored]] = fit_mlp(x, y, x_all[~scored], balance_keys=balance, seed=ALL_PAIRS_SEED)
        print(f'ELS scores complete: {model}', flush=True)
    return scores


def assign_split(data):
    """Split base problems 50/50 into calibration and held-out sets within each task and model (all axes)."""
    # One generator across (task, model) groups in sorted order; every prompt starts held out.
    rng, split = np.random.default_rng(SPLIT_SEED), pd.Series('heldout', index=data.index)
    for _, group in data.groupby(['task', 'model'], sort=True):
        # Shuffle the base problems and put about half (at least one, and leaving at
        # least one) in the calibration set.
        keys = np.array(sorted(group.pair_key.astype(str).unique()))
        rng.shuffle(keys)
        calibration = set(keys[:max(1, min(len(keys) - 1, int(round(len(keys) / 2))))])
        split[group.index[group.pair_key.astype(str).isin(calibration)]] = 'calibration'
    return split


def condition_cells(data):
    """Accuracy drop (severity) and mean ELS change of each condition relative to its axis baseline."""
    rows = []
    for (split, task, model, axis), group in data[data.category.notna()].groupby(
            ['split', 'task', 'model', 'axis'], sort=True):
        # Cells with fewer than MIN_CELL_PROMPTS prompts (baseline or condition) are skipped.
        base = group[group.is_baseline]
        if len(base) < MIN_CELL_PROMPTS:
            continue
        # Severity is positive when the condition is less accurate than the baseline; the
        # baseline itself is included with severity 0.
        for category, cell in group.groupby('category', sort=True):
            if len(cell) >= MIN_CELL_PROMPTS:
                rows.append(dict(split=split, task=task, model=model, axis=axis, category=category,
                                 severity=base.correct.mean() - cell.correct.mean(),
                                 delta_els=cell.els.mean() - base.els.mean(), n=len(cell)))
    return pd.DataFrame(rows)


def pair_rank_accuracy(pred, target):
    """Fraction of condition pairs with distinct targets that the prediction orders correctly (ties count half)."""
    good = total = 0
    for i, j in combinations(range(len(pred)), 2):
        if np.isfinite([pred[i], pred[j], target[i], target[j]]).all() and target[i] != target[j]:
            # Positive product: same order; zero: tied prediction; negative: reversed.
            product = (pred[i] - pred[j]) * (target[i] - target[j])
            total += 1
            good += 1.0 if product > 0 else 0.5 if product == 0 else 0.0
    return (good / total if total else np.nan), total


def predictions(data, cells):
    """Held-out condition predictions from raw ELS changes and from a ridge head on the 30 features."""
    rows = []
    for (task, model), held in cells[cells.split == 'heldout'].groupby(['task', 'model'], sort=True):
        calibration = cells[(cells.split == 'calibration') & (cells.task == task) & (cells.model == model)]
        if len(calibration) < MIN_CONDITIONS or len(held) < MIN_CONDITIONS:
            continue
        # Each held-out condition's target is its severity on the calibration half;
        # the raw method predicts it with the condition's held-out ELS change.
        severity = calibration.set_index('category').severity
        methods = [('raw', held.set_index('category').delta_els)]
        prompts = data[(data.task == task) & (data.model == model)]
        train = prompts[(prompts.split == 'calibration') & prompts.category.isin(severity.index)]
        test = prompts[(prompts.split == 'heldout') & prompts.category.isin(held.category)]
        # The calibrated method fits a ridge regression from the 30 features of
        # calibration prompts to their condition's calibration severity, then averages
        # its predictions over the held-out prompts of each condition.
        if len(train) >= MIN_HEAD_PROMPTS and len(test) >= MIN_HEAD_PROMPTS:
            x_train, medians = impute(feature_matrix(train, ELS_FEATURES))
            x_test, _ = impute(feature_matrix(test, ELS_FEATURES), medians)
            scaler = StandardScaler().fit(x_train)
            head = Ridge(alpha=1.0).fit(scaler.transform(x_train), train.category.map(severity).to_numpy(dtype=float))
            methods.append(('calibrated', pd.Series(head.predict(scaler.transform(x_test)), index=test.index)
                            .groupby(test.category).mean()))
        for method, values in methods:
            rows.append(pd.DataFrame(dict(task=task, model=model, method=method, category=held.category.to_numpy(),
                                          pred=values.reindex(held.category).to_numpy(),
                                          target=held.category.map(severity).to_numpy())))
    return pd.concat(rows, ignore_index=True)


def summarize(predicted):
    """Pair-rank accuracy per task, model and method, then mean over models and tasks, in percent."""
    rows = []
    for (task, model, method), group in predicted.groupby(['task', 'model', 'method'], sort=True):
        accuracy, pairs = pair_rank_accuracy(group.pred.to_numpy(), group.target.to_numpy())
        rows.append(dict(task=task, model=model, method=method, pair_rank_accuracy=accuracy, pairs=pairs))
    detail = pd.DataFrame(rows)
    table = 100 * detail.groupby(['task', 'method']).pair_rank_accuracy.mean().unstack()[['raw', 'calibrated']]
    # Rows are tasks plus their mean; improvement is calibrated minus raw.
    table = table.reindex(TASKS).rename(index=TASK_NAMES)
    table.loc['Mean'] = table.mean()
    table['improvement'] = table.calibrated - table.raw
    table.index.name = 'task'
    return detail, table


def analyze(data, scores):
    """Condition cells, held-out predictions and pair-rank accuracy from per-prompt ELS scores."""
    data = categories(data).assign(els=scores)
    data['split'] = assign_split(data)
    cells = condition_cells(data)
    predicted = predictions(data, cells)
    return cells, predicted, *summarize(predicted)


def evaluate(output, data):
    """Recompute the calibration and ranking from saved ELS scores."""
    # Scores are saved as float32, so evaluate at that precision for fresh and saved runs alike.
    saved = pd.read_csv(output / 'els_scores.csv.gz')
    if (saved.row.duplicated().any() or set(saved.row) != set(data.index)
            or not np.isfinite(saved.els).all() or not saved.els.between(0, 1).all()):
        raise ValueError('Saved ELS scores do not cover the feature table')
    return analyze(data, saved.set_index('row').els.astype(np.float32).astype(float).reindex(data.index))


def main():
    """Fit and save per-prompt ELS scores, then write the calibration tables."""
    parser = parser_with_output(__doc__)
    args = parser.parse_args()
    require_inputs(parser, args.features)
    output = new_output(args.output_dir)
    torch.set_num_threads(1)
    data = aggregated(load_features(args.features))
    scores = els_scores(data)
    pd.DataFrame(dict(row=data.index, els=scores.to_numpy(dtype=np.float32))).to_csv(output / 'els_scores.csv.gz', index=False, compression=GZIP)
    cells, predicted, detail, table = evaluate(output, data)
    cells.to_csv(output / 'cells.csv', index=False)
    predicted.to_csv(output / 'predictions.csv', index=False)
    detail.to_csv(output / 'per_model.csv', index=False)
    table.to_csv(output / 'summary.csv')
    print(table.round(1).to_string(), flush=True)


if __name__ == '__main__':
    main()
