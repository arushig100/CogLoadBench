"""Validate saved ELS identities, coverage, and predictions before aggregation.

The cross-axis (els.fit) and calibration (els.calibrate) outputs are large sets of
saved predictions. The functions here check the saved row identities against the
prompt rows, require every saved test fold to share no base problem with the rows
it was trained on, rebuild the calibration splits and pair orders, and recompute every
AUC and summary from the saved scores. They also hold the score cache used by both
experiments and the summary and rounding helpers shared with els.verify.
"""
from io import StringIO
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from els.scoring import AXES, N_FOLDS, TIERS
from els.splits import KS, SEEDS, complete_pairs, fixed_test_groups, identity_groups, pair_order

# Row-identity columns saved with the cross-axis predictions.
IDENTITY_COLUMNS = ['model', 'task', 'axis', 'pair_key', 'target_id', 'load_label']
# Feature-table columns needed to rebuild both experiments' splits.
SOURCE_COLUMNS = IDENTITY_COLUMNS + ['variant', 'condition']
# Per-row metadata saved by els.calibrate (rows_<model>.csv).
CALIBRATION_COLUMNS = ['model', 'task', 'axis', 'pair_key', 'target_id', 'variant', 'condition', 'load_label',
                       'group', 'is_test']


def rounded_as_listed(values, digits):
    """Round values to the listed digits (halves round up); digits is one int or one per value."""
    values = list(values)
    digits = [digits] * len(values) if isinstance(digits, int) else digits
    # The small offset makes exact halves such as 0.25 round up instead of to even.
    return tuple(round(float(v) + 1e-12, d) for v, d in zip(values, digits))


def cross_axis_summary(detail):
    """Mean AUC per scorer and axis over models, with the macro mean over axes."""
    summary = detail.groupby(['tier', 'axis']).auc.mean().unstack().reindex(columns=AXES)
    summary['macro'] = summary.mean(axis=1)
    return summary


def calibration_summary(detail):
    """Mean AUC per held-out axis and K over models and seeds, with in-distribution AUC and the macro row."""
    summary = detail.groupby(['axis', 'k_pairs']).auc.mean().unstack().reindex(AXES)
    summary['in_distribution'] = detail.groupby('axis').in_distribution_auc.mean()
    summary.loc['macro'] = summary.mean()
    return summary


def require_grid(frame, columns, levels, label):
    """Require each combination of levels exactly once in frame[columns], with no missing keys."""
    if frame[columns].isna().any().any() or frame.duplicated(columns).any():
        raise ValueError(f'{label}: missing or duplicate coordinates')
    if set(frame[columns].itertuples(index=False, name=None)) != set(product(*levels)):
        raise ValueError(f'{label}: incomplete or unexpected coordinates')


def require_disjoint(frame, train, test, groups):
    """Require disjoint groups and task-scoped base labels and target IDs between two row sets."""
    # Groups already merge linked rows; the base-label and target-ID checks confirm
    # the split directly.
    if set(groups[train]) & set(groups[test]):
        raise ValueError('training and test rows share a group')

    def identities(indices, column):
        """(task, value) pairs of a column for the given row positions, ignoring missing values."""
        rows = frame.iloc[indices].dropna(subset=[column])
        return set(zip(rows.task.astype(str), rows[column].astype(str)))

    for column in ['pair_key', 'target_id']:
        if identities(train, column) & identities(test, column):
            raise ValueError(f'training and test rows share a {column}')


def require_scores(scores, count, label):
    """Require count finite scores in [0, 1]."""
    scores = np.asarray(scores)
    if (scores.shape != (count,) or not np.isfinite(scores).all()
            or not ((scores >= 0) & (scores <= 1)).all()):
        raise ValueError(f'{label}: expected {count} finite scores in [0, 1]')


def cached_scores(path, count, label, compute):
    """Load saved scores, or compute and save them; require count scores in [0, 1]."""
    if path.exists():
        scores = np.load(path, allow_pickle=False)
    else:
        # Write to a temporary file and rename it, so an interrupted run never leaves a
        # truncated cache file.
        scores = compute()
        temporary = path.with_suffix('.partial')
        with temporary.open('wb') as handle:
            np.save(handle, scores, allow_pickle=False)
        temporary.replace(path)
    require_scores(scores, count, label)
    return scores


def require_close(actual, expected, label, atol=1e-12):
    """Require equal shapes and values within atol; missing values must be missing in both."""
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=0, atol=atol, equal_nan=True):
        raise ValueError(f'{label}: values differ by more than {atol:g}')


def require_rows(actual, expected, label):
    """Compare saved metadata with the writer's CSV representation of its source."""
    # Round-trip the expected rows through CSV so dtypes and float text match what was written.
    expected = pd.read_csv(StringIO(expected.to_csv(index=False)))
    if not actual.columns.is_unique or set(actual.columns) != set(expected.columns):
        raise ValueError(f'{label}: metadata columns differ from source')
    try:
        pd.testing.assert_frame_equal(actual[expected.columns].reset_index(drop=True),
                                      expected.reset_index(drop=True), check_dtype=False, check_exact=True)
    except AssertionError as exc:
        raise ValueError(f'{label}: metadata differs from source rows') from exc


def require_summary(actual, expected, label, *, ordered=False, atol=1e-12):
    """Require the same summary coordinates and values within atol.

    With ordered, rows and columns must also come in the same order and missing cells
    must match; otherwise any order is accepted and every value must be finite.
    """
    # Ordered comparison suits tables with fixed row order and intentionally empty cells.
    if ordered:
        if (list(actual.index) != list(expected.index)
                or list(map(str, actual.columns)) != list(map(str, expected.columns))):
            raise ValueError(f'{label}: unexpected summary coordinates')
    else:
        if (not actual.index.is_unique or not actual.columns.is_unique
                or set(actual.index) != set(expected.index)
                or set(actual.columns) != set(expected.columns)):
            raise ValueError(f'{label}: incomplete, duplicate, or unexpected summary coordinates')
        actual = actual.loc[expected.index, expected.columns]
        if not np.isfinite(actual.to_numpy(dtype=float)).all() or not np.isfinite(expected.to_numpy(dtype=float)).all():
            raise ValueError(f'{label}: nonfinite summary value')
    require_close(actual.to_numpy(dtype=float), expected.to_numpy(dtype=float), label, atol)


def validate_cross_axis(output, source):
    """Check cross-axis predictions, folds and AUCs against source rows; return the metrics."""
    # Expect exactly one metrics row per (model, scorer, axis) and one prediction file per model.
    models = sorted(source.model.dropna().unique())
    detail = pd.read_csv(output / 'per_model.csv')
    require_grid(detail, ['model', 'tier', 'axis'], [models, TIERS, AXES], 'cross-axis metrics')
    files = {f'oof_{model}.csv' for model in models}
    if {p.name for p in output.glob('oof_*.csv')} != files:
        raise ValueError('Cross-axis prediction files do not match source models')
    for model in models:
        # The saved rows are the model's labeled rows in feature-table order, and each
        # saved test fold shares no group, base label or target ID with the other folds.
        rows = source[(source.model == model) & source.load_label.notna()].reset_index(drop=True)
        saved = pd.read_csv(output / f'oof_{model}.csv')
        require_rows(saved[IDENTITY_COLUMNS], rows[IDENTITY_COLUMNS], f'{model} cross-axis rows')
        folds, groups = saved.fold.to_numpy(), saved.group.astype(str).to_numpy()
        if sorted(set(folds)) != list(range(N_FOLDS)):
            raise ValueError(f'{model}: expected test folds 0 to {N_FOLDS - 1}')
        for fold in range(N_FOLDS):
            require_disjoint(rows, np.flatnonzero(folds != fold), np.flatnonzero(folds == fold), groups)
        # Each saved AUC and count must follow from the saved out-of-fold scores.
        for tier in TIERS:
            require_scores(saved[tier].to_numpy(), len(rows), f'{model}/{tier}')
            for axis in AXES:
                mask = rows.axis == axis
                record = detail[(detail.model == model) & (detail.axis == axis) & (detail.tier == tier)].iloc[0]
                if record.n != int(mask.sum()):
                    raise ValueError(f'{model}/{axis}/{tier}: incorrect prediction count')
                expected_auc = roc_auc_score(rows.loc[mask, 'load_label'].astype(int), saved.loc[mask, tier])
                require_close(record.auc, expected_auc, f'{model}/{axis}/{tier} AUC')
    return detail


def validate_calibration(output, source):
    """Check calibration splits, pair orders, predictions and AUCs; return the metrics."""
    # Rebuild the fixed test split exactly as els.calibrate does.
    data = source[source.axis.isin(AXES) & source.load_label.notna()].copy()
    data['group'] = identity_groups(data)
    data['is_test'] = data.group.isin(fixed_test_groups(data))
    models = sorted(data.model.unique())
    detail = pd.read_csv(output / 'all_metrics.csv')
    require_grid(detail, ['model', 'axis', 'seed', 'k_pairs'], [models, AXES, SEEDS, KS], 'calibration metrics')
    if {p.name for p in output.glob('rows_*.csv')} != {f'rows_{model}.csv' for model in models}:
        raise ValueError('Calibration metadata files do not match source models')
    for model in models:
        frame = data[data.model == model].reset_index(drop=True)[CALIBRATION_COLUMNS]
        require_rows(pd.read_csv(output / f'rows_{model}.csv'), frame, f'{model} calibration rows')
        require_disjoint(frame, np.flatnonzero(~frame.is_test), np.flatnonzero(frame.is_test), frame.group.to_numpy())
        for axis in AXES:
            # Test rows and the in-distribution reference for this held-out axis.
            folder = output / model / axis
            test = np.flatnonzero(frame.is_test & (frame.axis == axis))
            other = np.flatnonzero(~frame.is_test & (frame.axis != axis))
            expected = frame.iloc[test].copy()
            expected['source_row'] = test
            require_rows(pd.read_csv(folder / 'test_rows.csv'), expected, f'{model}/{axis} test rows')
            labels = frame.iloc[test].load_label.astype(int)
            reference = np.load(folder / 'in_distribution.npy', allow_pickle=False)
            require_scores(reference, len(test), f'{model}/{axis} in-distribution')
            reference_auc = roc_auc_score(labels, reference)
            candidates = complete_pairs(frame[(frame.axis == axis) & ~frame.is_test])
            pool_size = len({pair['group'] for pair in candidates})
            # The saved pair order must equal a fresh draw, and every K's scores, counts
            # and AUC must agree with it.
            for seed in SEEDS:
                expected_order = pd.DataFrame(pair_order(candidates, seed)[:max(KS)])
                require_rows(pd.read_csv(folder / f'calibration_order_s{seed}.csv'), expected_order,
                             f'{model}/{axis}/seed{seed} calibration pairs')
                for k in KS:
                    name = 'k0' if k == 0 else f'k{k}_s{seed}'
                    scores = np.load(folder / f'{name}.npy', allow_pickle=False)
                    require_scores(scores, len(test), f'{model}/{axis}/{name}')
                    row = detail[(detail.model == model) & (detail.axis == axis)
                                 & (detail.seed == seed) & (detail.k_pairs == k)].iloc[0]
                    expected_counts = dict(calibration_rows=2*k, candidate_base_groups=pool_size,
                                           n_other_train=len(other), n_test=len(test))
                    if any(row[key] != value for key, value in expected_counts.items()):
                        raise ValueError(f'{model}/{axis}/{name}: incorrect split or calibration counts')
                    require_close([row.auc, row.in_distribution_auc], [roc_auc_score(labels, scores), reference_auc],
                                  f'{model}/{axis}/{name} AUCs')
    return detail
