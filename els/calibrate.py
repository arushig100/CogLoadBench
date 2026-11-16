"""Few-shot calibration to an unseen load axis, held out one axis at a time (Table D1)."""
# A fixed 20% of base-problem groups per task is held out as the test set for every
# model, axis, K and seed. For each model and held-out axis, ELS is trained on the
# non-test rows of the other four axes plus K low/high pairs from the held-out axis, and
# scored by ROC AUC on the held-out axis's test rows. K grows along one nested pair
# order per sampling seed, so the pairs used at K are a prefix of those used at 2K. The
# in-distribution reference trains on all non-test rows of all five axes. run.py fit
# calls this module, and els.verify rebuilds and checks its saved outputs.
from __future__ import annotations

import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from els.features import ELS_FEATURES, aggregated, balance_keys, feature_matrix, load_features
from els.paths import parser_with_output, require_inputs, start_output
from els.scoring import AXES, fit_mlp
from els.splits import KS, SEEDS, SPLIT_SEED, complete_pairs, fixed_test_groups, identity_groups, pair_order
from els.validation import CALIBRATION_COLUMNS, cached_scores, calibration_summary, require_disjoint


def run_cell(payload):
    """Return test AUCs for one model and held-out axis as K calibration pairs are added."""
    # frame: one model's row metadata (CALIBRATION_COLUMNS); X: its [rows, 30] ELS feature matrix.
    model, axis, frame, X, output = payload
    torch.set_num_threads(1)
    # Row sets: test = held-out axis in test groups; other = remaining axes outside
    # test groups (the K = 0 training set); all_train = every axis outside test groups.
    folder = output / model / axis
    folder.mkdir(parents=True, exist_ok=True)
    y = frame.load_label.astype(int).to_numpy()
    balance = balance_keys(frame)
    groups = frame.group.to_numpy()
    test = np.flatnonzero(frame.is_test.to_numpy() & (frame.axis.to_numpy() == axis))
    other = np.flatnonzero(~frame.is_test.to_numpy() & (frame.axis.to_numpy() != axis))
    all_train = np.flatnonzero(~frame.is_test.to_numpy())
    require_disjoint(frame, all_train, test, groups)
    held_train = frame[(frame.axis == axis) & ~frame.is_test]
    # Calibration pairs come from the held-out axis outside the test groups.
    candidates = complete_pairs(held_train)
    # pair_order takes one pair per group, so the largest K needs that many groups.
    pool_size = len({p['group'] for p in candidates})
    if pool_size < max(KS):
        raise ValueError(f'{model}/{axis}: {pool_size} calibration pairs available, {max(KS)} needed')
    if len(np.unique(y[test])) != 2:
        raise ValueError(f'{model}/{axis}: the test rows need both load labels')
    test_meta = frame.iloc[test].copy()
    test_meta['source_row'] = test
    test_meta.to_csv(folder / 'test_rows.csv', index=False)

    def score(train, name):
        """Fit ELS on the train rows (or load cached scores) and return the test AUC."""
        require_disjoint(frame, train, test, groups)
        result = cached_scores(folder / f'{name}.npy', len(test), f'{model}/{axis}/{name} predictions',
                               lambda: fit_mlp(X[train], y[train], X[test], balance_keys=balance[train]))
        return roc_auc_score(y[test], result)

    # K = 0 adds no pairs, so its single fit serves every seed.
    reference = score(all_train, 'in_distribution')
    zero = score(other, 'k0')
    rows = []
    for seed in SEEDS:
        ordered = pair_order(candidates, seed)
        pd.DataFrame(ordered[:max(KS)]).to_csv(folder / f'calibration_order_s{seed}.csv', index=False)
        for k in KS:
            # The first k pairs of this seed's order add 2k labeled rows to training.
            picked = ordered[:k]
            pair_rows = np.array([index for pair in picked for index in [pair['low'], pair['high']]], dtype=int)
            # Each pair must add one low and one high row from its own group.
            if len(pair_rows) != 2 * k or len({p['group'] for p in picked}) != k:
                raise ValueError(f'{model}/{axis}: expected {k} calibration pairs from {k} distinct groups')
            if k and np.bincount(y[pair_rows], minlength=2).tolist() != [k, k]:
                raise ValueError(f'{model}/{axis}: calibration pairs must add {k} low and {k} high rows')
            train = np.concatenate([other, pair_rows])
            auc = zero if k == 0 else score(train, f'k{k}_s{seed}')
            rows.append(dict(model=model, axis=axis, seed=seed, k_pairs=k,
                             calibration_rows=2*k, candidate_base_groups=pool_size,
                             n_other_train=len(other), n_test=len(test), auc=auc,
                             in_distribution_auc=reference))
    return rows


def protocol(features_path):
    """Settings that a resumed run must share, including the feature-table hash."""
    return dict(
        input_sha256=hashlib.sha256(features_path.read_bytes()).hexdigest(),
        axes=AXES, k_pairs=KS, calibration_seeds=SEEDS, split_seed=SPLIT_SEED,
        split='20% of base groups per task, same test groups across models, axes, K and seeds',
        pairing='one low/high pair with same task, variant and base label; one pair per connected base group',
        sampling='nested, task-balanced round robin; prefix task counts differ by at most one until a task pool runs out',
        training='fixed 30 features and the weighted BCE MLP used for cross-axis evaluation; training seed 0',
        metric='pooled test ROC AUC per model/axis; macro across models and sampling seeds',
        reference='all five axes on non-test groups, evaluated on identical held-axis test prompts')


def main():
    """Build the fixed split, run every model/axis cell, and write metrics and summaries."""
    parser = parser_with_output(__doc__)
    parser.add_argument('--workers', type=int, default=1, help='Parallel processes (default: 1)')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    require_inputs(parser, args.features)
    output = start_output(args.output_dir, protocol(args.features))
    data = aggregated(load_features(args.features))
    data = data[data.axis.isin(AXES) & data.load_label.notna()].copy()
    # Groups are formed over all models together, so the test split is the same for each model.
    data['group'] = identity_groups(data)
    data['is_test'] = data.group.isin(fixed_test_groups(data))
    # One payload per (model, held-out axis); cells run in separate processes.
    payloads = []
    for model, sub in data.groupby('model', sort=True):
        sub = sub.reset_index(drop=True)
        meta = sub[CALIBRATION_COLUMNS].copy()
        meta.to_csv(output / f'rows_{model}.csv', index=False)
        X = feature_matrix(sub, ELS_FEATURES)
        for axis in AXES:
            payloads.append((model, axis, meta, X, output))
    rows = []
    partial = output / 'all_metrics_partial.csv'
    # Workers start from a fresh interpreter (spawn) rather than a fork of this process.
    # Metrics are written to the partial CSV as cells finish, in completion order.
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
        futures = [pool.submit(run_cell, payload) for payload in payloads]
        for future in as_completed(futures):
            cell = future.result()
            # Every cell returns len(SEEDS) * len(KS) rows, so len(rows) // len(cell) counts cells.
            rows.extend(cell)
            pd.DataFrame(rows).to_csv(partial, index=False)
            print(f'Completed {cell[0]["model"]} {cell[0]["axis"]} ({len(rows) // len(cell)}/{len(payloads)} cells)', flush=True)
    detail = pd.DataFrame(rows).sort_values(['model', 'axis', 'seed', 'k_pairs'])
    if len(detail) != len(payloads) * len(SEEDS) * len(KS):
        raise ValueError('calibration results are missing cells')
    detail.to_csv(output / 'all_metrics.csv', index=False)
    summary = calibration_summary(detail)
    summary.to_csv(output / 'summary.csv')
    partial.unlink()
    print(summary.to_string(), flush=True)


if __name__ == '__main__':
    main()
