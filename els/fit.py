"""Fit ELS and baselines on the five load axes with base-disjoint folds (Table 3: held-out base problems)."""
# For each model, labeled prompts are split into five folds so that every prompt
# derived from the same base problem (on any axis) lands in the same fold. Each
# scorer in scoring.TIERS is trained on the other folds, restricted to the five
# benchmark axes, and scores the held-out fold; per-axis AUCs are averaged over
# models into summary.csv. Scores are cached per model and scorer, so an
# interrupted run resumes in the same output directory. run.py fit calls this as
# `python -m els.fit --output-dir DIR`, and els.verify checks the saved outputs.
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from els.features import aggregated, balance_keys, load_features
from els.paths import parser_with_output, require_inputs, start_output
from els.scoring import AXES, N_FOLDS, TIERS, group_folds, score_tier
from els.splits import identity_groups
from els.validation import IDENTITY_COLUMNS, cached_scores, cross_axis_summary, require_disjoint


def protocol(features_path):
    """Settings that a resumed run must share, including the feature-table hash."""
    return {
        'input_sha256': hashlib.sha256(features_path.read_bytes()).hexdigest(),
        'train_axes': AXES, 'folds': N_FOLDS,
        'groups': 'connected task/base labels and task/target IDs across all axes',
        'features': 'fixed 30 features; layer positions aggregated within each model',
        'training': 'train-only imputation/scaling, seed 0',
    }


def main():
    """Fit every scorer for every model and write predictions, AUCs and the summary."""
    parser = parser_with_output(__doc__)
    args = parser.parse_args()
    require_inputs(parser, args.features)
    # One torch thread, so every fit runs its floating-point operations in the same order.
    torch.set_num_threads(1)
    output = start_output(args.output_dir, protocol(args.features))
    data = aggregated(load_features(args.features))
    # One AUC row per (model, scorer, axis); the partial CSV is rewritten after each
    # scorer and removed once per_model.csv is complete.
    rows = []
    partial = output / 'per_model_partial.csv'
    models = sorted(data.model.dropna().unique())
    for model in models:
        sub = data[(data.model == model) & data.load_label.notna()].reset_index(drop=True)
        groups = identity_groups(sub)
        # Restrict each fold's training rows to the five axes, record each row's test
        # fold, and check that training rows share no group, base label or target ID
        # with the test rows.
        # folds holds each row's test fold (-1 until assigned); splits holds the (train, test)
        # row positions passed to every scorer.
        trainable = sub.axis.isin(AXES).to_numpy()
        folds, splits = np.full(len(sub), -1), []
        for fold, (train, test) in enumerate(group_folds(len(sub), groups)):
            train = train[trainable[train]]
            require_disjoint(sub, train, test, groups)
            if sub.iloc[train].load_label.nunique() != 2:
                raise ValueError(f'{model} fold {fold}: the training rows need both load labels')
            folds[test] = fold
            splits.append((train, test))
        if (folds < 0).any():
            raise ValueError(f'{model}: some rows were never assigned a test fold')
        # Out-of-fold predictions are saved with the row identities, groups and folds
        # so that els.verify can check them.
        saved = sub[IDENTITY_COLUMNS].copy()
        saved['group'] = groups
        saved['fold'] = folds
        y = sub.load_label.astype(int).to_numpy()
        balance = balance_keys(sub)
        # Every scorer uses the same splits; a saved <model>_<tier>.npy is reused on resume.
        for tier in TIERS:
            scores = cached_scores(output / f'{model}_{tier}.npy', len(y), f'{model}/{tier} predictions',
                                   lambda: score_tier(tier, sub, y, splits, balance))
            saved[tier] = scores
            # AUC per benchmark axis, from the scores of that axis's rows.
            for axis in AXES:
                mask = sub.axis.to_numpy() == axis
                rows.append(dict(model=model, tier=tier, axis=axis,
                                 n=int(mask.sum()), auc=roc_auc_score(y[mask], scores[mask])))
            pd.DataFrame(rows).to_csv(partial, index=False)
            print(f'Complete {model}: {tier}', flush=True)
        saved.to_csv(output / f'oof_{model}.csv', index=False)
    detail = pd.DataFrame(rows)
    if len(detail) != len(models) * len(TIERS) * len(AXES):
        raise ValueError('cross-axis results are missing cells')
    detail.to_csv(output / 'per_model.csv', index=False)
    summary = cross_axis_summary(detail)
    summary.to_csv(output / 'summary.csv')
    partial.unlink()
    print(summary.to_string(), flush=True)
    print(f'Table 3 results written to {output}')


if __name__ == '__main__':
    main()
