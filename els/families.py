"""Feature-family ablations and signatures, i.e. standardized high-minus-low differences (Tables D6-D7)."""
# Two analyses of the 30 ELS features grouped into four families (features.family):
# signatures average each feature's standardized high-minus-low load difference per
# axis and family; ablations retrain ELS on each family alone and with each family
# removed, and report per-axis AUC. main fits and saves the ablation scores;
# evaluate recomputes both tables from the features and those scores.
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from els.features import ELS_FEATURES, FAMILIES, aggregated, balance_keys, family, feature_matrix, load_features
from els.paths import GZIP, new_output, parser_with_output, require_inputs
from els.scoring import AXES, fit_mlp, group_folds


def labeled(data):
    """Load-labeled rows of the five axes."""
    data = data[data.axis.isin(AXES) & data.load_label.notna()].copy()
    data['load_label'] = data.load_label.astype(int)
    return data


def standardized_difference(sub, feature):
    """(mean high - mean low) / sqrt of the average sample variance; NaN with fewer than two per side."""
    use = sub[['load_label', feature]].dropna()
    low = use.loc[use.load_label == 0, feature].to_numpy(dtype=float)
    high = use.loc[use.load_label == 1, feature].to_numpy(dtype=float)
    if len(low) < 2 or len(high) < 2:
        return np.nan
    # Pooled standard deviation; a feature that is constant on both sides gets an effect of 0.
    scale = np.sqrt((high.var(ddof=1) + low.var(ddof=1)) / 2.0)
    return 0.0 if scale < 1e-12 else float((high.mean() - low.mean()) / scale)


def signatures(data):
    """Per model/axis/feature effects, averaged over defined cells within each axis and family."""
    # One effect size per model, axis and feature; cells lacking two prompts per label are dropped.
    rows = [dict(model=model, axis=axis, feature=feature, family=family(feature),
                 d=standardized_difference(data[(data.model == model) & (data.axis == axis)], feature))
            for model in sorted(data.model.unique()) for axis in AXES for feature in ELS_FEATURES]
    cells = pd.DataFrame(rows).dropna(subset=['d']).reset_index(drop=True)
    # summary: one row per axis and one column per family; each value is a mean effect
    # over models and the family's features.
    summary = cells.groupby(['axis', 'family']).d.mean().unstack().reindex(index=AXES, columns=FAMILIES)
    return cells, summary


def configurations():
    """(mode, family, features) for the full set, each family alone and each family dropped."""
    yield 'full', '-', ELS_FEATURES
    for mode in ['keep', 'drop']:
        # keep selects the family's features; drop selects every other feature.
        for name in FAMILIES:
            yield mode, name, [f for f in ELS_FEATURES if (family(f) == name) == (mode == 'keep')]


def ablation_scores(data):
    """Five-fold out-of-fold ELS scores (pair_key groups) for the full, keep-only and drop-one feature sets."""
    parts = []
    for model in sorted(data.model.unique()):
        sub = data[data.model == model]
        y, balance = sub.load_label.to_numpy(), balance_keys(sub)
        # The same pair_key folds are reused for every feature set, so the sets are compared on equal splits.
        splits = list(group_folds(len(sub), sub.pair_key.astype(str).to_numpy()))
        scores = pd.DataFrame(index=sub.index)
        for mode, name, features in configurations():
            x, column = feature_matrix(sub, features), np.full(len(sub), np.nan, dtype=np.float32)
            for train, test in splits:
                column[test] = fit_mlp(x[train], y[train], x[test], balance_keys=balance[train])
            scores[f'{mode}:{name}'] = column
        parts.append(scores)
        print(f'Family ablations complete: {model}', flush=True)
    return pd.concat(parts).rename_axis('row').reset_index()


def ablation_tables(data, scores):
    """Per-axis AUC for each model and feature set, then mean over models in percent."""
    # Saved scores are keyed by the feature-table row index.
    frame = scores.set_index('row').join(data[['model', 'axis', 'load_label']])
    rows = []
    for (model, axis), part in frame.groupby(['model', 'axis'], sort=True):
        for mode, name, features in configurations():
            rows.append(dict(model=model, axis=axis, mode=mode, family=name, n_features=len(features),
                             auc=roc_auc_score(part.load_label, part[f'{mode}:{name}'])))
    cells = pd.DataFrame(rows)
    # Rows are (mode, family) in configuration order; columns are the five axes.
    table = (100 * cells.groupby(['mode', 'family', 'axis']).auc.mean()).unstack()[AXES]
    order = [(mode, name) for mode, name, _ in configurations()]
    return cells, table.reindex(order)


def drop_effects(ablation_table):
    """AUC lost on each axis when one family is dropped from the full feature set."""
    return ablation_table.loc[('full', '-')] - ablation_table.xs('drop')


def summary_frame(signature_table, ablation_table):
    """Both tables in one frame indexed by (table, row), with family columns then axis columns.

    Each table fills only its own columns; the other columns are NaN.
    """
    signature = signature_table.rename_axis('row').assign(table='signature').set_index('table', append=True)
    ablation = ablation_table.copy()
    ablation.index = pd.MultiIndex.from_tuples([('ablation', f'{mode}:{name}') for mode, name in ablation.index])
    return pd.concat([signature.swaplevel(), ablation]).rename_axis(['table', 'row'])[[*FAMILIES, *AXES]]


def evaluate(output, data):
    """Recompute signatures from features and ablation AUCs from saved scores."""
    return (*signatures(data), *evaluate_ablations(output, data))


def evaluate_ablations(output, data):
    """Recompute the ablation AUCs from saved scores; data needs only the labeled prompt rows."""
    scores = pd.read_csv(output / 'ablation_scores.csv.gz')
    # Require one finite score per labeled prompt for every feature set, in configuration order
    # (the first column is the row index).
    columns = [f'{mode}:{name}' for mode, name, _ in configurations()]
    if (scores.row.duplicated().any() or set(scores.row) != set(data.index)
            or list(scores.columns[1:]) != columns or not np.isfinite(scores[columns].to_numpy()).all()):
        raise ValueError('Saved ablation scores do not cover the labeled prompts')
    # The scores were fitted as float32; evaluate them at that precision.
    scores[columns] = scores[columns].astype(np.float32)
    return ablation_tables(data, scores)


def main():
    """Fit and save ablation scores, then write the signature and ablation tables."""
    parser = parser_with_output(__doc__)
    args = parser.parse_args()
    require_inputs(parser, args.features)
    output = new_output(args.output_dir)
    torch.set_num_threads(1)
    data = labeled(aggregated(load_features(args.features)))
    ablation_scores(data).to_csv(output / 'ablation_scores.csv.gz', index=False, compression=GZIP)
    signature_cells, signature_table, ablation_cells, ablation_table = evaluate(output, data)
    signature_cells.to_csv(output / 'signature_cells.csv', index=False)
    signature_table.to_csv(output / 'signatures.csv')
    ablation_cells.to_csv(output / 'ablation_cells.csv', index=False)
    ablation_table.to_csv(output / 'ablations.csv')
    summary_frame(signature_table, ablation_table).to_csv(output / 'summary.csv')
    print(signature_table.round(2).to_string(), flush=True)
    print(ablation_table.round(1).to_string(), flush=True)


if __name__ == '__main__':
    main()
