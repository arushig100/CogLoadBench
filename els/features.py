"""Fixed ELS feature set and per-layer aggregation.

The feature table stores some activation metrics as one value per layer.
`aggregated` turns each of those arrays into five scalar columns (early, mid,
late, last and pooled layer means), and the ELS scorers read a fixed list of 30
of those columns (`ELS_FEATURES`). This module also maps feature names to the
four feature families used by the family analysis.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

# Per-layer array columns in the feature table; each becomes five aggregated columns.
# Suffixes name the prompt region: ctx = context, q = question, full = whole prompt.
# drift, angle and M_to_next hold one entry per pair of consecutive decoder layers;
# the rest hold one entry per decoder layer.
LAYER_COLUMNS = [
    '_per_layer_eff_rank_ctx', '_per_layer_eff_rank_q',
    '_per_layer_gini_ctx', '_per_layer_gini_q', '_per_layer_ctx_q_cos',
    '_per_layer_token_norm_ctx', '_per_layer_token_norm_q',
    '_per_layer_act_mag_ctx',    '_per_layer_act_mag_q',    '_per_layer_act_mag_full',
    '_per_layer_drift_ctx',      '_per_layer_drift_q',      '_per_layer_drift_full',
    '_per_layer_angle_ctx',      '_per_layer_angle_q',      '_per_layer_angle_full',
    '_per_layer_M_to_next_ctx',  '_per_layer_M_to_next_q',  '_per_layer_M_to_next_full',
]
# Suffixes of the aggregated columns, e.g. 'gini_q__mid'.
LAYER_POSITIONS = ('early', 'mid', 'late', 'last', 'pooled')
# Feature families, in the order used by the family tables.
FAMILIES = ['Magnitude', 'Sparsity', 'Stability', 'Alignment']
# The fixed 30 activation features of ELS, in training order.
ELS_FEATURES = [
    'angle_full__mid', 'token_norm_ctx__early', 'act_mag_ctx__mid',
    'eff_rank_ctx__pooled', 'token_norm_ctx__pooled', 'eff_rank_ctx__late',
    'angle_full__late', 'gini_q__mid', 'eff_rank_ctx__mid', 'ctx_q_cos__mid',
    'eff_rank_ctx__early', 'gini_q__pooled', 'gini_ctx__late',
    'drift_full__late', 'drift_full__early', 'ctx_q_cos__late',
    'act_mag_full__last', 'gini_q__early', 'angle_ctx__early',
    'M_to_next_q__early', 'eff_rank_q__last', 'angle_q__mid',
    'M_to_next_ctx__mid', 'drift_q__late', 'M_to_next_q__mid',
    'gini_ctx__mid', 'angle_q__late', 'act_mag_q__late',
    'M_to_next_full__last', 'gini_q__last',
]
# Prompt, context and question token counts.
LENGTH_FEATURES = ['prompt_tok_count', 'ctx_tok_count', 'q_tok_count']


def load_features(path) -> pd.DataFrame:
    """Read an ELS feature table."""
    return pd.read_parquet(path)


def aggregated(frame: pd.DataFrame) -> pd.DataFrame:
    """Expand per-layer arrays into early, mid, late, last and pooled means, using each model's own layer count."""
    if 'model' not in frame.columns or frame.empty:
        return _aggregated(frame)
    parts = [_aggregated(group) for _, group in frame.groupby('model', sort=False, dropna=False)]
    # Restore the input row order.
    return pd.concat(parts).loc[frame.index]


def _aggregated(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the per-layer arrays of one model's rows; nL is the longest array of each column."""
    sub = frame.copy()
    for col in LAYER_COLUMNS:
        if col not in sub.columns:
            continue
        # Output prefix, e.g. '_per_layer_gini_q' -> 'gini_q'.
        base = col.lstrip('_').replace('per_layer_', '')
        lengths = [len(x) for x in sub[col].dropna()]
        # A column with no nonempty arrays yields NaN aggregates.
        if not lengths or max(lengths) == 0:
            for pos in LAYER_POSITIONS:
                sub[f'{base}__{pos}'] = float('nan')
            continue
        nL = max(lengths)

        def _to_row(x, nL=nL):
            """One per-layer array padded with NaN to nL entries."""
            v = list(x) if isinstance(x, (list, tuple, np.ndarray)) else []
            return v + [np.nan] * (nL - len(v))
        # mat: [rows, nL]; a missing array becomes a row of NaN.
        mat = np.asarray(sub[col].map(_to_row).tolist(), dtype=float)
        # Split the nL layers into thirds: early = [0, a), mid = [a, b), late = [b, nL).
        # With three or more layers each third holds at least one layer; an empty third
        # has NaN means. NaN entries are ignored in the means.
        a, b = max(nL // 3, 1), max(2 * nL // 3, nL // 3 + 1)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'Mean of empty slice')
            sub[f'{base}__early']  = np.nanmean(mat[:, :a],   axis=1)
            sub[f'{base}__mid']    = np.nanmean(mat[:, a:b],  axis=1)
            sub[f'{base}__late']   = np.nanmean(mat[:, b:],   axis=1)
            sub[f'{base}__last']   = mat[:, -1]
            sub[f'{base}__pooled'] = np.nanmean(mat,          axis=1)
    return sub


def family(name: str) -> str:
    """Feature family of an aggregated feature name, from its metric prefix."""
    if 'ctx_q_cos' in name:
        return 'Alignment'
    if name.startswith(('token_norm_', 'act_mag_')):
        return 'Magnitude'
    if name.startswith(('eff_rank_', 'gini_')):
        return 'Sparsity'
    if name.startswith(('angle_', 'drift_', 'M_to_next_')):
        return 'Stability'
    raise ValueError(f'Unknown feature family: {name}')


def feature_matrix(df: pd.DataFrame, feats: list[str]) -> np.ndarray:
    """Float matrix of the named columns."""
    return df[feats].astype(float).to_numpy()


def balance_keys(frame: pd.DataFrame) -> np.ndarray:
    """Axis::task keys that weight each axis/task cell equally.

    The scorers pass these to `scoring._balanced_weights` so that large cells do not
    dominate training.
    """
    return (frame.axis.astype(str) + '::' + frame.task.astype(str)).to_numpy()
