"""Train-only preprocessing, fitted load scorers, and cross-validated scores.

Every scorer here imputes and standardizes features using statistics from the
training rows only, then fits a binary low-load vs high-load classifier:
`fit_mlp` is the ELS model, and `fit_logistic` serves the single-feature and
length baselines. `score_tier` runs one scorer over given folds and returns
out-of-fold scores for every row.
"""
from __future__ import annotations

from typing import Optional
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from els.features import ELS_FEATURES, LENGTH_FEATURES, feature_matrix

# The five benchmark load axes; ELS is trained only on rows of these axes.
AXES = ['quantity', 'similarity', 'interference', 'discontinuity', 'underspecification']
# Display names of the three benchmark tasks.
TASK_NAMES = {'math': 'Math', 'context': 'Context', 'analogy': 'Analogy'}
# Number of cross-validation folds.
N_FOLDS = 5
# Scorers compared in the cross-axis table: the length, entropy and margin baselines and ELS.
TIERS = ['length', 'entropy_only', 'margin_only', 'els']
# Feature columns of each baseline scorer.
BASELINE_FEATURES = {'length': LENGTH_FEATURES, 'entropy_only': ['mean_entropy'],
                     'margin_only': ['mean_logit_margin']}
# ELS MLP: hidden layer widths and full-batch Adam settings.
HIDDEN = (64, 32)
EPOCHS = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3


def impute(X: np.ndarray, med: Optional[np.ndarray] = None):
    """Median-impute NaN columns (all-NaN columns become 0); return (X_imputed, med)."""
    if med is None:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'All-NaN slice', RuntimeWarning)
            med = np.nanmedian(X, axis=0)
        med = np.where(np.isnan(med), 0.0, med)
    # Replace each NaN with its column's median.
    Xc = X.copy().astype(float)
    idx = np.where(np.isnan(Xc))
    Xc[idx] = np.take(med, idx[1])
    return Xc, med


def _balanced_weights(group_keys: np.ndarray) -> np.ndarray:
    """Sample weights so each unique group contributes equally to total mass."""
    unique, counts = np.unique(group_keys, return_counts=True)
    inv = {g: 1.0 / c for g, c in zip(unique, counts)}
    w = np.array([inv[g] for g in group_keys], dtype=float)
    w *= len(w) / w.sum()  # normalize so sum = N
    return w


def fit_logistic(Xtr, ytr, Xte, balance_keys=None):
    """Fit L2 logistic regression after training-only imputation and scaling.

    Returns the predicted high-load probability of each test row.
    """
    Xtr_imp, med = impute(Xtr)
    Xte_imp, _ = impute(Xte, med)
    sw = _balanced_weights(balance_keys) if balance_keys is not None else None
    pipe = Pipeline([('s', StandardScaler()),
                     ('c', LogisticRegression(penalty='l2', solver='liblinear', C=1.0, max_iter=1000))])
    pipe.fit(Xtr_imp, ytr, c__sample_weight=sw)
    return pipe.predict_proba(Xte_imp)[:, 1]


def fit_mlp(Xtr, ytr, Xte, balance_keys=None, seed=0):
    """Fit a weighted binary-cross-entropy MLP after training-only imputation and scaling.

    seed fixes the weight initialization. Returns the predicted high-load probability
    of each test row, as float32.
    """
    torch.manual_seed(seed)
    # Medians and scaling come from the training rows only.
    Xtr_imp, med = impute(Xtr)
    Xte_imp, _ = impute(Xte, med)
    scaler = StandardScaler().fit(Xtr_imp)
    Xtr_s = scaler.transform(Xtr_imp).astype('float32')
    Xte_s = scaler.transform(Xte_imp).astype('float32')

    # Per-row loss weights; unweighted when no balance keys are given.
    sw = _balanced_weights(balance_keys) if balance_keys is not None else np.ones(len(ytr))
    sw = sw.astype('float32')

    # Fully connected ReLU network with one output logit.
    layers = []
    in_dim = Xtr_s.shape[1]
    for h in HIDDEN:
        layers += [nn.Linear(in_dim, h), nn.ReLU()]
        in_dim = h
    layers += [nn.Linear(in_dim, 1)]
    net = nn.Sequential(*layers)

    opt = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    bce = nn.BCEWithLogitsLoss(reduction='none')

    Xt = torch.from_numpy(Xtr_s)
    yt = torch.from_numpy(ytr.astype('float32')).unsqueeze(1)
    wt = torch.from_numpy(sw).unsqueeze(1)
    # Full-batch training: each epoch is one Adam step on the weighted mean loss.
    for _ in range(EPOCHS):
        opt.zero_grad()
        logits = net(Xt)
        loss = (bce(logits, yt) * wt).mean()
        loss.backward()
        opt.step()

    net.eval()
    with torch.no_grad():
        Xe = torch.from_numpy(Xte_s)
        out = net(Xe)
        scores = torch.sigmoid(out).numpy().ravel()
    return scores


def group_folds(n: int, groups: np.ndarray, n_splits=N_FOLDS):
    """GroupKFold splits over n rows; all rows of a group fall in the same test fold.

    Uses fewer folds when there are fewer than n_splits groups, and no splits when
    there are fewer than two groups.
    """
    # GroupKFold does not shuffle, so the same rows and groups always give the same folds.
    n_g = len(np.unique(groups))
    splits = min(n_splits, n_g)
    if splits < 2:
        return iter([])
    return GroupKFold(n_splits=splits).split(np.arange(n), np.zeros(n), groups)


def score_tier(tier: str, frame: pd.DataFrame, y: np.ndarray, splits, balance: np.ndarray) -> np.ndarray:
    """Return one scorer's out-of-fold scores for the rows of frame.

    splits lists (train, test) row positions; each test fold is scored by a model fit
    on its train rows. ELS fits the MLP on ELS_FEATURES; the baselines fit logistic
    regression on their own columns.
    """
    if tier not in TIERS:
        raise ValueError(f'Unknown scorer: {tier}')
    if tier == 'els':
        x, fit = feature_matrix(frame, ELS_FEATURES), fit_mlp
    else:
        x, fit = feature_matrix(frame, BASELINE_FEATURES[tier]), fit_logistic
    scores = np.full(len(frame), np.nan)
    for train, test in splits:
        scores[test] = fit(x[train], y[train], x[test], balance_keys=balance[train])
    return scores
