#!/usr/bin/env python3
"""
Stable Boruta feature selection utilities.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from boruta import BorutaPy


def boruta_selector(X, y, *, k_features: int, task: str = "clf",
                    random_state: int = 42, max_iter: int = 150, perc: int = 85,
                    n_trees: int | None = None):
    """
    Stable Boruta that *always* returns exactly `k_features` indices,
    but never freezes your run.
    """
    # ── 0.  Sanity ────────────────────────────────────────────────────────────
    if np.isnan(X).any():
        raise ValueError("X still contains NaNs; impute before Boruta.")
    if len(np.unique(y)) < 2:
        raise ValueError("Need ≥ 2 classes / unique y‐values for Boruta.")
    # Optional quick variance filter – constant cols break Boruta's Z-score
    keep = X.var(axis=0) > 0
    X = X[:, keep]

    # ── 1.  Reasonable forest ────────────────────────────────────────────────
    n_trees = n_trees or max(64, k_features * 2)       # explicit, no 'auto'
    rf_cls = RandomForestClassifier if task == "clf" else RandomForestRegressor
    rf = rf_cls(
        n_estimators=n_trees,
        max_depth=8,                       # shallow → faster & still robust
        max_features="sqrt",
        class_weight="balanced" if task == "clf" else None,
        n_jobs=1,                          # avoid nested threading deadlock
        random_state=random_state,
    )

    # ── 2.  Boruta run ────────────────────────────────────────────────────────
    bor = BorutaPy(
        estimator=rf,
        n_estimators="auto",               # Boruta will *not* grow > 4× n_trees
        max_iter=min(max_iter, 40),        # cap – 40 ≈ enough up to 1 k features
        perc=perc,
        random_state=random_state,
        alpha=0.05,
        verbose=0,
    )
    bor.fit(X, y)

    # ── 3.  Post-processing to exactly k_features ────────────────────────────
    support_idx = np.flatnonzero(bor.support_)
    if support_idx.size == 0:
        support_idx = np.argsort(bor.ranking_)[:k_features]
    elif support_idx.size > k_features:
        ranking = bor.ranking_[support_idx]
        support_idx = support_idx[np.argsort(ranking)[:k_features]]
    elif support_idx.size < k_features:
        remaining = np.setdiff1d(np.arange(X.shape[1]), support_idx)
        extra = remaining[np.argsort(bor.ranking_[remaining])][:k_features - support_idx.size]
        support_idx = np.sort(np.concatenate([support_idx, extra]))

    # Map back to original column indices if we dropped constant columns
    if keep is not None and keep.sum() != X.shape[1]:
        support_idx = np.flatnonzero(keep)[support_idx]

    return support_idx 