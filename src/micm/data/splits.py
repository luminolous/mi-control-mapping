"""Train/test boundary for the main protocol.

Session T is fitted on, session E is evaluated on, per subject. That is the
dataset's own protocol and the whole of the split logic; there is deliberately
no option for a random split, because a random split over a cue-based session
leaks slow drift between train and test and inflates every number downstream.

`fit_transform_split` exists so that leakage is awkward to write. It takes the
estimator and does the fitting itself, so a caller never holds the untransformed
test half at the moment it calls `fit`. Prefer it over calling `fit` and
`transform` by hand.

Every function here is covered by `tests/test_no_leakage.py`, which was written
before this file existed.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from micm.data.constants import SESSION_TEST, SESSION_TRAIN, SESSIONS

_MIN_FOLDS = 2


class FitTransformer(Protocol):
    """Anything with the scikit-learn fit/transform pair.

    `y` is optional so unsupervised scalers and supervised transformers such as
    CSP can both travel through `fit_transform_split`. Without it, the supervised
    ones would need a second code path, and a second code path is where leakage
    gets written.
    """

    def fit(self, X: np.ndarray, y: Any = None) -> Any: ...

    def transform(self, X: np.ndarray) -> np.ndarray: ...


def _validate_trial_meta(trial_meta: pd.DataFrame) -> None:
    """Check the assumptions the index arithmetic below relies on.

    Raises:
        ValueError: if the `session` column is missing, holds an unknown label,
            does not contain both sessions, or if the frame is not indexed by
            position from zero.
    """
    if "session" not in trial_meta.columns:
        raise ValueError("trial_meta has no 'session' column")

    expected_index = pd.RangeIndex(len(trial_meta))
    if not trial_meta.index.equals(expected_index):
        raise ValueError(
            "trial_meta must be indexed by position from 0; "
            "call reset_index(drop=True) after concatenating subjects, otherwise "
            "the returned positions do not address the rows you think they do"
        )

    present = set(trial_meta["session"].unique())
    unknown = present - set(SESSIONS)
    if unknown:
        raise ValueError(f"unknown session labels {sorted(unknown)}, expected {list(SESSIONS)}")

    missing = set(SESSIONS) - present
    if missing:
        raise ValueError(
            f"trial_meta is missing session(s) {sorted(missing)}; the protocol fits on "
            f"{SESSION_TRAIN} and evaluates on {SESSION_TEST}, so both must be present"
        )


def train_test_indices(trial_meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, test_idx) as positional indices into `trial_meta`.

    Train is session T, test is session E. Guaranteed disjoint and jointly
    exhaustive; both are asserted internally. Trials are never dropped here, so
    artifact rejection must happen before this call.

    Raises:
        ValueError: if `trial_meta` fails `_validate_trial_meta`, or if either
            side comes out empty.
    """
    _validate_trial_meta(trial_meta)

    session = trial_meta["session"].to_numpy()
    train_idx = np.flatnonzero(session == SESSION_TRAIN)
    test_idx = np.flatnonzero(session == SESSION_TEST)

    if train_idx.size == 0 or test_idx.size == 0:
        raise ValueError(
            f"empty split: {train_idx.size} train and {test_idx.size} test trials"
        )

    assert np.intersect1d(train_idx, test_idx).size == 0
    assert train_idx.size + test_idx.size == len(trial_meta)
    return train_idx, test_idx


def inner_cv_indices(
    trial_meta: pd.DataFrame, n_folds: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified k-fold over session T only, for hyperparameter selection.

    Every returned index addresses `trial_meta` positionally and is a member of
    the train split. Session E is not reachable from here: the folds are built
    from the train subset and mapped back, so there is no code path along which
    a test row could enter a fold.

    Args:
        trial_meta: the trial table, indexed by position from zero.
        n_folds: number of folds, at least 2.
        seed: fixes the shuffle. Two calls with the same seed give identical folds.

    Returns:
        `n_folds` pairs of (inner_train_idx, inner_val_idx). The validation sets
        partition the train split.

    Raises:
        ValueError: if `n_folds` is below 2 or exceeds the smallest class count.
    """
    if n_folds < _MIN_FOLDS:
        raise ValueError(f"n_folds must be at least {_MIN_FOLDS}, got {n_folds}")

    train_idx, _ = train_test_indices(trial_meta)
    y_train = trial_meta.loc[train_idx, "class"].to_numpy()

    smallest_class = int(np.min(np.bincount(y_train.astype(np.int64))))
    if n_folds > smallest_class:
        raise ValueError(
            f"n_folds={n_folds} exceeds the smallest class count in session "
            f"{SESSION_TRAIN} ({smallest_class}); stratification is impossible"
        )

    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for inner_train, inner_val in splitter.split(np.zeros_like(y_train), y_train):
        folds.append((train_idx[inner_train], train_idx[inner_val]))
    return folds


def fit_transform_split(
    estimator: FitTransformer,
    X: np.ndarray,
    trial_meta: pd.DataFrame,
    y: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit `estimator` on the train rows only, then transform both halves.

    Args:
        estimator: fitted in place and left fitted, so the caller can reuse it.
        X: (n_trials, ...) array whose first axis lines up with `trial_meta`.
        trial_meta: the trial table.
        y: optional labels for supervised transformers such as CSP. Sliced to the
            train rows before being passed to `fit`.

    Returns:
        (X_train_transformed, X_test_transformed).

    Raises:
        ValueError: if `X` (or `y`) does not line up with `trial_meta` along the
            first axis.
    """
    if len(X) != len(trial_meta):
        raise ValueError(
            f"X has {len(X)} rows but trial_meta has {len(trial_meta)}; "
            "they must describe the same trials in the same order"
        )
    if y is not None and len(y) != len(trial_meta):
        raise ValueError(f"y has {len(y)} rows but trial_meta has {len(trial_meta)}")

    train_idx, test_idx = train_test_indices(trial_meta)

    estimator.fit(X[train_idx], None if y is None else y[train_idx])
    return estimator.transform(X[train_idx]), estimator.transform(X[test_idx])
