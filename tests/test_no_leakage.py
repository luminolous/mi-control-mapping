"""Train/test boundary tests, written before `micm.data.splits` exists.

Until T3 lands, this whole file fails to import. That is the point: the tests
define the contract, and the implementation is written to satisfy them.

The protocol is the dataset's own. Session T is fitted on, session E is
evaluated on, per subject. Nothing fitted may ever see session E, and inner
cross-validation for hyperparameter selection stays inside session T. None of
those violations raise on their own; they surface as an implausibly good number
months later, which is why they are checked here.

Contract under test (`micm.data.splits`):

    train_test_indices(trial_meta) -> (train_idx, test_idx)
    inner_cv_indices(trial_meta, n_folds, seed) -> [(train_idx, val_idx), ...]
    fit_transform_split(estimator, X, trial_meta) -> (X_train_t, X_test_t)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from micm.data.splits import fit_transform_split, inner_cv_indices, train_test_indices

N_PER_SESSION = 24
N_CLASSES = 4
N_FEATURES = 3


def _trial_meta(sessions: tuple[str, ...] = ("T", "E")) -> pd.DataFrame:
    """Minimal stand-in for the real trial table, same columns and dtypes."""
    rows = []
    trial_id = 0
    for session in sessions:
        for i in range(N_PER_SESSION):
            rows.append(
                {
                    "trial_id": trial_id,
                    "session": session,
                    "class": i % N_CLASSES,
                    "onset_s": float(i) * 8.0,
                    "artifact": False,
                }
            )
            trial_id += 1
    frame = pd.DataFrame(rows)
    return frame.astype({"trial_id": "int32", "class": "int8", "onset_s": "float32"})


@pytest.fixture
def trial_meta() -> pd.DataFrame:
    return _trial_meta()


@pytest.fixture
def features() -> np.ndarray:
    """Distinct per-row values, so a leaked row is identifiable by its content."""
    rng = np.random.default_rng(0)
    return rng.normal(size=(2 * N_PER_SESSION, N_FEATURES)).astype(np.float32)


class RecordingScaler:
    """Estimator that remembers exactly what it was fitted on."""

    def __init__(self) -> None:
        self.fitted_on: np.ndarray | None = None
        self.n_fit_calls = 0

    def fit(self, X: np.ndarray, y: Any = None) -> RecordingScaler:
        self.fitted_on = np.array(X, copy=True)
        self.n_fit_calls += 1
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.fitted_on is None:
            raise RuntimeError("transform called before fit")
        return np.asarray(X) - self.fitted_on.mean(axis=0)


def test_train_test_indices_disjoint(trial_meta: pd.DataFrame) -> None:
    train, test = train_test_indices(trial_meta)
    assert set(train.tolist()).isdisjoint(set(test.tolist()))
    assert len(train) + len(test) == len(trial_meta)


def test_train_test_indices_cover_every_trial(trial_meta: pd.DataFrame) -> None:
    """No trial may be silently dropped; dropping is the caller's decision."""
    train, test = train_test_indices(trial_meta)
    assert set(train.tolist()) | set(test.tolist()) == set(range(len(trial_meta)))


def test_sessions_are_separated(trial_meta: pd.DataFrame) -> None:
    train, test = train_test_indices(trial_meta)
    assert set(trial_meta.loc[train, "session"]) == {"T"}
    assert set(trial_meta.loc[test, "session"]) == {"E"}


@pytest.mark.parametrize("sessions", [("T",), ("E",)])
def test_missing_session_raises(sessions: tuple[str, ...]) -> None:
    """A one-session table means the protocol cannot be honoured. Raise, do not guess."""
    with pytest.raises(ValueError):
        train_test_indices(_trial_meta(sessions))


def test_unknown_session_label_raises(trial_meta: pd.DataFrame) -> None:
    corrupted = trial_meta.copy()
    corrupted.loc[0, "session"] = "X"
    with pytest.raises(ValueError):
        train_test_indices(corrupted)


def test_normalizer_never_sees_test(trial_meta: pd.DataFrame, features: np.ndarray) -> None:
    """The single most valuable test in the file. Fit must see the train rows only."""
    scaler = RecordingScaler()
    train, _ = train_test_indices(trial_meta)

    fit_transform_split(scaler, features, trial_meta)

    assert scaler.n_fit_calls == 1
    assert scaler.fitted_on is not None
    assert scaler.fitted_on.shape[0] == len(train)
    np.testing.assert_array_equal(scaler.fitted_on, features[train])


def test_fit_transform_split_applies_train_parameters_to_test(
    trial_meta: pd.DataFrame, features: np.ndarray
) -> None:
    """Test rows are transformed with train-fitted parameters, not refitted."""
    train, test = train_test_indices(trial_meta)
    x_train_t, x_test_t = fit_transform_split(RecordingScaler(), features, trial_meta)

    expected_offset = features[train].mean(axis=0)
    np.testing.assert_allclose(x_train_t, features[train] - expected_offset, rtol=1e-6)
    np.testing.assert_allclose(x_test_t, features[test] - expected_offset, rtol=1e-6)


def test_fit_transform_split_rejects_length_mismatch(trial_meta: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        fit_transform_split(RecordingScaler(), np.zeros((7, N_FEATURES), np.float32), trial_meta)


def test_inner_cv_stays_within_train(trial_meta: pd.DataFrame) -> None:
    """Hyperparameter selection must never reach session E."""
    train, _ = train_test_indices(trial_meta)
    train_set = set(train.tolist())

    folds = inner_cv_indices(trial_meta, n_folds=4, seed=0)
    assert len(folds) == 4
    for inner_train, inner_val in folds:
        assert set(inner_train.tolist()) <= train_set
        assert set(inner_val.tolist()) <= train_set


def test_inner_cv_folds_are_a_partition(trial_meta: pd.DataFrame) -> None:
    train, _ = train_test_indices(trial_meta)
    folds = inner_cv_indices(trial_meta, n_folds=4, seed=0)

    seen: list[int] = []
    for inner_train, inner_val in folds:
        assert set(inner_train.tolist()).isdisjoint(set(inner_val.tolist()))
        assert set(inner_train.tolist()) | set(inner_val.tolist()) == set(train.tolist())
        seen.extend(inner_val.tolist())

    assert sorted(seen) == sorted(train.tolist())


def test_inner_cv_is_deterministic_under_a_fixed_seed(trial_meta: pd.DataFrame) -> None:
    first = inner_cv_indices(trial_meta, n_folds=4, seed=0)
    second = inner_cv_indices(trial_meta, n_folds=4, seed=0)
    for (a_train, a_val), (b_train, b_val) in zip(first, second, strict=True):
        np.testing.assert_array_equal(a_train, b_train)
        np.testing.assert_array_equal(a_val, b_val)


def test_inner_cv_rejects_too_few_folds(trial_meta: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        inner_cv_indices(trial_meta, n_folds=1, seed=0)
