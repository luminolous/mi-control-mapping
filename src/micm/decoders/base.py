"""Decoder interface and the validation every implementation shares.

A decoder maps a batch of trials to calibrated posteriors. It never returns hard
labels: the whole point of this project is that the control mapping decides what
to do with the uncertainty, so throwing it away inside the decoder would remove
the independent variable.

`BaseDecoder` is a template. Subclasses implement `_fit` and `_predict_proba`
and inherit the input and output checks, so a new decoder cannot skip them by
forgetting to call a helper. The checks are cheap relative to fitting and they
catch the failure this project cares about: a posterior that looks like a
posterior but does not sum to one, or carries a NaN that silently propagates
into every velocity command downstream.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import numpy as np

# Rows of a posterior must sum to 1 within this tolerance. float32 accumulation
# over four classes stays far inside it, so a failure means a real bug.
POSTERIOR_SUM_TOL: float = 1e-5


@runtime_checkable
class Decoder(Protocol):
    """What the caching layer requires of a decoder."""

    name: str

    def fit(self, X: np.ndarray, y: np.ndarray) -> Decoder:
        """X: (n_trials, n_channels, n_times) float32. y: (n_trials,) int8."""
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return (n, K) float32, rows sum to 1. Never returns hard labels."""
        ...


def validate_epochs(X: np.ndarray, *, n_channels: int | None = None) -> np.ndarray:
    """Check a trial batch and return it as float64 for the numerics below.

    Args:
        X: (n_trials, n_channels, n_times).
        n_channels: expected channel count, when the decoder is already fitted.

    Raises:
        ValueError: on the wrong rank, an empty batch, a non-finite value, or a
            channel count that differs from the fitted one.
    """
    if X.ndim != 3:
        raise ValueError(f"expected (n_trials, n_channels, n_times), got shape {X.shape}")
    if X.shape[0] == 0:
        raise ValueError("empty trial batch")
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or inf")
    if n_channels is not None and X.shape[1] != n_channels:
        raise ValueError(
            f"fitted on {n_channels} channels but given {X.shape[1]}; "
            "the montage changed between fit and predict"
        )
    return np.asarray(X, dtype=np.float64)


def validate_labels(y: np.ndarray, n_trials: int) -> tuple[np.ndarray, int]:
    """Check labels and return them as int64 alongside the class count.

    Classes must be contiguous integers from zero. A gap would make the column
    index of a posterior stop matching the class index, which is the kind of
    silent permutation that survives all the way into the figures.

    Raises:
        ValueError: on the wrong rank, a length mismatch, or non-contiguous
            classes.
    """
    if y.ndim != 1:
        raise ValueError(f"expected (n_trials,) labels, got shape {y.shape}")
    if len(y) != n_trials:
        raise ValueError(f"{n_trials} trials but {len(y)} labels")

    labels = np.asarray(y, dtype=np.int64)
    present = np.unique(labels)
    expected = np.arange(len(present))
    if not np.array_equal(present, expected):
        raise ValueError(
            f"classes must be contiguous integers from 0, got {present.tolist()}; "
            "a gap would decouple posterior columns from class indices"
        )
    return labels, len(present)


def validate_posterior(p: np.ndarray, *, n_trials: int, n_classes: int) -> np.ndarray:
    """Check a posterior and return it as float32.

    Raises:
        ValueError: on the wrong shape, a non-finite value, a negative
            probability, or a row that does not sum to 1.
    """
    if p.shape != (n_trials, n_classes):
        raise ValueError(f"expected posterior shape {(n_trials, n_classes)}, got {p.shape}")
    if not np.isfinite(p).all():
        raise ValueError("posterior contains NaN or inf")
    if (p < 0.0).any():
        raise ValueError("posterior contains negative probabilities")

    sums = p.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=POSTERIOR_SUM_TOL):
        worst = float(np.max(np.abs(sums - 1.0)))
        raise ValueError(f"posterior rows must sum to 1, worst deviation {worst:.3e}")
    return np.asarray(p, dtype=np.float32)


class BaseDecoder(ABC):
    """Validation wrapper around a concrete decoder.

    Subclasses implement `_fit` and `_predict_proba`. Fitting must be
    deterministic given `seed`, must not touch global random state, and must
    leave the instance serializable with joblib.
    """

    name: str = "base"

    def __init__(self, *, seed: int) -> None:
        self.seed = int(seed)
        self.n_channels_: int | None = None
        self.n_classes_: int | None = None

    @property
    def is_fitted(self) -> bool:
        return self.n_classes_ is not None

    def fit(self, X: np.ndarray, y: np.ndarray) -> BaseDecoder:
        """Fit on (n_trials, n_channels, n_times) trials and (n_trials,) labels."""
        epochs = validate_epochs(X)
        labels, n_classes = validate_labels(y, len(epochs))

        self._fit(epochs, labels, n_classes)

        self.n_channels_ = epochs.shape[1]
        self.n_classes_ = n_classes
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return (n_trials, n_classes) float32 posteriors.

        Raises:
            RuntimeError: if called before `fit`.
        """
        if self.n_classes_ is None:
            raise RuntimeError(f"{self.name} is not fitted")

        epochs = validate_epochs(X, n_channels=self.n_channels_)
        posterior = self._predict_proba(epochs)
        return validate_posterior(
            posterior, n_trials=len(epochs), n_classes=self.n_classes_
        )

    @abstractmethod
    def _fit(self, X: np.ndarray, y: np.ndarray, n_classes: int) -> None:
        """Fit on validated float64 epochs and int64 labels."""

    @abstractmethod
    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return (n_trials, n_classes) probabilities for validated epochs."""
