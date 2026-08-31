"""Synthesising decoder quality by perturbing cached posteriors.

Operates on posteriors only and never touches EEG. That is what makes the
lambda sweep affordable: every quality level in the experiment matrix reads the
same cached `.npz` instead of retraining a decoder.

Every perturbation reports the effective accuracy it actually achieved rather
than the one it was aiming for. The structured-error variants exist to compare
error *shapes* at matched accuracy, and a matched-accuracy condition that was
assumed rather than measured is not a matched-accuracy condition.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from micm.utils.logging import get_logger

logger = get_logger(__name__)

# Rows must still sum to one after every perturbation.
ROW_SUM_TOL: Final[float] = 1e-5

ERROR_NONE: Final[str] = "none"
ERROR_SMOOTHING: Final[str] = "smoothing"
ERROR_JITTER: Final[str] = "jitter"
ERROR_BURST: Final[str] = "burst"
ERROR_STRUCTURES: Final[tuple[str, ...]] = (
    ERROR_NONE,
    ERROR_SMOOTHING,
    ERROR_JITTER,
    ERROR_BURST,
)


@dataclass(frozen=True)
class PerturbResult:
    """A perturbed posterior and the accuracy it turned out to have.

    Attributes:
        posterior: (n_windows, K) float32, rows sum to 1.
        effective_accuracy: fraction of windows whose argmax is the true class.
    """

    posterior: np.ndarray
    effective_accuracy: float


def _validate_posterior(p: np.ndarray) -> np.ndarray:
    if p.ndim != 2:
        raise ValueError(f"expected (n_windows, K), got shape {p.shape}")
    if not np.isfinite(p).all():
        raise ValueError("posterior contains NaN or inf")
    if (p < 0.0).any():
        raise ValueError("posterior contains negative probabilities")
    if not np.allclose(p.sum(axis=1), 1.0, atol=ROW_SUM_TOL):
        worst = float(np.max(np.abs(p.sum(axis=1) - 1.0)))
        raise ValueError(f"posterior rows must sum to 1, worst deviation {worst:.3e}")
    return np.asarray(p, dtype=np.float64)


def _validate_labels(labels: np.ndarray, n_windows: int, n_classes: int) -> np.ndarray:
    if labels.shape != (n_windows,):
        raise ValueError(f"expected {(n_windows,)} labels, got {labels.shape}")
    values = np.asarray(labels, dtype=np.int64)
    if values.size and (values.min() < 0 or values.max() >= n_classes):
        raise ValueError(f"labels must lie in [0, {n_classes})")
    return values


def _finish(p: np.ndarray, labels: np.ndarray) -> PerturbResult:
    """Assert the row sums survived, then measure what the perturbation achieved.

    Accuracy is measured on the float32 array that is handed out, not on the
    float64 intermediate. Oracle mixing can land two classes on exactly equal
    probability, and rounding to float32 breaks that tie differently, so
    measuring the intermediate would report an accuracy the caller never sees.
    """
    if not np.allclose(p.sum(axis=1), 1.0, atol=ROW_SUM_TOL):
        worst = float(np.max(np.abs(p.sum(axis=1) - 1.0)))
        raise AssertionError(f"perturbation broke the row sums, worst deviation {worst:.3e}")
    narrowed = p.astype(np.float32)
    return PerturbResult(
        posterior=narrowed,
        effective_accuracy=effective_accuracy(narrowed, labels),
    )


def effective_accuracy(p: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of windows whose most probable class is the true one."""
    if len(p) == 0:
        return float("nan")
    return float((np.argmax(p, axis=1) == np.asarray(labels)).mean())


def oracle_mixing(p: np.ndarray, labels: np.ndarray, lam: float) -> PerturbResult:
    """Blend the posterior toward ground truth.

        p' = lam * onehot(y) + (1 - lam) * p

    `lam = 0` leaves the decoder untouched, `lam = 1` makes it perfect. This is
    the primary decoder-quality axis of the experiment, and it is continuous,
    which is why the three real decoders are only a secondary axis.

    Args:
        p: (n_windows, K) float32, rows sum to 1.
        labels: (n_windows,) int in [0, K).
        lam: in [0, 1].

    Returns:
        (n_windows, K) float32 with rows summing to 1, plus its accuracy.

    Raises:
        ValueError: on an invalid posterior, mismatched labels, or `lam` outside
            [0, 1].
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1], got {lam}")

    values = _validate_posterior(p)
    y = _validate_labels(labels, len(values), values.shape[1])

    onehot = np.zeros_like(values)
    onehot[np.arange(len(y)), y] = 1.0
    return _finish(lam * onehot + (1.0 - lam) * values, y)


def lambda_for_accuracy(
    p: np.ndarray,
    labels: np.ndarray,
    target: float,
    *,
    tol: float = 1e-3,
    max_iter: int = 60,
) -> float:
    """Smallest `lam` whose oracle mixing reaches `target` effective accuracy.

    Effective accuracy is non-decreasing in `lam`, so a bisection is valid. It is
    also a step function of `lam`, one step per window, so the returned value is
    the boundary of a step rather than an exact solution: the achieved accuracy
    is at least `target` but may exceed it. Callers must report the achieved
    figure, not the requested one.

    Raises:
        ValueError: if `target` is outside [0, 1] or is below what `lam = 0`
            already gives.
    """
    if not 0.0 <= target <= 1.0:
        raise ValueError(f"target must be in [0, 1], got {target}")

    baseline = oracle_mixing(p, labels, 0.0).effective_accuracy
    if target <= baseline:
        return 0.0

    low, high = 0.0, 1.0
    for _ in range(max_iter):
        if high - low < tol:
            break
        mid = 0.5 * (low + high)
        if oracle_mixing(p, labels, mid).effective_accuracy >= target:
            high = mid
        else:
            low = mid
    return high


def label_smoothing(p: np.ndarray, labels: np.ndarray, eps: float) -> PerturbResult:
    """Flatten the posterior toward uniform without moving its argmax.

        p' = (1 - eps) * p + eps / K

    Confidence drops, the ranking does not, so effective accuracy is unchanged
    while a mapping that reads magnitude, such as S2, sees a weaker command. That
    separation is the point: it isolates confidence from correctness.

    Raises:
        ValueError: on an invalid posterior or `eps` outside [0, 1].
    """
    if not 0.0 <= eps <= 1.0:
        raise ValueError(f"eps must be in [0, 1], got {eps}")

    values = _validate_posterior(p)
    y = _validate_labels(labels, len(values), values.shape[1])
    n_classes = values.shape[1]
    return _finish((1.0 - eps) * values + eps / n_classes, y)


def temporal_jitter(
    p: np.ndarray, labels: np.ndarray, burst_id: np.ndarray, shift_windows: int
) -> PerturbResult:
    """Delay the posterior sequence inside each burst, so the right class lands late.

    Shifting within a burst rather than across the whole session keeps the
    perturbation from leaking one burst's evidence into the next, which would
    change the error rate as well as its timing. Vacated positions at the start
    of a burst are filled with that burst's first posterior, which is the
    stalest estimate genuinely available at that moment.

    A negative `shift_windows` moves the sequence earlier, which is not
    physically meaningful but is useful as a control.

    Raises:
        ValueError: on an invalid posterior or a mismatched `burst_id`.
    """
    values = _validate_posterior(p)
    y = _validate_labels(labels, len(values), values.shape[1])
    if burst_id.shape != (len(values),):
        raise ValueError(f"expected {(len(values),)} burst ids, got {burst_id.shape}")

    out = values.copy()
    for burst in np.unique(burst_id):
        rows = np.flatnonzero(burst_id == burst)
        block = values[rows]
        source = np.clip(np.arange(len(rows)) - shift_windows, 0, len(rows) - 1)
        out[rows] = block[source]
    return _finish(out, y)


def burst_error(
    p: np.ndarray,
    labels: np.ndarray,
    burst_id: np.ndarray,
    rng: np.random.Generator,
    *,
    p_flip: float,
    mean_len: float,
) -> PerturbResult:
    """Replace correlated stretches of windows with a confident wrong class.

    Real decoder failures arrive in runs, not independently: a subject loses the
    imagery for a second or two and every window in that second is wrong
    together. An i.i.d. error model at the same accuracy is much easier for an
    evidence-accumulating mapping to average away, so comparing the two is how
    S3's advantage gets tested honestly.

    Stretch lengths are geometric with mean `mean_len`, starting at rate
    `p_flip` per window, and never cross a burst boundary.

    Args:
        p_flip: probability per window of starting a wrong stretch.
        mean_len: mean stretch length in windows.

    Raises:
        ValueError: on an invalid posterior, `p_flip` outside [0, 1], or a
            non-positive `mean_len`.
    """
    if not 0.0 <= p_flip <= 1.0:
        raise ValueError(f"p_flip must be in [0, 1], got {p_flip}")
    if mean_len <= 0.0:
        raise ValueError(f"mean_len must be positive, got {mean_len}")

    values = _validate_posterior(p)
    y = _validate_labels(labels, len(values), values.shape[1])
    if burst_id.shape != (len(values),):
        raise ValueError(f"expected {(len(values),)} burst ids, got {burst_id.shape}")

    n_classes = values.shape[1]
    out = values.copy()

    for burst in np.unique(burst_id):
        rows = np.flatnonzero(burst_id == burst)
        position = 0
        while position < len(rows):
            if rng.random() >= p_flip:
                position += 1
                continue

            length = max(1, int(rng.geometric(1.0 / mean_len)))
            stop = min(position + length, len(rows))
            affected = rows[position:stop]

            truth = int(y[affected[0]])
            wrong = int(rng.choice([k for k in range(n_classes) if k != truth]))
            # One wrong class held for the whole stretch, matching the confidence
            # the untouched posterior had, so accuracy moves but sharpness does not.
            confidence = values[affected].max(axis=1, keepdims=True)
            replacement = (1.0 - confidence) / (n_classes - 1) * np.ones(n_classes)
            replacement[:, wrong] = confidence[:, 0]
            out[affected] = replacement
            position = stop

    return _finish(out, y)


QUALITY_GAP_FRACTION: Final[str] = "gap_fraction"
QUALITY_ABSOLUTE: Final[str] = "absolute"
QUALITY_MODES: Final[tuple[str, str]] = (QUALITY_GAP_FRACTION, QUALITY_ABSOLUTE)


@dataclass(frozen=True)
class QualityLevel:
    """One rung of the decoder-quality sweep, resolved for one cell.

    Attributes:
        level: the requested level as written in the config.
        target_accuracy: the effective accuracy that level asks for.
        lam: the oracle-mixing weight that reaches it.
        achieved_accuracy: what that `lam` actually gives. Effective accuracy is
            a step function of `lam`, one step per window, so this is at least
            the target and may exceed it. Report this, never the target.
    """

    level: float
    target_accuracy: float
    lam: float
    achieved_accuracy: float


def resolve_quality_levels(
    p: np.ndarray,
    labels: np.ndarray,
    levels: Sequence[float],
    *,
    mode: str = QUALITY_GAP_FRACTION,
) -> list[QualityLevel]:
    """Turn a sweep specification into the mixing weights for one cell.

    A fixed grid of `lam` values does not place comparable conditions. How much
    accuracy a given `lam` buys depends on how peaked the decoder's posteriors
    are, which differs by subject and by decoder, and oracle mixing saturates
    well before `lam = 1`. On subject 1 with the Riemannian decoder it saturates
    at 0.498, so half of the grid in `agents/05` §1 is the same perfect decoder.
    See docs/decisions.md D27a.

    Two modes:

    - `gap_fraction` (default): `target = baseline + level * (1 - baseline)`, so
      level 0 is the untouched decoder and level 1 is the oracle. Every cell
      spends its episodes on distinct conditions regardless of how good its
      decoder is.
    - `absolute`: `target = level`. Conditions sit at the same accuracy for
      every cell, at the cost of collapsing to `lam = 0` for any cell whose
      decoder is already better than the target.

    Note that oracle mixing only ever raises accuracy. Neither mode can place a
    cell below its own baseline; degrading a strong decoder needs
    `burst_error` or `label_smoothing`, not this.

    Raises:
        ValueError: on an unknown mode, or a level outside [0, 1].
    """
    if mode not in QUALITY_MODES:
        raise ValueError(f"unknown quality mode {mode!r}, expected one of {list(QUALITY_MODES)}")
    if any(not 0.0 <= level <= 1.0 for level in levels):
        raise ValueError(f"levels must lie in [0, 1], got {list(levels)}")

    baseline = effective_accuracy(p, labels)
    resolved: list[QualityLevel] = []
    for level in levels:
        target = (
            baseline + level * (1.0 - baseline) if mode == QUALITY_GAP_FRACTION else float(level)
        )
        lam = lambda_for_accuracy(p, labels, target)
        resolved.append(
            QualityLevel(
                level=float(level),
                target_accuracy=float(target),
                lam=lam,
                achieved_accuracy=oracle_mixing(p, labels, lam).effective_accuracy,
            )
        )
    return resolved
