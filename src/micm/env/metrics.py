"""Episode metrics.

One pure function per metric, each taking arrays or outcome records and
returning a number. No state, so every one of them can be checked against a
case computed by hand, which is what `tests/test_metrics.py` does.

`user_contribution_index` is the metric that tests H3. Under high autonomy the
robot reaches targets whether or not the user contributed anything, so without
UCI "shared control works" is indistinguishable from "the robot did it alone".
Read the note on the estimator before changing it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from micm.env.task import TargetOutcome
from micm.utils.logging import get_logger

logger = get_logger(__name__)

# A direction change beyond this counts as a reversal.
REVERSAL_THRESHOLD_DEG: Final[float] = 90.0

# Velocities below this are treated as no motion, so numerical dust around zero
# does not register as a direction.
MOTION_EPS: Final[float] = 1e-9


@dataclass(frozen=True)
class EpisodeMetrics:
    """Everything one episode contributes to `episodes.parquet`."""

    success_rate: float
    n_success: int
    n_timeouts: int
    time_to_target_median: float
    time_to_target_excluded: int
    path_efficiency: float
    direction_reversals: float
    collisions: int
    effective_itr: float
    user_contribution_index: float
    uci_excluded_frac: float
    episode_duration_s: float


def n_success(outcomes: Sequence[TargetOutcome]) -> int:
    return sum(1 for outcome in outcomes if outcome.acquired)


def success_rate(outcomes: Sequence[TargetOutcome], n_targets: int) -> float:
    """Acquired targets over the number presented.

    The denominator is the configured target count rather than `len(outcomes)`,
    so an episode that ended early scores against what it was asked to do.

    Raises:
        ValueError: on a non-positive `n_targets`.
    """
    if n_targets <= 0:
        raise ValueError(f"n_targets must be positive, got {n_targets}")
    return n_success(outcomes) / n_targets


def n_timeouts(outcomes: Sequence[TargetOutcome]) -> int:
    return sum(1 for outcome in outcomes if not outcome.acquired)


def time_to_target(outcomes: Sequence[TargetOutcome]) -> tuple[float, int]:
    """Median seconds per acquired target, and how many attempts were excluded.

    Timeouts are excluded rather than counted at the timeout value, which would
    make the metric a function of the timeout setting. The exclusion count is
    returned alongside so a median over two survivors is not mistaken for a
    median over eight.

    Returns:
        (median seconds, number of excluded attempts). The median is NaN when no
        target was acquired.
    """
    acquired = [outcome.duration_s for outcome in outcomes if outcome.acquired]
    excluded = len(outcomes) - len(acquired)
    if not acquired:
        return float("nan"), excluded
    return float(np.median(acquired)), excluded


def path_efficiency(outcomes: Sequence[TargetOutcome]) -> float:
    """Straight-line distance over distance travelled, averaged across targets.

    Computed over acquired targets only. For a target that timed out the robot
    never arrived, so the straight-line distance is not a distance it covered and
    the ratio can exceed one, which would make the metric report a trajectory as
    better than optimal. Excluding those keeps the value in (0, 1].

    `straight_line` is the distance to the target boundary rather than its
    centre, because acquisition needs the robot inside the radius and not at the
    middle. Measured to the centre, a robot that stopped just inside the edge
    scores above 1.

    A target already satisfied without travelling scores 1: there was no shorter
    path available.

    Returns NaN when nothing was acquired.
    """
    ratios = [
        1.0 if outcome.straight_line == 0.0 else outcome.straight_line / outcome.path_length
        for outcome in outcomes
        if outcome.acquired and outcome.path_length > 0.0
    ]
    return float(np.mean(ratios)) if ratios else float("nan")


def collisions(outcomes: Sequence[TargetOutcome]) -> int:
    return sum(outcome.collisions for outcome in outcomes)


def direction_reversals(
    vel: np.ndarray,
    target_idx: np.ndarray,
    *,
    threshold_deg: float = REVERSAL_THRESHOLD_DEG,
) -> float:
    """Mean number of sharp direction changes per target.

    A reversal is a change of more than `threshold_deg` between the current
    heading and the last heading the robot actually had. Comparing against the
    last *moving* heading rather than the previous sample matters: the command is
    held between decoder updates, so consecutive samples are usually identical
    and a sample-to-sample comparison would report nothing.

    Args:
        vel: (n_steps, 2) velocities.
        target_idx: (n_steps,) which target was active at each step.

    Returns:
        Reversals per target, averaged. Zero for an empty trace.

    Raises:
        ValueError: on mismatched lengths or a threshold outside (0, 180).
    """
    if not 0.0 < threshold_deg < 180.0:
        raise ValueError(f"threshold_deg must be in (0, 180), got {threshold_deg}")
    velocities = np.asarray(vel, dtype=np.float64).reshape(-1, 2)
    targets = np.asarray(target_idx).reshape(-1)
    if len(velocities) != len(targets):
        raise ValueError(f"{len(velocities)} velocities but {len(targets)} target indices")
    if len(velocities) == 0:
        return 0.0

    cos_threshold = float(np.cos(np.deg2rad(threshold_deg)))
    counts: dict[int, int] = {}
    previous: dict[int, np.ndarray] = {}

    speeds = np.hypot(velocities[:, 0], velocities[:, 1])
    for index in range(len(velocities)):
        target = int(targets[index])
        counts.setdefault(target, 0)
        if speeds[index] <= MOTION_EPS:
            continue

        heading = velocities[index] / speeds[index]
        last = previous.get(target)
        if last is not None and float(last @ heading) < cos_threshold:
            counts[target] += 1
        previous[target] = heading

    return float(np.mean(list(counts.values()))) if counts else 0.0


def effective_itr(rate: float, n_targets: int, duration_s: float) -> float:
    """Wolpaw information transfer rate in bits per minute.

    Each target attempt is one selection from `n_targets` possibilities, correct
    with probability `rate`; a timeout counts as incorrect. Targets are presented
    one at a time, so there is no confusion matrix to estimate from and this
    binary form is the honest reading. See docs/decisions.md D2.

        B = log2(N) + P log2 P + (1 - P) log2((1 - P) / (N - 1))

    Bits per selection are clamped at zero. Below chance the formula turns upward
    again, which would report a decoder that is reliably wrong as informative;
    that is true of the underlying information measure but is not what an ITR is
    used to say.

    Returns NaN for a zero-length episode.

    Raises:
        ValueError: on `rate` outside [0, 1], fewer than two targets, or a
            negative duration.
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"rate must be in [0, 1], got {rate}")
    if n_targets < 2:
        raise ValueError(f"n_targets must be at least 2, got {n_targets}")
    if duration_s < 0.0:
        raise ValueError(f"duration_s must be non-negative, got {duration_s}")
    if duration_s == 0.0:
        return float("nan")

    if rate <= 1.0 / n_targets:
        bits = 0.0
    elif rate == 1.0:
        bits = float(np.log2(n_targets))
    else:
        bits = float(
            np.log2(n_targets)
            + rate * np.log2(rate)
            + (1.0 - rate) * np.log2((1.0 - rate) / (n_targets - 1))
        )

    selections_per_minute = n_targets / duration_s * 60.0
    return max(bits, 0.0) * selections_per_minute


def _unit_directions(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit vectors and a mask of which rows had a direction at all."""
    array = np.asarray(vectors, dtype=np.float64).reshape(-1, 2)
    norms = np.hypot(array[:, 0], array[:, 1])
    usable = norms > MOTION_EPS
    units = np.zeros_like(array)
    units[usable] = array[usable] / norms[usable, None]
    return units, usable


def user_contribution_index(
    decoded_intent: np.ndarray,
    realized: np.ndarray,
    sample_mask: np.ndarray | None = None,
) -> tuple[float, float]:
    """How closely realised motion follows the decoded intent.

    The metric that tests H3. Under high autonomy the robot reaches targets
    regardless of what the user decoded, and success rate alone cannot tell that
    apart from genuine shared control. UCI is expected to collapse toward zero as
    alpha rises while success rate stays high.

    **Estimator.** The mean cosine of the angle between the two directions:

        UCI = mean(cos(theta_intent - theta_realized))

    Perfect alignment gives 1, orthogonal gives 0, anti-alignment gives -1.

    This is deliberately *not* the Jammalamadaka circular correlation, which is
    the usual textbook choice. That coefficient is invariant to a constant
    angular offset, so a robot moving in exactly the opposite direction to the
    decoded intent scores +1 rather than -1. Invariance to offset is the wrong
    property here: the question is whether the robot went where the user pointed,
    not whether the two rotate together.

    **Exclusions.** A sample counts only when both directions exist. A zero
    command has no direction, which happens during a gap, before the latency
    buffer fills, and while S3 is below threshold; a zero decoded intent has none
    either. `sample_mask` restricts the estimate further, and the caller passes
    the decoder-update steps: at 100 Hz against a 4 Hz decoder, every posterior
    would otherwise be counted about 25 times.

    Args:
        decoded_intent: (n, 2) direction implied by the posterior.
        realized: (n, 2) direction the robot actually moved.
        sample_mask: (n,) bool, which samples to consider at all.

    Returns:
        (uci, excluded_fraction). `uci` is NaN when nothing survives the
        exclusions, and `excluded_fraction` is then 1.0.

    Raises:
        ValueError: on mismatched lengths.
    """
    intent_units, intent_ok = _unit_directions(decoded_intent)
    realized_units, realized_ok = _unit_directions(realized)
    if len(intent_units) != len(realized_units):
        raise ValueError(f"{len(intent_units)} intents but {len(realized_units)} realised")

    considered = np.ones(len(intent_units), dtype=bool)
    if sample_mask is not None:
        considered = np.asarray(sample_mask, dtype=bool).reshape(-1)
        if len(considered) != len(intent_units):
            raise ValueError(f"{len(intent_units)} samples but a mask of {len(considered)}")

    n_considered = int(considered.sum())
    if n_considered == 0:
        return float("nan"), 1.0

    usable = considered & intent_ok & realized_ok
    n_usable = int(usable.sum())
    excluded_fraction = 1.0 - n_usable / n_considered
    if n_usable == 0:
        return float("nan"), 1.0

    cosines = np.einsum("ij,ij->i", intent_units[usable], realized_units[usable])
    return float(np.mean(cosines)), float(excluded_fraction)


def summarize(
    outcomes: Sequence[TargetOutcome],
    arrays: dict[str, np.ndarray],
    *,
    n_targets: int,
    dt: float,
) -> EpisodeMetrics:
    """Every metric for one episode, from its outcome records and its trace.

    Args:
        outcomes: one record per target attempt.
        arrays: the output of `EpisodeTrace.arrays()`.
        n_targets: how many targets the episode presented.
        dt: environment step in seconds, used for the episode duration.
    """
    duration = len(arrays["pos"]) * dt
    rate = success_rate(outcomes, n_targets)
    median, excluded = time_to_target(outcomes)
    uci, uci_excluded = user_contribution_index(
        arrays["decoded_intent"],
        arrays["vel"],
        sample_mask=arrays["decoder_update"] & arrays["has_command"],
    )

    return EpisodeMetrics(
        success_rate=rate,
        n_success=n_success(outcomes),
        n_timeouts=n_timeouts(outcomes),
        time_to_target_median=median,
        time_to_target_excluded=excluded,
        path_efficiency=path_efficiency(outcomes),
        direction_reversals=direction_reversals(arrays["vel"], arrays["target_idx"]),
        collisions=collisions(outcomes),
        effective_itr=effective_itr(rate, n_targets, duration),
        user_contribution_index=uci,
        uci_excluded_frac=uci_excluded,
        episode_duration_s=duration,
    )
