"""Metric tests, every one against a value worked out by hand.

A metric that is only checked for being finite and in range is a metric that can
be wrong by a constant factor for the life of the project. Each case below has
an answer derived independently of the implementation: a semicircular detour is
2/pi efficient, a perfect selection out of eight targets carries exactly 3 bits,
anti-aligned motion scores exactly -1.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from micm.env.metrics import (
    EpisodeMetrics,
    collisions,
    direction_reversals,
    effective_itr,
    n_success,
    n_timeouts,
    path_efficiency,
    success_rate,
    summarize,
    time_to_target,
    user_contribution_index,
)
from micm.env.task import TargetOutcome

N_TARGETS = 8


def outcome(
    *,
    acquired: bool = True,
    duration_s: float = 10.0,
    path_length: float = 1.0,
    straight_line: float = 1.0,
    collisions_count: int = 0,
    target_idx: int = 0,
) -> TargetOutcome:
    return TargetOutcome(
        target_idx=target_idx,
        acquired=acquired,
        duration_s=duration_s,
        path_length=path_length,
        straight_line=straight_line,
        collisions=collisions_count,
    )


# --- success and timing ---


def test_success_rate_is_a_proportion_of_the_targets_presented() -> None:
    outcomes = [outcome(acquired=i < 6) for i in range(N_TARGETS)]
    assert success_rate(outcomes, N_TARGETS) == pytest.approx(6 / 8)
    assert n_success(outcomes) == 6
    assert n_timeouts(outcomes) == 2


def test_success_rate_scores_a_short_episode_against_what_was_asked() -> None:
    """The denominator is the configured count, not the number of attempts made."""
    assert success_rate([outcome(), outcome()], N_TARGETS) == pytest.approx(2 / 8)


def test_success_rate_rejects_a_zero_denominator() -> None:
    with pytest.raises(ValueError, match="n_targets"):
        success_rate([], 0)


def test_time_to_target_excludes_timeouts_and_reports_how_many() -> None:
    """Counting a timeout at its timeout value would make this a function of the config."""
    outcomes = [
        outcome(duration_s=10.0),
        outcome(duration_s=20.0),
        outcome(duration_s=30.0),
        outcome(acquired=False, duration_s=120.0),
    ]
    median, excluded = time_to_target(outcomes)
    assert median == pytest.approx(20.0)
    assert excluded == 1


def test_time_to_target_is_nan_when_nothing_was_acquired() -> None:
    median, excluded = time_to_target([outcome(acquired=False) for _ in range(N_TARGETS)])
    assert np.isnan(median)
    assert excluded == N_TARGETS


# --- path efficiency ---


def test_a_straight_line_is_perfectly_efficient() -> None:
    assert path_efficiency([outcome(path_length=1.0, straight_line=1.0)]) == pytest.approx(1.0)


def test_a_semicircular_detour_scores_two_over_pi() -> None:
    """Diameter 2r against an arc of pi*r. The classic hand-computed case."""
    radius = 0.5
    result = path_efficiency(
        [outcome(path_length=np.pi * radius, straight_line=2.0 * radius)]
    )
    assert result == pytest.approx(2.0 / np.pi)


def test_path_efficiency_ignores_targets_that_timed_out() -> None:
    """The robot never arrived, so its straight-line distance is not distance covered.

    Including it lets the ratio exceed one, which would report a trajectory as
    better than optimal.
    """
    outcomes = [
        outcome(path_length=2.0, straight_line=1.0),
        outcome(acquired=False, path_length=0.1, straight_line=1.0),
    ]
    assert path_efficiency(outcomes) == pytest.approx(0.5)


def test_path_efficiency_is_nan_when_nothing_was_acquired() -> None:
    assert np.isnan(path_efficiency([outcome(acquired=False)]))


def test_collisions_sum_across_targets() -> None:
    assert collisions([outcome(collisions_count=2), outcome(collisions_count=3)]) == 5


# --- direction reversals ---


def _zigzag(directions: list[tuple[float, float]], hold: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Hold each direction for `hold` steps, as a zero-order hold would."""
    vel = np.repeat(np.asarray(directions, dtype=np.float64), hold, axis=0)
    return vel, np.zeros(len(vel), dtype=np.int32)


def test_a_hand_built_zigzag_gives_the_count_counted_by_hand() -> None:
    """Right, left, right, left: three reversals, each a 180 degree turn."""
    vel, targets = _zigzag([(1.0, 0.0), (-1.0, 0.0), (1.0, 0.0), (-1.0, 0.0)])
    assert direction_reversals(vel, targets) == pytest.approx(3.0)


def test_a_held_command_is_not_counted_repeatedly() -> None:
    """The command is held between decoder updates, so identical samples must not count."""
    vel, targets = _zigzag([(1.0, 0.0)], hold=100)
    assert direction_reversals(vel, targets) == 0.0


def test_a_gentle_turn_is_below_the_threshold() -> None:
    """45 degrees is a course correction, not a reversal."""
    vel, targets = _zigzag([(1.0, 0.0), (1.0, 1.0)])
    assert direction_reversals(vel, targets) == 0.0


def test_a_turn_just_past_the_threshold_counts() -> None:
    vel, targets = _zigzag([(1.0, 0.0), (-1.0, 0.1)])
    assert direction_reversals(vel, targets) == pytest.approx(1.0)


def test_reversals_are_counted_per_target_then_averaged() -> None:
    vel = np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    targets = np.array([0, 0, 1, 1], dtype=np.int32)
    # Two reversals on target 0, none on target 1.
    assert direction_reversals(vel, targets) == pytest.approx(0.5)


def test_a_stationary_robot_has_no_reversals() -> None:
    """Numerical dust around zero must not register as a direction."""
    vel = np.zeros((50, 2))
    assert direction_reversals(vel, np.zeros(50, dtype=np.int32)) == 0.0


def test_reversals_reject_a_bad_threshold() -> None:
    vel, targets = _zigzag([(1.0, 0.0)])
    with pytest.raises(ValueError, match="threshold_deg"):
        direction_reversals(vel, targets, threshold_deg=200.0)


# --- information transfer rate ---


def test_perfect_performance_matches_the_analytic_wolpaw_value() -> None:
    """Eight equally likely targets, always correct: exactly log2(8) = 3 bits."""
    duration = 60.0
    rate = effective_itr(1.0, N_TARGETS, duration)
    assert rate == pytest.approx(3.0 * N_TARGETS / duration * 60.0)


def test_chance_performance_carries_no_information() -> None:
    assert effective_itr(1.0 / N_TARGETS, N_TARGETS, 60.0) == pytest.approx(0.0)


def test_below_chance_is_clamped_rather_than_reported_as_informative() -> None:
    """The formula turns upward below chance; an ITR is not used to say that."""
    assert effective_itr(0.0, N_TARGETS, 60.0) == 0.0


def test_itr_falls_as_the_episode_takes_longer() -> None:
    fast = effective_itr(0.75, N_TARGETS, 60.0)
    slow = effective_itr(0.75, N_TARGETS, 120.0)
    assert fast == pytest.approx(2.0 * slow)


def test_itr_is_nan_for_a_zero_length_episode() -> None:
    assert np.isnan(effective_itr(1.0, N_TARGETS, 0.0))


@pytest.mark.parametrize(
    ("rate", "n_targets", "duration"), [(1.5, 8, 60.0), (0.5, 1, 60.0), (0.5, 8, -1.0)]
)
def test_itr_rejects_impossible_arguments(
    rate: float, n_targets: int, duration: float
) -> None:
    with pytest.raises(ValueError):
        effective_itr(rate, n_targets, duration)


# --- user contribution index ---


def _directions(angles_deg: list[float]) -> np.ndarray:
    radians = np.deg2rad(angles_deg)
    return np.stack([np.cos(radians), np.sin(radians)], axis=1)


def test_perfectly_aligned_motion_scores_one() -> None:
    intent = _directions([0.0, 45.0, 90.0, 180.0])
    uci, excluded = user_contribution_index(intent, intent.copy())
    assert uci == pytest.approx(1.0)
    assert excluded == pytest.approx(0.0)


def test_orthogonal_motion_scores_zero() -> None:
    angles = [0.0, 45.0, 90.0, 180.0]
    uci, _ = user_contribution_index(
        _directions(angles), _directions([a + 90.0 for a in angles])
    )
    assert uci == pytest.approx(0.0, abs=1e-9)


def test_anti_aligned_motion_scores_minus_one() -> None:
    """The property the Jammalamadaka circular correlation would get wrong.

    That coefficient is invariant to a constant angular offset, so a robot moving
    in exactly the opposite direction to the decoded intent would score +1. The
    question UCI answers is whether the robot went where the user pointed.
    """
    angles = [0.0, 45.0, 90.0, 180.0]
    uci, _ = user_contribution_index(
        _directions(angles), _directions([a + 180.0 for a in angles])
    )
    assert uci == pytest.approx(-1.0)


def test_a_constant_angular_offset_is_penalised() -> None:
    """Rotating together is not the same as following, and must not score 1."""
    angles = [0.0, 30.0, 60.0, 120.0]
    uci, _ = user_contribution_index(
        _directions(angles), _directions([a + 60.0 for a in angles])
    )
    assert uci == pytest.approx(np.cos(np.deg2rad(60.0)))


def test_magnitude_does_not_matter_only_direction() -> None:
    intent = _directions([0.0, 90.0])
    uci, _ = user_contribution_index(intent, intent * np.array([[0.001], [1000.0]]))
    assert uci == pytest.approx(1.0)


def test_all_zero_commands_give_nan_and_a_full_exclusion() -> None:
    """A zero vector has no direction, so there is nothing to correlate."""
    uci, excluded = user_contribution_index(_directions([0.0, 90.0]), np.zeros((2, 2)))
    assert np.isnan(uci)
    assert excluded == pytest.approx(1.0)


def test_zero_command_samples_are_excluded_and_counted() -> None:
    intent = _directions([0.0, 0.0, 0.0, 0.0])
    realized = intent.copy()
    realized[2:] = 0.0
    uci, excluded = user_contribution_index(intent, realized)
    assert uci == pytest.approx(1.0)
    assert excluded == pytest.approx(0.5)


def test_a_sample_mask_restricts_the_estimate() -> None:
    """The caller passes decoder-update steps, or each posterior counts about 25 times."""
    intent = _directions([0.0, 0.0, 0.0, 0.0])
    realized = _directions([0.0, 180.0, 0.0, 180.0])
    mask = np.array([True, False, True, False])
    uci, excluded = user_contribution_index(intent, realized, sample_mask=mask)
    assert uci == pytest.approx(1.0)
    assert excluded == pytest.approx(0.0)


def test_an_empty_mask_gives_nan() -> None:
    intent = _directions([0.0, 90.0])
    uci, excluded = user_contribution_index(intent, intent, sample_mask=np.zeros(2, bool))
    assert np.isnan(uci)
    assert excluded == pytest.approx(1.0)


def test_uci_rejects_a_mismatched_mask() -> None:
    intent = _directions([0.0, 90.0])
    with pytest.raises(ValueError, match="mask"):
        user_contribution_index(intent, intent, sample_mask=np.ones(5, bool))


# --- properties ---


@settings(max_examples=50, deadline=None)
@given(
    flags=st.lists(st.booleans(), min_size=1, max_size=8),
    lengths=st.lists(st.floats(0.1, 10.0), min_size=8, max_size=8),
)
def test_success_rate_and_counts_stay_in_range(
    flags: list[bool], lengths: list[float]
) -> None:
    outcomes = [
        outcome(acquired=flag, path_length=length, straight_line=length * 0.5)
        for flag, length in zip(flags, lengths, strict=False)
    ]
    rate = success_rate(outcomes, N_TARGETS)
    assert 0.0 <= rate <= 1.0
    assert n_success(outcomes) <= N_TARGETS


@settings(max_examples=50, deadline=None)
@given(
    straight=st.floats(0.1, 5.0),
    extra=st.floats(0.0, 5.0),
)
def test_path_efficiency_never_exceeds_one_for_an_acquired_target(
    straight: float, extra: float
) -> None:
    """An acquired target was reached, so the path is at least the straight line."""
    value = path_efficiency([outcome(path_length=straight + extra, straight_line=straight)])
    assert 0.0 < value <= 1.0


@settings(max_examples=50, deadline=None)
@given(rate=st.floats(0.0, 1.0), duration=st.floats(1.0, 600.0))
def test_itr_is_finite_and_non_negative(rate: float, duration: float) -> None:
    value = effective_itr(rate, N_TARGETS, duration)
    assert np.isfinite(value)
    assert value >= 0.0


@settings(max_examples=50, deadline=None)
@given(
    intent_angles=st.lists(st.floats(-180.0, 180.0), min_size=1, max_size=20),
    offset=st.floats(-180.0, 180.0),
)
def test_uci_stays_within_minus_one_and_one(
    intent_angles: list[float], offset: float
) -> None:
    intent = _directions(intent_angles)
    realized = _directions([angle + offset for angle in intent_angles])
    uci, excluded = user_contribution_index(intent, realized)
    assert -1.0 - 1e-9 <= uci <= 1.0 + 1e-9
    assert 0.0 <= excluded <= 1.0


# --- assembly ---


def test_summarize_fills_every_column(cfg_free_trace: dict[str, np.ndarray]) -> None:
    outcomes = [outcome(acquired=i < 5, target_idx=i) for i in range(N_TARGETS)]
    metrics = summarize(outcomes, cfg_free_trace, n_targets=N_TARGETS, dt=0.01)

    assert isinstance(metrics, EpisodeMetrics)
    assert metrics.n_success == 5
    assert metrics.n_timeouts == 3
    assert metrics.success_rate == pytest.approx(5 / 8)
    assert metrics.episode_duration_s == pytest.approx(len(cfg_free_trace["pos"]) * 0.01)
    assert np.isfinite(metrics.user_contribution_index)


@pytest.fixture
def cfg_free_trace() -> dict[str, np.ndarray]:
    """A short synthetic trace with a decoder update every 25 steps, as at 4 Hz."""
    n_steps = 200
    intent = np.tile(np.array([1.0, 0.0]), (n_steps, 1))
    return {
        "pos": np.zeros((n_steps, 2)),
        "vel": np.tile(np.array([0.05, 0.0]), (n_steps, 1)),
        "command": np.tile(np.array([0.05, 0.0]), (n_steps, 1)),
        "decoded_intent": intent,
        "target_direction": intent.copy(),
        "has_command": np.ones(n_steps, dtype=bool),
        "decoder_update": np.arange(n_steps) % 25 == 0,
        "target_idx": np.zeros(n_steps, dtype=np.int32),
    }
