"""Environment tests: dynamics, dwell, timeout, collisions, and the oracle ceiling.

`test_a_perfect_controller_acquires_every_target` is the one that matters. It is
the precondition for the oracle sanity check in every real run: if a controller
that always knows the right class cannot clear the task, then a success rate
below 1.0 says nothing about the decoder or the mapping, and the whole matrix
measures the environment instead.

It is also where the risk flagged in docs/decisions.md D1 is settled. Four
directions have to serve eight targets, so a diagonal target is approached in a
staircase, and whether that staircase settles inside the target depends on
`v_max`, `target_radius`, and the burst length together.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf

from micm.env.dynamics import RobotState, clip_speed, integrate
from micm.env.task import (
    DEFAULT_DIRECTIONS,
    N_CLASSES,
    CenterOutTask,
    direction_table,
    intended_class,
    target_positions,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# The burst protocol as the replay and data configs define it: a 3.5 s trial
# followed by a 2 s rest. A command does not exist for the whole burst. The
# first decoding window needs `WINDOW_S` of data and the latency buffer holds
# the result for `LATENCY_S`, so the robot is only driven for the remainder.
BURST_S = 3.5
GAP_S = 2.0
WINDOW_S = 2.0
LATENCY_S = 0.25
COMMAND_S = BURST_S - WINDOW_S - LATENCY_S


@pytest.fixture
def cfg() -> DictConfig:
    loaded = OmegaConf.load(REPO_ROOT / "configs" / "env" / "centerout.yaml")
    assert isinstance(loaded, DictConfig)
    return loaded


def build_task(cfg: DictConfig, seed: int = 0, **overrides: object) -> CenterOutTask:
    settings: dict[str, object] = {
        "n_targets": int(cfg.n_targets),
        "radius": float(cfg.radius),
        "target_radius": float(cfg.target_radius),
        "dwell_s": float(cfg.dwell_s),
        "timeout_s": float(cfg.timeout_s),
        "dt": float(cfg.dt),
        "v_max": float(cfg.v_max),
        "tau": float(cfg.tau),
        "obstacle_positions": np.asarray(cfg.obstacles.positions, dtype=np.float64),
        "obstacle_radius": float(cfg.obstacles.radius),
        "obstacle_jitter": float(cfg.obstacles.jitter),
        "reset_between_targets": bool(cfg.reset_between_targets),
    }
    settings.update(overrides)
    return CenterOutTask(rng=np.random.default_rng(seed), **settings)  # type: ignore[arg-type]


def run_oracle_episode(task: CenterOutTask, *, max_seconds: float = 800.0) -> int:
    """Drive the task with a controller that always knows the right class.

    Obeys the burst protocol exactly: the intended class is read once at each
    burst onset and the resulting direction is held for the whole burst, then no
    command at all for the gap. That constant-direction-per-burst behaviour is
    the hard part of the task, not the decoding.
    """
    task.reset()
    silent_steps = round((WINDOW_S + LATENCY_S) / task.dt)
    command_steps = round(COMMAND_S / task.dt)
    gap_steps = round(GAP_S / task.dt)
    limit = round(max_seconds / task.dt)

    acquired = 0
    steps = 0
    while not task.done and steps < limit:
        direction = task.directions[task.intended_class()]
        command = direction * task.v_max

        # No estimate exists yet: the first window is still filling, then the
        # latency buffer is holding it.
        for _ in range(silent_steps + gap_steps):
            if task.done:
                break
            acquired += int(task.step(None).acquired)
            steps += 1

        for _ in range(command_steps):
            if task.done:
                break
            acquired += int(task.step(command).acquired)
            steps += 1
    return acquired


# --- dynamics ---


def test_zero_lag_velocity_equals_the_clipped_command() -> None:
    pos, vel = integrate(np.zeros(2), np.zeros(2), np.array([2.0, 0.0]), dt=0.01, v_max=0.5)
    np.testing.assert_allclose(vel, [0.5, 0.0])
    np.testing.assert_allclose(pos, [0.005, 0.0])


def test_clip_preserves_direction_rather_than_clipping_per_component() -> None:
    """Componentwise clipping would make a diagonal sqrt(2) times faster than a cardinal.

    That would reward every mapping that blends directions, for a reason that has
    nothing to do with control.
    """
    clipped = clip_speed(np.array([3.0, 3.0]), 1.0)
    assert float(np.hypot(*clipped)) == pytest.approx(1.0)
    np.testing.assert_allclose(clipped, [1 / np.sqrt(2), 1 / np.sqrt(2)])


def test_a_command_within_the_limit_is_untouched() -> None:
    np.testing.assert_allclose(clip_speed(np.array([0.1, 0.2]), 1.0), [0.1, 0.2])


def test_first_order_lag_approaches_the_command_gradually() -> None:
    vel = np.zeros(2)
    pos = np.zeros(2)
    speeds = []
    for _ in range(20):
        pos, vel = integrate(pos, vel, np.array([1.0, 0.0]), dt=0.01, v_max=1.0, tau=0.1)
        speeds.append(float(vel[0]))

    assert speeds[0] < speeds[-1] < 1.0
    assert all(b >= a for a, b in itertools.pairwise(speeds))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"dt": 0.0}, "dt"), ({"dt": 0.01, "tau": -1.0}, "tau"), ({"dt": 0.01, "v_max": 0.0}, "v_max")],
)
def test_integrate_rejects_bad_parameters(kwargs: dict[str, float], match: str) -> None:
    settings = {"dt": 0.01, "v_max": 1.0, **kwargs}
    with pytest.raises(ValueError, match=match):
        integrate(np.zeros(2), np.zeros(2), np.ones(2), **settings)  # type: ignore[arg-type]


def test_robot_state_is_frozen() -> None:
    """A mapping that mutated env state would be a bug that never announces itself."""
    state = RobotState(pos=np.zeros(2), vel=np.zeros(2), target_idx=0, dwell_t=0.0, t=0.0)
    with pytest.raises(AttributeError):
        state.target_idx = 1  # type: ignore[misc]


# --- geometry and intent ---


def test_targets_are_evenly_spaced_on_the_circle(cfg: DictConfig) -> None:
    targets = target_positions(int(cfg.n_targets), float(cfg.radius))
    assert targets.shape == (8, 2)
    np.testing.assert_allclose(np.hypot(targets[:, 0], targets[:, 1]), float(cfg.radius))


def test_targets_do_not_overlap_each_other(cfg: DictConfig) -> None:
    targets = target_positions(int(cfg.n_targets), float(cfg.radius))
    gaps = np.hypot(*(targets[1:] - targets[:-1]).T)
    assert float(gaps.min()) > 2 * float(cfg.target_radius)


def test_direction_table_is_a_unit_basis() -> None:
    assert DEFAULT_DIRECTIONS.shape == (N_CLASSES, 2)
    np.testing.assert_allclose(np.hypot(*DEFAULT_DIRECTIONS.T), 1.0)


def test_direction_permutation_reorders_and_validates() -> None:
    """No mapping may depend on left hand meaning left."""
    permuted = direction_table(np.array([2, 3, 0, 1]))
    np.testing.assert_allclose(permuted[0], DEFAULT_DIRECTIONS[2])
    with pytest.raises(ValueError, match="permutation"):
        direction_table(np.array([0, 0, 1, 2]))


@pytest.mark.parametrize(
    ("target", "expected"), [((1.0, 0.0), 1), ((-1.0, 0.0), 0), ((0.0, 1.0), 3), ((0.0, -1.0), 2)]
)
def test_intended_class_picks_the_nearest_direction(
    target: tuple[float, float], expected: int
) -> None:
    assert intended_class(np.zeros(2), np.array(target), DEFAULT_DIRECTIONS) == expected


def test_intended_class_on_a_diagonal_picks_one_of_the_two_neighbours() -> None:
    """A diagonal target has no correct class; the staircase comes from this."""
    chosen = intended_class(np.zeros(2), np.array([1.0, 1.0]), DEFAULT_DIRECTIONS)
    assert chosen in {1, 3}


# --- dwell, timeout, collisions ---


def test_dwell_is_required_and_must_be_continuous(cfg: DictConfig) -> None:
    """Momentary contact is reachable by drift, so it must not count."""
    task = build_task(cfg, timeout_s=200.0)
    target = task.active_target
    steps_to_acquire = 0

    task.reset()
    while not task.done and steps_to_acquire < 40_000:
        delta = target - task.state.pos
        command = delta / max(float(np.hypot(*delta)), 1e-9) * task.v_max
        outcome = task.step(command)
        steps_to_acquire += 1
        if outcome.acquired:
            break

    outcome_record = task.trace.outcomes[0]
    assert outcome_record.acquired
    # The robot reaches the target boundary and must then remain inside for the
    # full dwell before the attempt counts.
    travel_time = (float(cfg.radius) - float(cfg.target_radius)) / float(cfg.v_max)
    assert outcome_record.duration_s >= travel_time + float(cfg.dwell_s) - 2 * float(cfg.dt)


def test_leaving_the_target_resets_the_dwell_timer(cfg: DictConfig) -> None:
    task = build_task(cfg, timeout_s=500.0)
    target = task.active_target
    task.reset()

    # Approach until just inside, then leave before the dwell completes.
    for _ in range(40_000):
        delta = target - task.state.pos
        if float(np.hypot(*delta)) <= float(cfg.target_radius):
            break
        task.step(delta / float(np.hypot(*delta)) * task.v_max)

    task.step(np.zeros(2))
    assert task.state.dwell_t > 0.0

    away = task.state.pos - target
    for _ in range(2000):
        task.step(away / max(float(np.hypot(*away)), 1e-9) * task.v_max)
        if float(np.hypot(*(task.state.pos - target))) > float(cfg.target_radius):
            break
    assert task.state.dwell_t == 0.0
    assert not task.done


def test_a_target_times_out_and_the_episode_advances(cfg: DictConfig) -> None:
    """On timeout the attempt is a failure and the next target is presented."""
    task = build_task(cfg, timeout_s=1.0)
    task.reset()
    while not task.done:
        task.step(np.zeros(2))

    assert len(task.trace.outcomes) == int(cfg.n_targets)
    assert not any(outcome.acquired for outcome in task.trace.outcomes)


def test_every_episode_has_exactly_n_targets_attempts(cfg: DictConfig) -> None:
    """Which is what makes success rate a clean proportion out of eight."""
    task = build_task(cfg, timeout_s=1.0)
    task.reset()
    while not task.done:
        task.step(np.zeros(2))
    assert len(task.trace.outcomes) == 8


def test_stepping_a_finished_episode_raises(cfg: DictConfig) -> None:
    task = build_task(cfg, timeout_s=1.0)
    task.reset()
    while not task.done:
        task.step(np.zeros(2))
    with pytest.raises(RuntimeError, match="episode is over"):
        task.step(np.zeros(2))


def test_a_head_on_collision_is_counted_and_the_robot_pushed_out(cfg: DictConfig) -> None:
    """Counted as a cost, not terminated: otherwise collisions and success are one metric."""
    task = build_task(
        cfg,
        obstacle_positions=np.array([[0.3, 0.0]]),
        obstacle_radius=0.1,
        timeout_s=500.0,
    )
    task.reset()

    collisions = 0
    for _ in range(2000):
        collisions += int(task.step(np.array([task.v_max, 0.0])).collided)

    assert collisions >= 1
    distance = float(np.hypot(*(task.state.pos - np.array([0.3, 0.0]))))
    assert distance >= 0.1 - 1e-9


def test_one_collision_is_counted_per_entry_not_per_step(cfg: DictConfig) -> None:
    task = build_task(
        cfg, obstacle_positions=np.array([[0.3, 0.0]]), obstacle_radius=0.1, timeout_s=500.0
    )
    task.reset()
    entries = sum(int(task.step(np.array([task.v_max, 0.0])).collided) for _ in range(2000))
    assert entries == 1


def test_a_grazing_pass_leaves_the_robot_outside(cfg: DictConfig) -> None:
    task = build_task(
        cfg, obstacle_positions=np.array([[0.3, 0.2]]), obstacle_radius=0.1, timeout_s=500.0
    )
    task.reset()
    for _ in range(1500):
        task.step(np.array([task.v_max, 0.0]))
        assert float(np.hypot(*(task.state.pos - np.array([0.3, 0.2])))) >= 0.1 - 1e-9


def test_no_obstacles_is_a_valid_configuration(cfg: DictConfig) -> None:
    task = build_task(cfg, obstacle_positions=np.zeros((0, 2)), timeout_s=5.0)
    task.reset()
    while not task.done:
        assert not task.step(np.zeros(2)).collided


# --- the oracle ceiling ---


def test_a_perfect_controller_acquires_every_target(cfg: DictConfig) -> None:
    """The precondition for every oracle sanity check in a real run.

    The controller reads the true intended class at each burst onset and holds
    that one cardinal direction for the whole 3.5 s burst, then sits idle for the
    2 s gap. If this cannot clear eight targets, a success rate below 1.0
    measures the environment rather than the decoder or the mapping.
    """
    task = build_task(cfg)
    acquired = run_oracle_episode(task)

    assert task.done
    assert acquired == int(cfg.n_targets), (
        f"oracle acquired {acquired} of {int(cfg.n_targets)}; "
        f"outcomes {[o.acquired for o in task.trace.outcomes]}"
    )


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_the_oracle_ceiling_holds_across_seeds(cfg: DictConfig, seed: int) -> None:
    """Target order is seeded, so the ceiling must not depend on a lucky ordering."""
    task = build_task(cfg, seed=seed)
    assert run_oracle_episode(task) == int(cfg.n_targets)


def test_the_oracle_ceiling_holds_under_a_permuted_direction_table(cfg: DictConfig) -> None:
    task = build_task(cfg, directions=direction_table(np.array([3, 2, 1, 0])))
    assert run_oracle_episode(task) == int(cfg.n_targets)


def test_a_task_with_no_command_at_all_acquires_nothing(cfg: DictConfig) -> None:
    """The floor to the oracle's ceiling: a stationary robot must score zero."""
    task = build_task(cfg, timeout_s=5.0)
    task.reset()
    while not task.done:
        task.step(None)
    assert not any(outcome.acquired for outcome in task.trace.outcomes)


def test_the_trace_records_one_row_per_step(cfg: DictConfig) -> None:
    task = build_task(cfg, timeout_s=2.0)
    task.reset()
    steps = 0
    while not task.done:
        task.step(np.array([task.v_max, 0.0]))
        steps += 1

    arrays = task.trace.arrays()
    assert arrays["pos"].shape == (steps, 2)
    assert arrays["has_command"].shape == (steps,)
    assert arrays["target_idx"].shape == (steps,)


def test_the_trace_distinguishes_no_command_from_a_zero_command(cfg: DictConfig) -> None:
    """UCI excludes samples with no direction; a zero vector has none to report."""
    task = build_task(cfg, timeout_s=2.0)
    task.reset()
    task.step(None)
    task.step(np.zeros(2))

    has_command = task.trace.arrays()["has_command"]
    assert not has_command[0]
    assert has_command[1]


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"target_radius": 0.0}, "target_radius"),
        ({"target_radius": 2.0}, "target_radius"),
        ({"timeout_s": 0.0}, "timeout_s"),
        ({"dwell_s": -1.0}, "dwell_s"),
    ],
)
def test_task_rejects_impossible_settings(
    cfg: DictConfig, overrides: dict[str, float], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        build_task(cfg, **overrides)


def test_timeout_covers_the_worst_case_reach(cfg: DictConfig) -> None:
    """A timeout the oracle cannot meet turns every success rate into a timeout measurement.

    Derived from the geometry so a later config edit fails here rather than
    silently lowering the ceiling: the robot is idle during the gap, consecutive
    targets can be two radii apart because attempts are not reset, and the
    staircase around a diagonal adds about sqrt(2) to the path.
    """
    duty = COMMAND_S / (BURST_S + GAP_S)
    effective_speed = float(cfg.v_max) * duty
    worst_case = 2.0 * np.sqrt(2.0) * float(cfg.radius) / effective_speed

    assert float(cfg.timeout_s) >= worst_case + float(cfg.dwell_s)


def test_per_burst_travel_stays_in_the_measured_window(cfg: DictConfig) -> None:
    """A coarse guard. The oracle ceiling tests are the authority.

    With four directions and eight targets, a diagonal target is approached by
    alternating cardinals and the robot oscillates around it. Both extremes fail:
    too large a step and it never dwells inside, too small and the staircase
    straddles the target rather than settling in it. At exactly one target radius
    per burst the oracle stalls on a diagonal even at a 200 s timeout; 1.25 radii
    clears all eight. The workable band is narrow and was found by measurement.
    """
    radii_per_burst = float(cfg.v_max) * COMMAND_S / float(cfg.target_radius)
    assert 1.1 <= radii_per_burst <= 1.6


def test_uniform_random_commands_stay_near_the_chance_floor(cfg: DictConfig) -> None:
    """The floor beneath the oracle ceiling.

    A task solvable by drift would make every mapping look good. Choosing a
    direction uniformly at random each burst has to leave success far below the
    0.3 the sanity check allows, even with the longer timeout.
    """
    rates = []
    for seed in range(8):
        task = build_task(cfg, seed=seed)
        task.reset()
        rng = np.random.default_rng(1000 + seed)
        silent = round((WINDOW_S + LATENCY_S + GAP_S) / task.dt)
        command_steps = round(COMMAND_S / task.dt)

        acquired = 0
        while not task.done:
            command = task.directions[rng.integers(0, N_CLASSES)] * task.v_max
            for _ in range(silent):
                if task.done:
                    break
                acquired += int(task.step(None).acquired)
            for _ in range(command_steps):
                if task.done:
                    break
                acquired += int(task.step(command).acquired)
        rates.append(acquired / task.n_targets)

    assert float(np.mean(rates)) < 0.3


def test_the_class_order_matches_the_decoder_side_constants() -> None:
    """env must not import micm.data, so the two orderings are tied together here.

    A silent disagreement would permute every direction and would look like a
    mapping that simply performs badly.
    """
    from micm.data.constants import CLASS_NAMES

    assert len(CLASS_NAMES) == N_CLASSES
    expected = {"left_hand": (-1.0, 0.0), "right_hand": (1.0, 0.0), "feet": (0.0, -1.0),
                "tongue": (0.0, 1.0)}
    for index, name in enumerate(CLASS_NAMES):
        assert tuple(DEFAULT_DIRECTIONS[index]) == expected[name]
