"""Sequential center-out reaching with static obstacles.

Eight targets on a circle, presented one at a time in a seeded order. Every
episode therefore has exactly eight attempts, which makes success rate a clean
proportion out of eight rather than a quantity whose denominator has to be
tracked.

The task also decides what the user is *trying* to do at each burst onset: the
class whose direction is closest to the vector from the robot to the active
target. That is the intent the replay pool matches a trial against
(docs/decisions.md D1), and it is the only reason this module knows about
classes at all.

Direction assignment is permuted across seeds, so no mapping and no result may
depend on left hand meaning left.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np

from micm.env.dynamics import RobotState, integrate
from micm.utils.logging import get_logger

logger = get_logger(__name__)

# Class index to unit direction, before any seed permutation. The order matches
# micm.data.constants.CLASS_NAMES: left hand, right hand, feet, tongue. That
# module is not imported here, because env must not depend on the decoder half
# of the project; the two orderings are tied together by a test instead.
DEFAULT_DIRECTIONS: Final[np.ndarray] = np.array(
    [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]], dtype=np.float64
)
N_CLASSES: Final[int] = len(DEFAULT_DIRECTIONS)


@dataclass(frozen=True)
class TargetOutcome:
    """What happened on one target attempt."""

    target_idx: int
    acquired: bool
    duration_s: float
    path_length: float
    straight_line: float
    collisions: int


@dataclass
class EpisodeTrace:
    """Per-step record of an episode, consumed by `env.metrics`.

    Kept as growing lists during the episode and converted once at the end, since
    the number of steps is not known in advance and reallocating a NumPy array
    per step would dominate the runtime.
    """

    pos: list[np.ndarray] = field(default_factory=list)
    vel: list[np.ndarray] = field(default_factory=list)
    command: list[np.ndarray] = field(default_factory=list)
    decoded_intent: list[np.ndarray] = field(default_factory=list)
    target_direction: list[np.ndarray] = field(default_factory=list)
    has_command: list[bool] = field(default_factory=list)
    decoder_update: list[bool] = field(default_factory=list)
    target_idx: list[int] = field(default_factory=list)
    outcomes: list[TargetOutcome] = field(default_factory=list)

    def arrays(self) -> dict[str, np.ndarray]:
        """Stack the per-step lists into arrays for the metric functions."""
        return {
            "pos": np.asarray(self.pos, dtype=np.float64).reshape(-1, 2),
            "vel": np.asarray(self.vel, dtype=np.float64).reshape(-1, 2),
            "command": np.asarray(self.command, dtype=np.float64).reshape(-1, 2),
            "decoded_intent": np.asarray(self.decoded_intent, dtype=np.float64).reshape(-1, 2),
            "target_direction": np.asarray(self.target_direction, dtype=np.float64).reshape(
                -1, 2
            ),
            "has_command": np.asarray(self.has_command, dtype=bool),
            "decoder_update": np.asarray(self.decoder_update, dtype=bool),
            "target_idx": np.asarray(self.target_idx, dtype=np.int32),
        }


@dataclass(frozen=True)
class StepOutcome:
    """What one environment step did to the task."""

    acquired: bool
    timed_out: bool
    collided: bool
    done: bool


def direction_table(permutation: np.ndarray | None = None) -> np.ndarray:
    """Class-to-direction table, optionally permuted.

    Args:
        permutation: (K,) ordering; `permutation[k]` is the row of
            `DEFAULT_DIRECTIONS` that class `k` receives.

    Raises:
        ValueError: if the permutation is not a permutation of range(K).
    """
    if permutation is None:
        return DEFAULT_DIRECTIONS.copy()

    order = np.asarray(permutation, dtype=np.int64)
    if sorted(order.tolist()) != list(range(N_CLASSES)):
        raise ValueError(f"expected a permutation of 0..{N_CLASSES - 1}, got {order.tolist()}")
    return DEFAULT_DIRECTIONS[order].copy()


def target_positions(n_targets: int, radius: float) -> np.ndarray:
    """(n_targets, 2) points evenly spaced on a circle, starting at angle zero.

    Raises:
        ValueError: on a non-positive count or radius.
    """
    if n_targets < 1:
        raise ValueError(f"n_targets must be at least 1, got {n_targets}")
    if radius <= 0.0:
        raise ValueError(f"radius must be positive, got {radius}")

    angles = np.arange(n_targets, dtype=np.float64) * (2.0 * np.pi / n_targets)
    return np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)


def intended_class(pos: np.ndarray, target: np.ndarray, directions: np.ndarray) -> int:
    """The class whose direction points most nearly from `pos` toward `target`.

    This is the ground-truth intent for the burst about to start. With four
    directions and eight targets, a diagonal target has no single correct class,
    so the intent alternates between the two neighbouring cardinals from burst to
    burst and the trajectory approaches in a staircase. That is a property of the
    task geometry, not of any mapping, and it is the reason argmax is at a
    structural disadvantage against a mapping that can blend directions.

    Returns the lowest-index class on an exact tie, so the choice is
    deterministic.
    """
    delta = np.asarray(target, dtype=np.float64) - np.asarray(pos, dtype=np.float64)
    norm = float(np.hypot(delta[0], delta[1]))
    if norm == 0.0:
        return 0
    return int(np.argmax(np.asarray(directions, dtype=np.float64) @ (delta / norm)))


class CenterOutTask:
    """One episode of the sequential center-out task.

    The caller drives it: ask for `intended_class` at each burst onset, obtain a
    command from a mapping at the environment rate, and call `step`. The task
    owns the clock, the dwell timer, the timeout, and the trace.
    """

    def __init__(
        self,
        *,
        n_targets: int,
        radius: float,
        target_radius: float,
        dwell_s: float,
        timeout_s: float,
        dt: float,
        v_max: float,
        tau: float,
        obstacle_positions: np.ndarray,
        obstacle_radius: float,
        obstacle_jitter: float,
        reset_between_targets: bool,
        rng: np.random.Generator,
        directions: np.ndarray | None = None,
    ) -> None:
        if target_radius <= 0.0 or target_radius >= radius:
            raise ValueError(
                f"target_radius must be in (0, radius), got {target_radius} against {radius}"
            )
        if dwell_s < 0.0:
            raise ValueError(f"dwell_s must be non-negative, got {dwell_s}")
        if timeout_s <= 0.0:
            raise ValueError(f"timeout_s must be positive, got {timeout_s}")

        self.n_targets = int(n_targets)
        self.target_radius = float(target_radius)
        self.dwell_s = float(dwell_s)
        self.timeout_s = float(timeout_s)
        self.dt = float(dt)
        self.v_max = float(v_max)
        self.tau = float(tau)
        self.reset_between_targets = bool(reset_between_targets)
        self.obstacle_radius = float(obstacle_radius)
        self.directions = direction_table() if directions is None else np.asarray(directions)

        self.targets = target_positions(self.n_targets, radius)
        # Presentation order is seeded, so the same seed visits the same targets
        # in the same order for every mapping.
        self.order = rng.permutation(self.n_targets)

        obstacles = np.asarray(obstacle_positions, dtype=np.float64).reshape(-1, 2)
        if obstacle_jitter > 0.0 and len(obstacles):
            obstacles = obstacles + rng.normal(scale=obstacle_jitter, size=obstacles.shape)
        self.obstacles = obstacles

        self._rng = rng
        self.trace = EpisodeTrace()
        self._reset_state()

    # --- episode state ---

    def _reset_state(self) -> None:
        self._pos = np.zeros(2, dtype=np.float64)
        self._vel = np.zeros(2, dtype=np.float64)
        self._attempt = 0
        self._dwell = 0.0
        self._t = 0.0
        self._target_started_at = 0.0
        self._target_path = 0.0
        self._target_start_pos = self._pos.copy()
        self._target_collisions = 0
        self._inside_obstacle = False

    def reset(self) -> RobotState:
        """Return the robot to the centre and clear the trace."""
        self.trace = EpisodeTrace()
        self._reset_state()
        return self.state

    @property
    def state(self) -> RobotState:
        return RobotState(
            pos=self._pos.copy(),
            vel=self._vel.copy(),
            target_idx=int(self.active_target_index),
            dwell_t=self._dwell,
            t=self._t,
        )

    @property
    def active_target_index(self) -> int:
        """Index into `self.targets` of the target currently being reached for."""
        return int(self.order[min(self._attempt, self.n_targets - 1)])

    @property
    def active_target(self) -> np.ndarray:
        target: np.ndarray = self.targets[self.active_target_index]
        return target

    @property
    def done(self) -> bool:
        return self._attempt >= self.n_targets

    def intended_class(self) -> int:
        """Ground-truth intent for a burst starting now. See `intended_class`."""
        return intended_class(self._pos, self.active_target, self.directions)

    def target_direction(self) -> np.ndarray:
        """Unit vector from the robot to the active target.

        The ground-truth intent, used to choose which trial the replay pool
        draws. Not the quantity UCI correlates against: that is the decoded
        intent, which under a poor decoder points somewhere else entirely.
        """
        delta = self.active_target - self._pos
        norm = float(np.hypot(delta[0], delta[1]))
        unit: np.ndarray = delta / norm if norm > 0.0 else np.zeros(2, dtype=np.float64)
        return unit

    # --- stepping ---

    def _resolve_collision(self) -> bool:
        """Push the robot out of any obstacle it has entered. Returns True on entry.

        The episode is not terminated: a collision is a cost to be counted, and
        ending the episode would make the collision metric and the success metric
        the same measurement.
        """
        if not len(self.obstacles):
            self._inside_obstacle = False
            return False

        offsets = self._pos - self.obstacles
        distances = np.hypot(offsets[:, 0], offsets[:, 1])
        nearest = int(np.argmin(distances))
        inside = bool(distances[nearest] < self.obstacle_radius)

        entered = inside and not self._inside_obstacle
        if inside:
            direction = offsets[nearest]
            norm = float(np.hypot(direction[0], direction[1]))
            # A robot exactly at the centre of an obstacle has no direction to be
            # pushed along; any fixed choice will do, and it cannot recur because
            # the next step starts outside.
            unit = direction / norm if norm > 0.0 else np.array([1.0, 0.0])
            self._pos = self.obstacles[nearest] + unit * self.obstacle_radius
        self._inside_obstacle = inside
        return entered

    def step(
        self,
        command: np.ndarray | None,
        *,
        decoded_intent: np.ndarray | None = None,
        decoder_update: bool = False,
    ) -> StepOutcome:
        """Advance one `dt`.

        Args:
            command: (2,) velocity, or None when no command is available. None
                and a zero vector are the same instruction to the robot but are
                recorded differently, because the UCI metric excludes samples
                with no command rather than treating them as a direction.
            decoded_intent: (2,) direction the posterior points in, before the
                mapping turned it into a command. This is what UCI correlates
                the realised motion against; the direction to the target is
                recorded separately and is not a substitute, since under high
                autonomy the robot heads for the target whatever the user
                intended, which is precisely what UCI has to expose.
            decoder_update: whether a new posterior arrived on this step. UCI is
                sampled at decoder update times, not at the 100 Hz env rate,
                which would otherwise count each posterior about 25 times.

        Raises:
            RuntimeError: if called after the episode is done.
        """
        if self.done:
            raise RuntimeError("episode is over; call reset() before stepping again")

        has_command = command is not None
        applied = np.zeros(2, dtype=np.float64) if command is None else np.asarray(command)

        previous = self._pos
        self._pos, self._vel = integrate(
            self._pos, self._vel, applied, dt=self.dt, v_max=self.v_max, tau=self.tau
        )
        collided = self._resolve_collision()
        if collided:
            self._target_collisions += 1

        self._target_path += float(np.hypot(*(self._pos - previous)))
        self._t += self.dt

        self.trace.pos.append(self._pos.copy())
        self.trace.vel.append(self._vel.copy())
        self.trace.command.append(applied.copy())
        self.trace.decoded_intent.append(
            np.zeros(2, dtype=np.float64)
            if decoded_intent is None
            else np.asarray(decoded_intent, dtype=np.float64)
        )
        self.trace.target_direction.append(self.target_direction())
        self.trace.has_command.append(has_command)
        self.trace.decoder_update.append(bool(decoder_update))
        self.trace.target_idx.append(self.active_target_index)

        inside = bool(np.hypot(*(self._pos - self.active_target)) <= self.target_radius)
        # Dwell must be continuous: leaving the target resets it, so momentary
        # contact from a fast pass does not count as an acquisition.
        self._dwell = self._dwell + self.dt if inside else 0.0

        acquired = self._dwell >= self.dwell_s
        elapsed = self._t - self._target_started_at
        timed_out = (not acquired) and elapsed >= self.timeout_s

        if acquired or timed_out:
            self._finish_target(acquired)
        return StepOutcome(
            acquired=acquired, timed_out=timed_out, collided=collided, done=self.done
        )

    def _finish_target(self, acquired: bool) -> None:
        self.trace.outcomes.append(
            TargetOutcome(
                target_idx=self.active_target_index,
                acquired=acquired,
                duration_s=self._t - self._target_started_at,
                path_length=self._target_path,
                straight_line=float(
                    np.hypot(*(self.active_target - self._target_start_pos))
                ),
                collisions=self._target_collisions,
            )
        )
        self._attempt += 1
        self._dwell = 0.0
        self._target_started_at = self._t
        self._target_path = 0.0
        self._target_collisions = 0

        if self.reset_between_targets:
            self._pos = np.zeros(2, dtype=np.float64)
            self._vel = np.zeros(2, dtype=np.float64)
        self._target_start_pos = self._pos.copy()
