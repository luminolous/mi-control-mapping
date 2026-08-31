"""Planar velocity-controlled robot.

Roughly forty lines of arithmetic, and deliberately so. A physics engine would
change no conclusion in this paper and would multiply the runtime of a matrix
that executes about 10,000 episodes. `tests/test_architecture.py` enforces that
nothing here imports one.

    v <- clip(u, -v_max, v_max)
    x <- x + v * dt

With `tau > 0` the velocity chases the command through a first-order lag instead
of tracking it instantly, which is closer to a real actuator. It is off by
default: a second-order arm adds tuning burden and moves no result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RobotState:
    """The robot and the task clock at one instant.

    Frozen. A mapping that mutated the environment state would be a bug that
    never announces itself: the trajectory would simply differ from the one the
    dynamics computed, and every metric would still look plausible.

    Attributes:
        pos: (2,) float64 workspace position.
        vel: (2,) float64 velocity, workspace units per second.
        target_idx: which of the task's targets is active.
        dwell_t: seconds spent continuously inside the active target.
        t: seconds since the episode began.
    """

    pos: np.ndarray
    vel: np.ndarray
    target_idx: int
    dwell_t: float
    t: float


def clip_speed(command: np.ndarray, v_max: float) -> np.ndarray:
    """Limit a command to `v_max` in magnitude, preserving its direction.

    Clipping the vector rather than each component keeps a diagonal command
    diagonal. Clipping componentwise would make a diagonal faster than a
    cardinal one by a factor of sqrt(2), which would quietly reward the mappings
    that blend directions.

    Raises:
        ValueError: on a non-positive `v_max` or a command that is not (2,).
    """
    if v_max <= 0.0:
        raise ValueError(f"v_max must be positive, got {v_max}")

    vector = np.asarray(command, dtype=np.float64)
    if vector.shape != (2,):
        raise ValueError(f"command must be (2,), got {vector.shape}")

    speed = float(np.hypot(vector[0], vector[1]))
    if speed <= v_max or speed == 0.0:
        return vector
    return vector * (v_max / speed)


def integrate(
    pos: np.ndarray,
    vel: np.ndarray,
    command: np.ndarray,
    *,
    dt: float,
    v_max: float,
    tau: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance the robot by one time step.

    Args:
        pos: (2,) current position.
        vel: (2,) current velocity, used only when `tau > 0`.
        command: (2,) requested velocity.
        dt: step length in seconds.
        v_max: speed limit.
        tau: actuator time constant. Zero means the velocity equals the clipped
            command immediately.

    Returns:
        The new (pos, vel).

    Raises:
        ValueError: on a non-positive `dt` or a negative `tau`.
    """
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if tau < 0.0:
        raise ValueError(f"tau must be non-negative, got {tau}")

    target_velocity = clip_speed(command, v_max)
    if tau == 0.0:
        new_velocity = target_velocity
    else:
        current = np.asarray(vel, dtype=np.float64)
        new_velocity = current + (target_velocity - current) * (dt / tau)
        new_velocity = clip_speed(new_velocity, v_max)

    return np.asarray(pos, dtype=np.float64) + new_velocity * dt, new_velocity
