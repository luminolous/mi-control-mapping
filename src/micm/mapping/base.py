"""The mapping interface.

A mapping turns a posterior and the robot state into a velocity command. This
subpackage imports nothing from `micm.decoders` or `micm.data`: mappings read
cached posterior files, which is what makes designing a new one cheap.

`step` is called at the environment rate, 100 Hz, while the posterior changes at
4 Hz, so a mapping sees the same value across roughly 25 consecutive calls. Any
mapping that accumulates evidence must accumulate **per decoder update**, not per
call, or its leak and threshold parameters mean nothing. `BaseMapping` detects
the update and calls `on_new_posterior`, so a subclass cannot get this wrong by
forgetting to check.

The runner's side of that contract: it must pass the *same array object* for
every call until a new posterior arrives. Identity is what the detection uses,
because two consecutive posteriors can legitimately hold equal values.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import numpy as np

from micm.env.dynamics import RobotState


@runtime_checkable
class Mapping(Protocol):
    """What the runner requires of a control mapping."""

    name: str

    def reset(self, rng: np.random.Generator) -> None:
        """Clear internal state. Called once per episode."""
        ...

    def step(
        self, posterior: np.ndarray | None, state: RobotState, dt: float
    ) -> np.ndarray:
        """Return a velocity command in workspace units per second."""
        ...


def validate_directions(directions: np.ndarray) -> np.ndarray:
    """Check the class-to-direction table and return it as float64 unit rows.

    Raises:
        ValueError: on the wrong shape or a row that is not a unit vector.
    """
    table = np.asarray(directions, dtype=np.float64)
    if table.ndim != 2 or table.shape[1] != 2:
        raise ValueError(f"directions must be (K, 2), got {table.shape}")
    norms = np.hypot(table[:, 0], table[:, 1])
    if not np.allclose(norms, 1.0, atol=1e-6):
        raise ValueError(f"directions must be unit vectors, got norms {norms.tolist()}")
    return table


def validate_posterior(posterior: np.ndarray, n_classes: int) -> np.ndarray:
    """Check a single posterior row.

    Raises:
        ValueError: on the wrong shape, a non-finite value, a negative
            probability, or a row that does not sum to 1.
    """
    row = np.asarray(posterior, dtype=np.float64).reshape(-1)
    if row.shape != (n_classes,):
        raise ValueError(f"posterior must be ({n_classes},), got {row.shape}")
    if not np.isfinite(row).all():
        raise ValueError("posterior contains NaN or inf")
    if (row < 0.0).any():
        raise ValueError("posterior contains negative probabilities")
    if not np.isclose(row.sum(), 1.0, atol=1e-5):
        raise ValueError(f"posterior must sum to 1, got {float(row.sum()):.6f}")
    return row


class BaseMapping(ABC):
    """Shared machinery: direction table, the zero-command path, update detection.

    Subclasses implement `command`, and optionally `on_new_posterior` and
    `on_reset`.
    """

    name: str = "base"

    def __init__(self, *, directions: np.ndarray, v_max: float) -> None:
        if v_max <= 0.0:
            raise ValueError(f"v_max must be positive, got {v_max}")
        self.directions = validate_directions(directions)
        self.n_classes = len(self.directions)
        self.v_max = float(v_max)
        self._last_posterior_id: int | None = None

    def reset(self, rng: np.random.Generator) -> None:
        """Clear internal state at the start of an episode."""
        self._last_posterior_id = None
        self.on_reset(rng)

    def step(
        self, posterior: np.ndarray | None, state: RobotState, dt: float
    ) -> np.ndarray:
        """Return the (2,) velocity command for this environment step.

        `posterior is None` means no command is available: the burst has not
        started, it has ended, or the latency buffer is not yet filled. Every
        mapping returns a zero vector then. This is the burst protocol working as
        intended, not an edge case to paper over, and it is the reason the trace
        records the absence of a command separately from a zero command.
        """
        if posterior is None:
            self._last_posterior_id = None
            return np.zeros(2, dtype=np.float64)

        row = validate_posterior(posterior, self.n_classes)
        # Identity, not equality: two consecutive posteriors may hold equal
        # values, and treating that as "no update" would stall an accumulator.
        current_id = id(posterior)
        if current_id != self._last_posterior_id:
            self._last_posterior_id = current_id
            self.on_new_posterior(row)

        return self.command(row, state, dt)

    # Optional hooks, deliberately not abstract: S1 has no state, and forcing
    # every mapping to write two empty overrides would make the stateless case
    # noisier than the stateful one.
    def on_reset(self, rng: np.random.Generator) -> None:  # noqa: B027
        """Hook for subclasses with per-episode state. Default does nothing."""

    def on_new_posterior(self, posterior: np.ndarray) -> None:  # noqa: B027
        """Called once per decoder update, before `command`. Default does nothing."""

    @abstractmethod
    def command(self, posterior: np.ndarray, state: RobotState, dt: float) -> np.ndarray:
        """Velocity command for a validated posterior row."""
