"""S3: evidence accumulation with a leak and a threshold.

    e <- gamma * e + log(p + epsilon)      once per DECODER UPDATE
    if max(e) > theta:  emit d[argmax(e)] for hold_s, then e <- 0

Between emissions the command is zero. The mapping trades latency for accuracy:
it waits until the evidence is convincing, which is exactly why mappings have to
be compared at matched latency rather than at matched update rate, or S3 gets
credit for being slow.

**The accumulation is per decoder update, not per environment step.** `step` is
called at 100 Hz while posteriors arrive at 4 Hz, so accumulating per call would
apply the leak twenty-five times as often and add each posterior twenty-five
times over, leaving `gamma` and `theta` meaning nothing. `BaseMapping` detects
the update and calls `on_new_posterior`, and a test fails if the accumulation
ever moves into `command`.

**Watchdog.** If the threshold is never crossed during a burst, no command is
emitted for that burst at all. That is legitimate behaviour for this mapping,
and it is recorded in the episode row as `bursts_without_command` rather than
being smoothed over as a zero command.
"""

from __future__ import annotations

import numpy as np

from micm.env.dynamics import RobotState
from micm.mapping.base import BaseMapping


class EvidenceMapping(BaseMapping):
    """Leaky log-evidence accumulator with a threshold and a hold."""

    name = "s3_evidence"

    def __init__(
        self,
        *,
        directions: np.ndarray,
        v_max: float,
        gamma: float,
        theta: float,
        hold_s: float,
        epsilon: float,
    ) -> None:
        super().__init__(directions=directions, v_max=v_max)
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")
        if hold_s <= 0.0:
            raise ValueError(f"hold_s must be positive, got {hold_s}")
        if epsilon <= 0.0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        self.gamma = float(gamma)
        self.theta = float(theta)
        self.hold_s = float(hold_s)
        self.epsilon = float(epsilon)

        self.evidence = np.zeros(self.n_classes, dtype=np.float64)
        self.n_emissions = 0
        self._hold_remaining = 0.0
        self._emitted: np.ndarray | None = None

    def on_reset(self, rng: np.random.Generator) -> None:  # noqa: ARG002
        self.evidence = np.zeros(self.n_classes, dtype=np.float64)
        self.n_emissions = 0
        self._hold_remaining = 0.0
        self._emitted = None

    def on_new_posterior(self, posterior: np.ndarray) -> None:
        """Integrate one decoder update. Called once per posterior, never per step."""
        self.evidence = self.gamma * self.evidence + np.log(posterior + self.epsilon)

        if float(self.evidence.max()) > self.theta:
            self._emitted = self.directions[int(np.argmax(self.evidence))]
            self._hold_remaining = self.hold_s
            self.n_emissions += 1
            # Reset at emission, as the specification writes it: evidence for the
            # next decision starts from nothing rather than from a total that has
            # already been acted on.
            self.evidence = np.zeros(self.n_classes, dtype=np.float64)

    def command(self, posterior: np.ndarray, state: RobotState, dt: float) -> np.ndarray:  # noqa: ARG002
        if self._hold_remaining <= 0.0 or self._emitted is None:
            return np.zeros(2, dtype=np.float64)

        self._hold_remaining -= dt
        emitted: np.ndarray = self._emitted * self.v_max
        if self._hold_remaining <= 0.0:
            self._emitted = None
        return emitted
