"""S4: shared control between the decoded user command and an autonomous policy.

    u = alpha * u_auto(s) + (1 - alpha) * u_user(p)

`u_user` is S2, reused rather than reimplemented, so `alpha = 0` reduces to S2
**exactly** rather than approximately. There is a test that asserts bit equality.

`u_auto` is a potential field: attraction toward a target, repulsion from each
obstacle within `d0`, the sum clipped to `v_max`.

Two variants, and the difference between them is the point of the ablation.

**intent-aware** (the default) chooses the attractive target as the one most
consistent with the decoded intent direction. That uses the decoder output twice:
once through `u_user` and again through the choice of target. Part of what looks
like shared control succeeding is therefore the decoder informing the autonomy
term, and this variant cannot separate the two.

**intent-blind** always attracts toward the target the task has made active,
independent of the posterior. This is the honest control condition. If S4's
advantage over S2 disappears here, the advantage was the decoder leaking into
the autonomy term rather than arbitration doing useful work.

A consequence worth stating: only intent-blind is decoder-independent at
`alpha = 1`. Intent-aware still reads the posterior to pick a target, so the
alpha = 1 independence check is a check on intent-blind, and its failure on
intent-aware is the leakage rather than a bug.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from micm.env.dynamics import RobotState, clip_speed
from micm.mapping.base import BaseMapping
from micm.mapping.weighted import WeightedMapping, normalised_entropy

ALPHA_FIXED: Final[str] = "fixed"
ALPHA_ADAPTIVE: Final[str] = "adaptive"
ALPHA_MODES: Final[tuple[str, str]] = (ALPHA_FIXED, ALPHA_ADAPTIVE)

INTENT_AWARE: Final[str] = "aware"
INTENT_BLIND: Final[str] = "blind"
INTENT_MODES: Final[tuple[str, str]] = (INTENT_AWARE, INTENT_BLIND)

# Distances below this are treated as contact, so the 1/d repulsion term stays
# finite. The robot is pushed out of an obstacle by the environment anyway.
_MIN_DISTANCE: Final[float] = 1e-6


class SharedMapping(BaseMapping):
    """Blend of a posterior-weighted user command and a potential-field autonomy."""

    name = "s4_shared"

    def __init__(
        self,
        *,
        directions: np.ndarray,
        v_max: float,
        alpha: float,
        alpha_mode: str,
        alpha_min: float,
        alpha_max: float,
        intent_mode: str,
        entropy_scaling: bool,
        k_rep: float,
        d0: float,
        angular_window_deg: float,
    ) -> None:
        super().__init__(directions=directions, v_max=v_max)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if alpha_mode not in ALPHA_MODES:
            raise ValueError(f"unknown alpha_mode {alpha_mode!r}, expected {list(ALPHA_MODES)}")
        if intent_mode not in INTENT_MODES:
            raise ValueError(f"unknown intent_mode {intent_mode!r}, expected {list(INTENT_MODES)}")
        if not 0.0 <= alpha_min <= alpha_max <= 1.0:
            raise ValueError(f"need 0 <= alpha_min <= alpha_max <= 1, got {alpha_min}, {alpha_max}")
        if d0 <= 0.0:
            raise ValueError(f"d0 must be positive, got {d0}")
        if not 0.0 < angular_window_deg <= 180.0:
            raise ValueError(f"angular_window_deg must be in (0, 180], got {angular_window_deg}")

        self.alpha = float(alpha)
        self.alpha_mode = str(alpha_mode)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.intent_mode = str(intent_mode)
        self.k_rep = float(k_rep)
        self.d0 = float(d0)
        self.cos_window = float(np.cos(np.deg2rad(angular_window_deg)))

        # The user term is S2 itself, not a copy of its formula.
        self._user = WeightedMapping(
            directions=self.directions, v_max=self.v_max, entropy_scaling=entropy_scaling
        )

        self._targets: np.ndarray | None = None
        self._obstacles: np.ndarray | None = None

    def bind_task(
        self, *, targets: np.ndarray, obstacles: np.ndarray
    ) -> None:
        """Give the mapping the geometry its autonomy term needs.

        Called by the runner after the task is built, because obstacles are
        jittered per seed and targets are a property of the episode rather than
        of the mapping. Mappings that do not need geometry ignore this.
        """
        self._targets = np.asarray(targets, dtype=np.float64).reshape(-1, 2)
        self._obstacles = np.asarray(obstacles, dtype=np.float64).reshape(-1, 2)

    # --- autonomy ---

    def _attractive_target(self, state: RobotState, posterior: np.ndarray) -> np.ndarray | None:
        """Which target the autonomy pulls toward, or None if it cannot tell.

        Under `blind` this is whatever the task has made active. Under `aware` it
        is the target best aligned with the decoded intent, provided that
        alignment is inside the angular window; outside it the autonomy has no
        opinion and contributes repulsion only, rather than falling back on the
        active target, which would quietly make `aware` behave like `blind`.
        """
        if self._targets is None:
            raise RuntimeError(
                "s4_shared was not given the task geometry; the runner must call "
                "bind_task() before the first step"
            )

        if self.intent_mode == INTENT_BLIND:
            active: np.ndarray = self._targets[state.target_idx]
            return active

        intent = posterior @ self.directions
        norm = float(np.hypot(intent[0], intent[1]))
        if norm <= _MIN_DISTANCE:
            return None

        offsets = self._targets - state.pos
        distances = np.hypot(offsets[:, 0], offsets[:, 1])
        reachable = distances > _MIN_DISTANCE
        if not reachable.any():
            return None

        alignment = np.full(len(self._targets), -np.inf)
        alignment[reachable] = (offsets[reachable] / distances[reachable, None]) @ (intent / norm)

        best = int(np.argmax(alignment))
        if alignment[best] < self.cos_window:
            return None
        chosen: np.ndarray = self._targets[best]
        return chosen

    def _repulsion(self, state: RobotState) -> np.ndarray:
        """Sum of k_rep * (1/d - 1/d0) away from each obstacle inside d0."""
        if self._obstacles is None or len(self._obstacles) == 0:
            return np.zeros(2, dtype=np.float64)

        offsets = state.pos - self._obstacles
        distances = np.maximum(np.hypot(offsets[:, 0], offsets[:, 1]), _MIN_DISTANCE)
        inside = distances < self.d0
        if not inside.any():
            return np.zeros(2, dtype=np.float64)

        magnitude = self.k_rep * (1.0 / distances[inside] - 1.0 / self.d0)
        units = offsets[inside] / distances[inside, None]
        pushed: np.ndarray = (units * magnitude[:, None]).sum(axis=0)
        return pushed

    def autonomy_command(self, state: RobotState, posterior: np.ndarray) -> np.ndarray:
        """The potential field, clipped to the speed limit."""
        target = self._attractive_target(state, posterior)

        attraction = np.zeros(2, dtype=np.float64)
        if target is not None:
            delta = target - state.pos
            norm = float(np.hypot(delta[0], delta[1]))
            if norm > _MIN_DISTANCE:
                attraction = delta / norm * self.v_max

        return clip_speed(attraction + self._repulsion(state), self.v_max)

    # --- arbitration ---

    def effective_alpha(self, posterior: np.ndarray) -> float:
        """The autonomy weight for this posterior.

        Fixed mode returns `alpha`. Adaptive mode returns
        `alpha_min + (alpha_max - alpha_min) * H(p) / log K`, so the robot takes
        over exactly when the user is uncertain. That is expected to perform best
        and to be hit hardest by H3, which is why UCI is reported alongside.
        """
        if self.alpha_mode == ALPHA_FIXED:
            return self.alpha
        return self.alpha_min + (self.alpha_max - self.alpha_min) * normalised_entropy(posterior)

    def command(self, posterior: np.ndarray, state: RobotState, dt: float) -> np.ndarray:
        user = self._user.command(posterior, state, dt)
        alpha = self.effective_alpha(posterior)

        # Short-circuit at alpha = 0 so the reduction to S2 is exact rather than
        # exact-up-to-floating-point, and so an unbound task cannot raise for a
        # mapping that never consults the geometry.
        if alpha == 0.0:
            return user

        auto = self.autonomy_command(state, posterior)
        return clip_speed(alpha * auto + (1.0 - alpha) * user, self.v_max)
