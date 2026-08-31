"""S1: argmax. The literature default and the lower bound of this study.

    u = v_max * d[argmax(p)]

No parameters and no state. Every bit of the posterior beyond which class is
largest is discarded, which is the practice this project exists to measure the
cost of. It is also at a structural disadvantage in this task: four directions
serve eight targets, so a diagonal target can only be approached by alternating
cardinals, while a mapping that blends directions can point straight at it. That
disadvantage is a property of the task geometry and is reported as such rather
than being designed away.
"""

from __future__ import annotations

import numpy as np

from micm.env.dynamics import RobotState
from micm.mapping.base import BaseMapping


class ArgmaxMapping(BaseMapping):
    """Full speed along the direction of the most probable class."""

    name = "s1_argmax"

    def command(self, posterior: np.ndarray, state: RobotState, dt: float) -> np.ndarray:  # noqa: ARG002
        direction: np.ndarray = self.directions[int(np.argmax(posterior))]
        return direction * self.v_max
