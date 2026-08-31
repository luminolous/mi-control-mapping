"""S2: posterior-weighted blending of the class directions.

    u = v_max * sum_k p_k d_k

Opposing directions cancel, so an uncertain posterior produces a slower command
rather than a confident wrong one. This is also the mapping that can point at a
diagonal target directly, where argmax has to zigzag between two cardinals.

With `entropy_scaling` the command is additionally scaled by

    1 - H(p) / log K

so a flat posterior gives a near standstill. Without it a flat posterior still
produces a small command, because the four direction vectors cancel but rarely
exactly. The two variants separate "the blend is short because the classes
disagree" from "the blend is short because the decoder is unsure", which are the
same statement only for a symmetric direction table. Both are reported.

S4 will use this as its user term, so `alpha = 0` reduces exactly to S2.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from micm.env.dynamics import RobotState
from micm.mapping.base import BaseMapping

# Probabilities below this contribute nothing to the entropy sum, and taking
# their logarithm would raise rather than tend to zero.
_ENTROPY_EPS: Final[float] = 1e-12


def normalised_entropy(posterior: np.ndarray) -> float:
    """Shannon entropy over log K, running from 0 (certain) to 1 (uniform)."""
    n_classes = len(posterior)
    if n_classes < 2:
        return 0.0
    usable = posterior[posterior > _ENTROPY_EPS]
    entropy = float(-np.sum(usable * np.log(usable)))
    return entropy / float(np.log(n_classes))


class WeightedMapping(BaseMapping):
    """Posterior-weighted direction, optionally scaled by confidence."""

    name = "s2_weighted"

    def __init__(
        self, *, directions: np.ndarray, v_max: float, entropy_scaling: bool
    ) -> None:
        super().__init__(directions=directions, v_max=v_max)
        self.entropy_scaling = bool(entropy_scaling)

    def blended_direction(self, posterior: np.ndarray) -> np.ndarray:
        """The (2,) blend before the speed limit is applied.

        Exposed so S4 can reuse S2 as its user term rather than reimplementing
        it, which is what makes the alpha = 0 identity exact rather than close.
        """
        blended: np.ndarray = posterior @ self.directions
        if self.entropy_scaling:
            blended = blended * (1.0 - normalised_entropy(posterior))
        return blended

    def command(self, posterior: np.ndarray, state: RobotState, dt: float) -> np.ndarray:  # noqa: ARG002
        return self.blended_direction(posterior) * self.v_max
