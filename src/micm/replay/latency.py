"""Command latency, modelled as a delay buffer.

A command computed from a window ending at `t_rel` becomes available at
`t_rel + latency`. The delay lives here and is applied by the runner at
simulation time; it is never baked into the cached posteriors. That separation
is what keeps latency an experiment parameter: the ablation sweeps four values
against one cache, instead of decoding the dataset four times.

The buffer is deliberately not a queue of fixed length. The environment steps at
100 Hz while posteriors arrive at 4 Hz, so the mapping asks for a value far more
often than one is pushed, and what it needs is "the newest thing that has
arrived by now", not "the thing pushed N steps ago".
"""

from __future__ import annotations

import numpy as np


class LatencyBuffer[T]:
    """Holds values until their availability time has passed.

    Times are simulated seconds on the episode clock, not wall time.
    """

    def __init__(self, latency_s: float) -> None:
        """
        Args:
            latency_s: delay between a value being produced and being usable.
                Zero is a valid ablation condition; negative is not.

        Raises:
            ValueError: on a negative latency.
        """
        if latency_s < 0.0:
            raise ValueError(f"latency_s must be non-negative, got {latency_s}")
        self.latency_s = float(latency_s)
        self._times: list[float] = []
        self._values: list[T] = []
        self._cursor = 0

    def reset(self) -> None:
        """Clear the buffer. Called once per episode."""
        self._times.clear()
        self._values.clear()
        self._cursor = 0

    def push(self, produced_at: float, value: T) -> None:
        """Record a value produced at `produced_at`, usable `latency_s` later.

        Raises:
            ValueError: if values arrive out of order. The runner walks the
                window sequence forward, so out-of-order arrival means the
                sequence was sorted wrongly, and a buffer that quietly accepted
                it would return stale commands for the rest of the episode.
        """
        available_at = produced_at + self.latency_s
        if self._times and available_at < self._times[-1]:
            raise ValueError(
                f"value available at {available_at:.4f}s arrived after one available at "
                f"{self._times[-1]:.4f}s; windows must be pushed in time order"
            )
        self._times.append(available_at)
        self._values.append(value)

    def latest(self, now: float) -> T | None:
        """The newest value available at `now`, or None if nothing has arrived.

        None means no command is available and every mapping returns a zero
        velocity: the burst has not started, it has ended, or the buffer is not
        yet filled. That is the burst protocol working as intended, not an edge
        case to paper over.

        Advances an internal cursor, so a full episode costs one pass over the
        pushed values rather than a scan per environment step. Calling with a
        `now` earlier than a previous call is therefore not supported.
        """
        while self._cursor < len(self._times) and self._times[self._cursor] <= now:
            self._cursor += 1
        if self._cursor == 0:
            return None
        return self._values[self._cursor - 1]

    def __len__(self) -> int:
        return len(self._values)


def available_mask(t_rel: np.ndarray, latency_s: float, until: float) -> np.ndarray:
    """Which windows have arrived by `until`, given the latency.

    A vectorised counterpart to the buffer, for analysis code that works over a
    whole burst at once rather than stepping an episode.

    Raises:
        ValueError: on a negative latency.
    """
    if latency_s < 0.0:
        raise ValueError(f"latency_s must be non-negative, got {latency_s}")
    return np.asarray(t_rel, dtype=np.float64) + latency_s <= until
