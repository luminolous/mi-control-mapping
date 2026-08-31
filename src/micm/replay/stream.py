"""Turning epoched trials into the timed window sequence a decoder would see online.

Two protocols. **Burst** is the main condition and the faithful one for cue-based
data: each trial is one command period, the robot receives nothing between them,
and windows never cross a trial boundary. **Stitched** is the ablation: trials
are concatenated with no gaps and windows may span boundaries, which is what a
continuous-control paper would do with this dataset.

`t_rel` is the time from burst onset to the **end** of the window, because that
is when the estimate becomes available in a causal system. Every latency result
in the paper is measured against it, and an off-by-one stride here would shift
all of them without raising. `tests/test_stream.py` pins it against a ramp
signal whose samples name their own index.

A note on the stride. At 250 Hz a 250 ms stride is 62.5 samples, which is not an
integer. Window starts are therefore `round_half_up(i * stride * sfreq)`, which
alternates 62 and 63 sample steps and keeps the long-run rate exactly 4 Hz.
`t_rel` is computed from the sample index actually used, not from the nominal
stride, so it stays exact instead of accumulating drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from micm.utils.logging import get_logger

logger = get_logger(__name__)

PROTOCOL_BURST: Final[str] = "burst"
PROTOCOL_STITCHED: Final[str] = "stitched"
PROTOCOLS: Final[tuple[str, str]] = (PROTOCOL_BURST, PROTOCOL_STITCHED)

FIRST_WINDOW_DELAY: Final[str] = "delay"


@dataclass(frozen=True)
class WindowIndex:
    """Where every window comes from, without holding the signal itself.

    Attributes:
        trial_index: (n_windows,) int32, row of `X` the window is cut from. Under
            the stitched protocol this is the trial holding the window's last
            sample.
        start_sample: (n_windows,) int32, first sample of the window inside that
            trial for the burst protocol, or inside the concatenated session for
            the stitched one.
        t_rel: (n_windows,) float32, seconds from burst onset to the window end.
        burst_id: (n_windows,) int32, index of the burst the window belongs to.
        label: (n_windows,) int8, class of that burst.
        boundary: (n_windows,) bool, whether the window spans a trial boundary.
            Always False under the burst protocol.
        burst_onset: (n_bursts,) float32, burst start on the original session
            timeline. Provenance only: the environment builds its own timeline.
        burst_offset: (n_bursts,) float32, burst end on that same timeline.
    """

    trial_index: np.ndarray
    start_sample: np.ndarray
    t_rel: np.ndarray
    burst_id: np.ndarray
    label: np.ndarray
    boundary: np.ndarray
    burst_onset: np.ndarray
    burst_offset: np.ndarray

    def __len__(self) -> int:
        return len(self.trial_index)


def round_half_up(value: float) -> int:
    """Round .5 upwards, unlike Python's banker's rounding.

    Window starts land on half-samples at a 250 ms stride and 250 Hz. Banker's
    rounding sends 62.5 down to 62 and 312.5 down to 312, making the step
    sequence depend on the parity of the window index. Half-up keeps it
    predictable, which matters only because someone will eventually check these
    numbers by hand.
    """
    return int(np.floor(value + 0.5))


def window_starts(n_samples: int, window_samples: int, stride_samples: float) -> np.ndarray:
    """First sample of every window that fits entirely inside `n_samples`.

    A window is emitted only once enough data exists for it, which is the
    `delay` policy. The alternative is padding the start of a burst from the
    pre-cue baseline, and padding fabricates signal. A caller who wants those
    earlier estimates should widen `epoch.tmin` so the baseline is real data.

    Raises:
        ValueError: on a non-positive window or stride.
    """
    if window_samples <= 0:
        raise ValueError(f"window_samples must be positive, got {window_samples}")
    if stride_samples <= 0:
        raise ValueError(f"stride_samples must be positive, got {stride_samples}")

    starts: list[int] = []
    index = 0
    while True:
        start = round_half_up(index * stride_samples)
        if start + window_samples > n_samples:
            break
        starts.append(start)
        index += 1
    return np.asarray(starts, dtype=np.int32)


def _validate(X: np.ndarray, trial_meta: pd.DataFrame, first_window: str) -> None:
    if X.ndim != 3:
        raise ValueError(f"expected (n_trials, n_channels, n_times), got shape {X.shape}")
    if len(X) != len(trial_meta):
        raise ValueError(f"{len(X)} trials but {len(trial_meta)} rows of trial_meta")
    if len(X) == 0:
        raise ValueError("no trials to replay")
    if first_window != FIRST_WINDOW_DELAY:
        raise ValueError(
            f"first_window={first_window!r} is not supported; only "
            f"{FIRST_WINDOW_DELAY!r} is. Padding a burst from the pre-cue baseline "
            "needs samples that epoching does not retain. To get the earlier "
            "estimates from real data, widen data.epoch.tmin instead"
        )


def _burst_times(
    trial_meta: pd.DataFrame, tmin: float, tmax: float
) -> tuple[np.ndarray, np.ndarray]:
    """Burst start and end on the original recording timeline.

    Kept as provenance. The environment builds its own timeline, because the
    replay order is decided per episode rather than by the recording.
    """
    onset = trial_meta["onset_s"].to_numpy(dtype=np.float32)
    return onset + np.float32(tmin), onset + np.float32(tmax)


def burst_windows(
    X: np.ndarray,
    trial_meta: pd.DataFrame,
    *,
    sfreq: float,
    window_s: float,
    stride_s: float,
    tmin: float,
    tmax: float,
    first_window: str = FIRST_WINDOW_DELAY,
) -> WindowIndex:
    """Index the windows of the burst protocol: one burst per trial, no crossing.

    Raises:
        ValueError: on malformed input, an unsupported `first_window`, or a
            window longer than a trial.
    """
    _validate(X, trial_meta, first_window)

    window_samples = round_half_up(window_s * sfreq)
    n_times = X.shape[2]
    if window_samples > n_times:
        raise ValueError(
            f"a {window_s}s window is {window_samples} samples but a trial holds "
            f"{n_times}; no window would ever be emitted"
        )

    starts = window_starts(n_times, window_samples, stride_s * sfreq)
    n_trials = len(X)

    trial_index = np.repeat(np.arange(n_trials, dtype=np.int32), len(starts))
    start_sample = np.tile(starts, n_trials).astype(np.int32)
    labels = trial_meta["class"].to_numpy(dtype=np.int8)
    onset, offset = _burst_times(trial_meta, tmin, tmax)

    logger.debug("burst protocol: %d bursts x %d windows", n_trials, len(starts))

    return WindowIndex(
        trial_index=trial_index,
        start_sample=start_sample,
        t_rel=((start_sample + window_samples) / sfreq).astype(np.float32),
        burst_id=trial_index.copy(),
        label=labels[trial_index],
        boundary=np.zeros(len(trial_index), dtype=bool),
        burst_onset=onset,
        burst_offset=offset,
    )


def stitched_windows(
    X: np.ndarray,
    trial_meta: pd.DataFrame,
    *,
    sfreq: float,
    window_s: float,
    stride_s: float,
    tmin: float,
    tmax: float,
    first_window: str = FIRST_WINDOW_DELAY,
) -> WindowIndex:
    """Index the windows of the stitched ablation: trials concatenated, no gaps.

    A window spanning a trial boundary takes the label of the trial holding its
    **last** sample, since that is the intent in force when the estimate lands,
    and is flagged in `boundary` so the analysis can drop it.

    `start_sample` indexes the concatenated session rather than a single trial.

    Raises:
        ValueError: as for `burst_windows`.
    """
    _validate(X, trial_meta, first_window)

    window_samples = round_half_up(window_s * sfreq)
    n_trials, _, n_times = X.shape
    total = n_trials * n_times
    if window_samples > total:
        raise ValueError(
            f"a {window_s}s window is {window_samples} samples but the session holds {total}"
        )

    starts = window_starts(total, window_samples, stride_s * sfreq)
    last_sample = starts + window_samples - 1
    end_trial = (last_sample // n_times).astype(np.int32)
    start_trial = (starts // n_times).astype(np.int32)

    labels = trial_meta["class"].to_numpy(dtype=np.int8)
    onset, offset = _burst_times(trial_meta, tmin, tmax)

    return WindowIndex(
        trial_index=end_trial,
        start_sample=starts.astype(np.int32),
        # Measured from the onset of the burst the window is attributed to, so
        # the column means the same thing under both protocols.
        t_rel=((last_sample + 1 - end_trial * n_times) / sfreq).astype(np.float32),
        burst_id=end_trial.copy(),
        label=labels[end_trial],
        boundary=start_trial != end_trial,
        burst_onset=onset,
        burst_offset=offset,
    )


def build_windows(
    X: np.ndarray,
    trial_meta: pd.DataFrame,
    *,
    protocol: str,
    sfreq: float,
    window_s: float,
    stride_s: float,
    tmin: float,
    tmax: float,
    first_window: str = FIRST_WINDOW_DELAY,
) -> WindowIndex:
    """Dispatch to the protocol named in the config.

    Raises:
        ValueError: on an unknown protocol.
    """
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol!r}, expected one of {list(PROTOCOLS)}")

    builder = burst_windows if protocol == PROTOCOL_BURST else stitched_windows
    return builder(
        X,
        trial_meta,
        sfreq=sfreq,
        window_s=window_s,
        stride_s=stride_s,
        tmin=tmin,
        tmax=tmax,
        first_window=first_window,
    )


def extract_windows(
    X: np.ndarray, index: WindowIndex, *, window_samples: int, protocol: str
) -> np.ndarray:
    """Cut the signal for every indexed window.

    Args:
        X: (n_trials, n_channels, n_times) float32.
        index: from `build_windows`.
        window_samples: window length, which must match the one the index was
            built with.
        protocol: which index space `start_sample` lives in. Passed explicitly
            rather than inferred from the index, because inferring it from, say,
            whether any boundary flag is set would silently do the wrong thing
            on a stitched session that happens to contain no boundary window.

    Returns:
        (n_windows, n_channels, window_samples) float32.

    Raises:
        ValueError: on an unknown protocol or a window running past the data.
    """
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol!r}, expected one of {list(PROTOCOLS)}")

    n_trials, n_channels, n_times = X.shape
    if protocol == PROTOCOL_STITCHED:
        flat = X.transpose(1, 0, 2).reshape(n_channels, n_trials * n_times)
        if len(index) and int(index.start_sample.max()) + window_samples > flat.shape[1]:
            raise ValueError("a window runs past the end of the concatenated session")
        return np.stack(
            [flat[:, start : start + window_samples] for start in index.start_sample]
        ).astype(np.float32)

    if len(index) and int(index.start_sample.max()) + window_samples > n_times:
        raise ValueError("a window runs past the end of its trial")
    return np.stack(
        [
            X[trial, :, start : start + window_samples]
            for trial, start in zip(index.trial_index, index.start_sample, strict=True)
        ]
    ).astype(np.float32)
