"""Preprocessing and epoching for cue-based motor-imagery recordings.

Order of operations, and why it is this order:

1. Drop EOG, keep the 22 EEG channels, and record the channel order.
2. Bandpass. Causal by default, because a zero-phase filter reads samples from
   the future and an online system cannot.
3. Exponential moving standardization per channel, on the *continuous*
   recording. Applied after epoching it would either see the whole session at
   once or restart at every trial; applied here it stays causal and matches what
   a running system would compute.
4. Epoch relative to cue onset.
5. Drop trials flagged as artifacts, and report how many.

Epochs are cut with plain array slicing rather than `mne.Epochs`, so the sample
that each epoch starts on is visible in this file. The replay window arithmetic
downstream depends on that boundary, and an off-by-one here shifts every latency
result in the paper.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import mne
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from micm.data.constants import (
    MOABB_LABEL_TO_CLASS,
    N_CLASSES,
    SESSIONS,
    TRIAL_META_COLUMNS,
)
from micm.utils.logging import get_logger

logger = get_logger(__name__)

_MIN_STD: Final[float] = 0.0


@dataclass(frozen=True)
class EpochedData:
    """Epoched trials for one subject.

    Attributes:
        X: (n_trials, n_channels, n_times) float32.
        y: (n_trials,) int8 in [0, N_CLASSES).
        trial_meta: one row per trial, columns `TRIAL_META_COLUMNS`, indexed by
            position from zero so it can be handed straight to `splits`.
        channel_names: channel order of axis 1 of `X`.
        n_artifact_dropped: trials removed because the recording flagged them.
        artifact_annotations_found: whether any reject annotation existed at all.
            False with `n_artifact_dropped == 0` means the source carried no
            artifact information, which is different from a clean recording.
        n_truncated_dropped: trials removed because the recording ended before
            their window did.
    """

    X: np.ndarray
    y: np.ndarray
    trial_meta: pd.DataFrame
    channel_names: tuple[str, ...]
    n_artifact_dropped: int
    artifact_annotations_found: bool
    n_truncated_dropped: int


def exponential_moving_standardize(
    data: np.ndarray,  # (n_channels, n_times) float
    *,
    factor_new: float,
    init_block_size: int,
    eps: float,
) -> np.ndarray:  # (n_channels, n_times) float32
    """Causal per-channel standardization with an exponentially decaying window.

    Each sample is standardized using only the samples up to and including it.
    The first `init_block_size` samples are instead standardized with that
    block's own mean and standard deviation, because the exponential estimate
    has not converged yet and would otherwise divide by a near-zero variance.

    Raises:
        ValueError: on a non-2D input, a `factor_new` outside (0, 1], a negative
            `init_block_size`, or a non-positive `eps`.
    """
    if data.ndim != 2:
        raise ValueError(f"expected (n_channels, n_times), got shape {data.shape}")
    if not 0.0 < factor_new <= 1.0:
        raise ValueError(f"factor_new must be in (0, 1], got {factor_new}")
    if init_block_size < 0:
        raise ValueError(f"init_block_size must be non-negative, got {init_block_size}")
    if eps <= _MIN_STD:
        raise ValueError(f"eps must be positive, got {eps}")

    frame = pd.DataFrame(np.asarray(data, dtype=np.float64).T)
    meaned = frame.ewm(alpha=factor_new).mean()
    demeaned = frame - meaned
    variance = (demeaned * demeaned).ewm(alpha=factor_new).mean()

    # copy=True because pandas 3 hands back a read-only view, and the init block
    # below is written in place.
    std = np.sqrt(variance.to_numpy(copy=True))
    standardized: np.ndarray = demeaned.to_numpy(copy=True) / np.maximum(eps, std)

    block = min(init_block_size, data.shape[1])
    if block > 0:
        head = np.asarray(data[:, :block], dtype=np.float64)
        head_mean = head.mean(axis=1, keepdims=True)
        head_std = head.std(axis=1, keepdims=True)
        standardized[:block] = ((head - head_mean) / np.maximum(eps, head_std)).T

    return standardized.T.astype(np.float32)


def preprocess_raw(raw: mne.io.BaseRaw, cfg: DictConfig) -> mne.io.BaseRaw:
    """Drop EOG, bandpass, and standardize one continuous recording.

    Operates on a copy; the input is left untouched so a caller can re-run with
    a different config without reloading from disk.

    Raises:
        ValueError: if the channel count after dropping EOG is not the expected
            number of EEG channels.
    """
    out = raw.copy()

    present = [name for name in cfg.channels.drop_names if name in out.ch_names]
    if present:
        out.drop_channels(present)

    if bool(cfg.channels.pick_eeg_only):
        out.pick("eeg")

    expected = int(cfg.channels.expected_eeg)
    if len(out.ch_names) != expected:
        raise ValueError(
            f"expected {expected} EEG channels, got {len(out.ch_names)} after dropping "
            f"{present or 'nothing'} and picking EEG: {out.ch_names}. "
            "Update data.channels.drop_names rather than relaxing the count"
        )

    phase = "zero" if cfg.filter.filtfilt else "forward"
    out.filter(
        l_freq=float(cfg.filter.l_freq),
        h_freq=float(cfg.filter.h_freq),
        method="iir",
        iir_params={
            "order": int(cfg.filter.iir_order),
            "ftype": str(cfg.filter.iir_ftype),
            "output": "sos",
        },
        phase=phase,
        verbose=False,
    )

    def standardize(data: np.ndarray) -> np.ndarray:
        return exponential_moving_standardize(
            data * float(cfg.standardize.input_scale),
            factor_new=float(cfg.standardize.factor_new),
            init_block_size=int(cfg.standardize.init_block_size),
            eps=float(cfg.standardize.eps),
        ).astype(np.float64)

    out.apply_function(standardize, channel_wise=False, verbose=False)

    low, high = (float(v) for v in cfg.standardize.expected_std_range)
    observed = float(out.get_data().std())
    if not low <= observed <= high:
        logger.warning(
            "standardized signal has std %.4g, outside the expected [%.2g, %.2g]. "
            "The usual cause is standardize.input_scale not matching the recording's "
            "units: below the range means eps dominated the division and the signal "
            "was never standardized, which sends a scale-sensitive decoder to chance "
            "while covariance-based ones hide it",
            observed,
            low,
            high,
        )
    return out


def epoch_length(cfg: DictConfig, sfreq: float) -> int:
    """Number of samples in one epoch, from the config alone.

    Exposed so the replay stream can compute window counts without loading data.
    """
    span = float(cfg.epoch.tmax) - float(cfg.epoch.tmin)
    n_times = round(span * sfreq)
    return n_times + 1 if bool(cfg.epoch.include_tmax) else n_times


def _reject_windows(raw: mne.io.BaseRaw, cfg: DictConfig) -> tuple[list[tuple[float, float]], bool]:
    """Time spans marked as rejected, and whether any such annotation existed."""
    labels = {str(label) for label in cfg.artifacts.reject_labels}
    spans = [
        (float(onset), float(onset) + float(duration))
        for onset, duration, description in zip(
            raw.annotations.onset,
            raw.annotations.duration,
            raw.annotations.description,
            strict=True,
        )
        if str(description) in labels
    ]
    return spans, bool(spans)


def epoch_raw(
    raw: mne.io.BaseRaw,
    cfg: DictConfig,
    *,
    subject: int,
    session: str,
    run: int,
) -> EpochedData:
    """Cut one preprocessed recording into trials.

    Expects `raw` to have been through `preprocess_raw` already; this function
    does not filter or standardize, so the two steps stay separately testable.

    Raises:
        ValueError: on an unknown session label, or if a trial's epoch window
            runs past the end of the recording. A truncated trial is a real
            problem with the recording and is not silently dropped.
    """
    if session not in SESSIONS:
        raise ValueError(f"unknown session {session!r}, expected one of {list(SESSIONS)}")

    sfreq = float(raw.info["sfreq"])
    n_times = epoch_length(cfg, sfreq)
    start_offset = round(float(cfg.epoch.tmin) * sfreq)
    event_offset = float(cfg.epoch.event_offset_s)

    reject_spans, annotations_found = _reject_windows(raw, cfg)
    data = raw.get_data()

    max_truncation = round(float(cfg.epoch.max_truncation_s) * sfreq)
    trials: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    n_dropped = 0
    n_truncated = 0

    for onset, description in zip(
        raw.annotations.onset, raw.annotations.description, strict=True
    ):
        label = str(description)
        if label not in MOABB_LABEL_TO_CLASS:
            continue

        cue_time = float(onset) + event_offset
        start = round(cue_time * sfreq) + start_offset
        stop = start + n_times
        shortfall = stop - data.shape[1]
        if start < 0 or shortfall > max_truncation:
            raise ValueError(
                f"trial at t={cue_time:.3f}s needs samples [{start}, {stop}) but the "
                f"recording has {data.shape[1]}, short by {shortfall} samples "
                f"({shortfall / sfreq:.3f}s), more than epoch.max_truncation_s "
                f"({float(cfg.epoch.max_truncation_s)}s). A shortfall this large means the "
                "epoch geometry is wrong, not that the file ended early; check epoch.tmin, "
                "epoch.tmax, and epoch.event_offset_s against this dataset"
            )
        if shortfall > 0:
            # The recording stopped just after this trial's imagery period. Drop
            # it rather than padding: padding fabricates signal, and the sample
            # count is what the replay window arithmetic depends on.
            n_truncated += 1
            continue

        epoch_start = cue_time + float(cfg.epoch.tmin)
        epoch_stop = cue_time + float(cfg.epoch.tmax)
        is_artifact = any(
            span_start < epoch_stop and epoch_start < span_stop
            for span_start, span_stop in reject_spans
        )
        if is_artifact and bool(cfg.artifacts.drop_flagged):
            n_dropped += 1
            continue

        trials.append(data[:, start:stop])
        rows.append(
            {
                "trial_id": -1,  # assigned by the caller once sessions are merged
                "subject": subject,
                "session": session,
                "run": run,
                "class": MOABB_LABEL_TO_CLASS[label],
                "onset_s": cue_time,
                "artifact": is_artifact,
            }
        )

    X = (
        np.stack(trials).astype(np.float32)
        if trials
        else np.empty((0, data.shape[0], n_times), dtype=np.float32)
    )
    meta = pd.DataFrame(rows, columns=list(TRIAL_META_COLUMNS))
    y = meta["class"].to_numpy(dtype=np.int8) if rows else np.empty(0, dtype=np.int8)

    return EpochedData(
        X=X,
        y=y,
        trial_meta=meta,
        channel_names=tuple(raw.ch_names),
        n_artifact_dropped=n_dropped,
        artifact_annotations_found=annotations_found,
        n_truncated_dropped=n_truncated,
    )


def epoch_subject(
    raws_by_session: Mapping[str, Sequence[mne.io.BaseRaw]],
    cfg: DictConfig,
    *,
    subject: int,
) -> EpochedData:
    """Preprocess and epoch every run of every session for one subject.

    Sessions are concatenated in the order given by `SESSIONS`, runs in the order
    they arrive. Trial order within a session is the recording's own; shuffling
    it would destroy the slow drift structure that makes replay realistic.

    Raises:
        ValueError: if a session is missing, if channel order differs between
            runs, or if no trials survive.
    """
    missing = set(SESSIONS) - set(raws_by_session)
    if missing:
        raise ValueError(f"missing session(s) {sorted(missing)} for subject {subject}")

    parts: list[EpochedData] = []
    for session in SESSIONS:
        for run, raw in enumerate(raws_by_session[session]):
            prepared = preprocess_raw(raw, cfg)
            parts.append(epoch_raw(prepared, cfg, subject=subject, session=session, run=run))

    channel_names = parts[0].channel_names
    for part in parts[1:]:
        if part.channel_names != channel_names:
            raise ValueError(
                "channel order differs between runs; the decoder would be fitted on "
                f"a different montage than it is applied to: {channel_names} vs "
                f"{part.channel_names}"
            )

    X = np.concatenate([part.X for part in parts], axis=0)
    meta = pd.concat([part.trial_meta for part in parts], ignore_index=True)
    if len(meta) == 0:
        raise ValueError(f"subject {subject} has no usable trials")

    meta["trial_id"] = np.arange(len(meta), dtype=np.int64)
    meta = meta.astype(
        {"trial_id": "int32", "subject": "int8", "run": "int8", "class": "int8", "onset_s": "float32"}
    )
    y = meta["class"].to_numpy(dtype=np.int8)

    n_dropped = sum(part.n_artifact_dropped for part in parts)
    n_truncated = sum(part.n_truncated_dropped for part in parts)
    annotations_found = any(part.artifact_annotations_found for part in parts)

    expect_annotations = bool(cfg.artifacts.expect_annotations)
    if not annotations_found:
        if expect_annotations:
            raise ValueError(
                f"subject {subject}: data.artifacts.expect_annotations is true but no "
                f"annotation matching {list(cfg.artifacts.reject_labels)} was found in any "
                "recording. Either the loader stopped preserving them or the labels are "
                "wrong; both change which trials the decoder is fitted on"
            )
        logger.info(
            "subject %d: source carries no artifact annotations, as configured; "
            "no trial rejected",
            subject,
        )
    if n_truncated:
        logger.info(
            "subject %d: %d trial(s) dropped because the recording ended before their "
            "window did",
            subject,
            n_truncated,
        )
    logger.info(
        "subject %d: %d trials kept, %d dropped as artifacts, %d truncated, classes %s",
        subject,
        len(meta),
        n_dropped,
        n_truncated,
        np.bincount(y, minlength=N_CLASSES).tolist(),
    )

    return EpochedData(
        X=X,
        y=y,
        trial_meta=meta,
        channel_names=channel_names,
        n_artifact_dropped=n_dropped,
        artifact_annotations_found=annotations_found,
        n_truncated_dropped=n_truncated,
    )
