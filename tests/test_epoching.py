"""Preprocessing and epoching tests, run against synthetic recordings.

The dataset is mocked, never the code under test. Every recording here is an
`mne.io.RawArray` built in this file, so the epoching path can be executed
without MOABB, without a download, and with data whose correct answer is known
by construction.

The alignment test is the one that matters. An epoch that starts one sample
early shifts every `t_rel` in the cached posteriors and therefore every latency
result in the paper, and nothing about it raises.
"""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf

from micm.data.constants import CLASS_NAMES
from micm.data.epoching import (
    EpochedData,
    epoch_length,
    epoch_raw,
    epoch_subject,
    exponential_moving_standardize,
    preprocess_raw,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SFREQ = 250.0
EEG_NAMES = tuple(f"EEG-{i:02d}" for i in range(22))
TRIAL_SPACING_S = 8.0
FIRST_CUE_S = 10.0


@pytest.fixture
def cfg() -> DictConfig:
    loaded = OmegaConf.load(REPO_ROOT / "configs" / "data" / "bci2a.yaml")
    assert isinstance(loaded, DictConfig)
    return loaded


def _build_raw(
    cfg: DictConfig,
    *,
    n_trials: int = 8,
    ramp: bool = False,
    reject_trials: tuple[int, ...] = (),
    duration_s: float | None = None,
) -> mne.io.RawArray:
    """Synthetic recording with `n_trials` cued trials, one class in rotation.

    With `ramp=True` every channel holds its own sample index, so an epoch's
    first value reveals exactly which sample it was cut from.
    """
    eog_names = tuple(str(name) for name in cfg.channels.eog_names)
    names = [*EEG_NAMES, *eog_names]
    types = ["eeg"] * len(EEG_NAMES) + ["eog"] * len(eog_names)

    total_s = duration_s if duration_s is not None else FIRST_CUE_S + TRIAL_SPACING_S * n_trials
    n_samples = int(total_s * SFREQ)

    if ramp:
        data = np.tile(np.arange(n_samples, dtype=np.float64), (len(names), 1))
    else:
        rng = np.random.default_rng(0)
        data = rng.normal(scale=1e-5, size=(len(names), n_samples))

    info = mne.create_info(ch_names=names, sfreq=SFREQ, ch_types=types)
    raw = mne.io.RawArray(data, info, verbose=False)

    onsets = [FIRST_CUE_S + TRIAL_SPACING_S * i for i in range(n_trials)]
    descriptions = [CLASS_NAMES[i % len(CLASS_NAMES)] for i in range(n_trials)]
    durations = [0.0] * n_trials

    # Place each rejection inside the epoch that will actually be cut, which sits
    # at cue + tmin, not at the annotation onset.
    epoch_start_offset = float(cfg.epoch.event_offset_s) + float(cfg.epoch.tmin)
    for index in reject_trials:
        onsets.append(onsets[index] + epoch_start_offset + 0.5)
        durations.append(0.5)
        descriptions.append("1023")

    raw.set_annotations(
        mne.Annotations(onset=onsets, duration=durations, description=descriptions),
        verbose=False,
    )
    return raw


# --- exponential moving standardization ---


def test_standardize_returns_shape_and_dtype() -> None:
    data = np.random.default_rng(0).normal(size=(4, 500))
    out = exponential_moving_standardize(data, factor_new=0.01, init_block_size=100, eps=1e-4)
    assert out.shape == data.shape
    assert out.dtype == np.float32


def test_standardize_is_causal() -> None:
    """Changing a late sample must not change any earlier output.

    This is the property that makes the whole preprocessing chain honest. A
    standardizer fitted over the full recording would fail it, and would also
    quietly leak session statistics into every trial.
    """
    rng = np.random.default_rng(0)
    data = rng.normal(size=(3, 400))
    perturbed = data.copy()
    perturbed[:, 300:] += 50.0

    kwargs = {"factor_new": 0.01, "init_block_size": 50, "eps": 1e-4}
    baseline = exponential_moving_standardize(data, **kwargs)
    changed = exponential_moving_standardize(perturbed, **kwargs)

    np.testing.assert_allclose(baseline[:, :300], changed[:, :300], rtol=1e-6)
    assert not np.allclose(baseline[:, 300:], changed[:, 300:])


def test_standardize_init_block_uses_block_statistics() -> None:
    data = np.random.default_rng(1).normal(size=(2, 300))
    out = exponential_moving_standardize(data, factor_new=0.01, init_block_size=100, eps=1e-9)
    head = data[:, :100]
    expected = (head - head.mean(axis=1, keepdims=True)) / head.std(axis=1, keepdims=True)
    np.testing.assert_allclose(out[:, :100], expected.astype(np.float32), rtol=1e-4)


def test_standardize_handles_a_constant_channel() -> None:
    """A dead channel has zero variance; eps must keep it finite rather than NaN."""
    data = np.zeros((2, 200))
    out = exponential_moving_standardize(data, factor_new=0.01, init_block_size=50, eps=1e-4)
    assert np.isfinite(out).all()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"factor_new": 0.0, "init_block_size": 10, "eps": 1e-4}, "factor_new"),
        ({"factor_new": 1.5, "init_block_size": 10, "eps": 1e-4}, "factor_new"),
        ({"factor_new": 0.01, "init_block_size": -1, "eps": 1e-4}, "init_block_size"),
        ({"factor_new": 0.01, "init_block_size": 10, "eps": 0.0}, "eps"),
    ],
)
def test_standardize_rejects_bad_parameters(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        exponential_moving_standardize(np.zeros((2, 50)), **kwargs)  # type: ignore[arg-type]


def test_standardize_rejects_non_2d_input() -> None:
    with pytest.raises(ValueError, match="n_channels, n_times"):
        exponential_moving_standardize(
            np.zeros((2, 3, 4)), factor_new=0.01, init_block_size=10, eps=1e-4
        )


# --- epoch geometry ---


def test_epoch_length_matches_the_configured_span(cfg: DictConfig) -> None:
    assert epoch_length(cfg, SFREQ) == 875  # 3.5 s at 250 Hz

    with_tmax = OmegaConf.merge(cfg, {"epoch": {"include_tmax": True}})
    assert isinstance(with_tmax, DictConfig)
    assert epoch_length(with_tmax, SFREQ) == 876


def test_epoch_starts_on_the_expected_sample(cfg: DictConfig) -> None:
    """The off-by-one guard: a ramp signal names the sample each epoch begins on."""
    raw = _build_raw(cfg, n_trials=3, ramp=True)
    result = epoch_raw(raw, cfg, subject=1, session="T", run=0)

    n_times = epoch_length(cfg, SFREQ)
    for trial, onset in enumerate(result.trial_meta["onset_s"]):
        expected_start = round(float(onset) * SFREQ) + round(float(cfg.epoch.tmin) * SFREQ)
        assert result.X[trial, 0, 0] == pytest.approx(expected_start)
        assert result.X[trial, 0, -1] == pytest.approx(expected_start + n_times - 1)


def test_configured_window_matches_the_moabb_interval(cfg: DictConfig) -> None:
    """The absolute window taken from each annotation must be [2.5, 6.0] s.

    MOABB anchors BNCI2014_001 events at trial start and reports its imagery
    interval as [2, 6] from that anchor. Our epoch is expressed relative to the
    cue, so `event_offset_s` has to carry the 2 s from trial start to cue. If
    that offset is ever set back to zero, every epoch moves 2 s earlier, lands
    on the fixation period, and the decoders quietly drop to chance.
    """
    offset = float(cfg.epoch.event_offset_s)
    assert offset + float(cfg.epoch.tmin) == pytest.approx(2.5)
    assert offset + float(cfg.epoch.tmax) == pytest.approx(6.0)


def test_epoch_offset_shifts_the_extracted_samples(cfg: DictConfig) -> None:
    """A ramp signal shows the offset moving the cut, not just sitting in config."""
    raw = _build_raw(cfg, n_trials=2, ramp=True)
    shifted = OmegaConf.merge(cfg, {"epoch": {"event_offset_s": 0.0}})
    assert isinstance(shifted, DictConfig)

    with_offset = epoch_raw(raw, cfg, subject=1, session="T", run=0)
    without = epoch_raw(raw, shifted, subject=1, session="T", run=0)

    delta = float(cfg.epoch.event_offset_s) * SFREQ
    assert with_offset.X[0, 0, 0] - without.X[0, 0, 0] == pytest.approx(delta)


def test_epoch_raw_returns_documented_shapes_and_dtypes(cfg: DictConfig) -> None:
    raw = _build_raw(cfg, n_trials=8)
    result = epoch_raw(raw, cfg, subject=3, session="E", run=2)

    assert result.X.shape == (8, len(EEG_NAMES) + 3, epoch_length(cfg, SFREQ))
    assert result.X.dtype == np.float32
    assert result.y.dtype == np.int8
    assert set(result.y.tolist()) == {0, 1, 2, 3}
    assert list(result.trial_meta["session"]) == ["E"] * 8


def test_epoch_raw_rejects_an_unknown_session(cfg: DictConfig) -> None:
    with pytest.raises(ValueError, match="unknown session"):
        epoch_raw(_build_raw(cfg, n_trials=2), cfg, subject=1, session="X", run=0)


def test_trial_running_past_the_recording_raises(cfg: DictConfig) -> None:
    """A truncated trial is a problem with the recording, not something to drop."""
    raw = _build_raw(cfg, n_trials=2, duration_s=FIRST_CUE_S + TRIAL_SPACING_S + 1.0)
    with pytest.raises(ValueError, match="needs samples"):
        epoch_raw(raw, cfg, subject=1, session="T", run=0)


# --- artifact handling ---


def test_flagged_trials_are_dropped_and_counted(cfg: DictConfig) -> None:
    raw = _build_raw(cfg, n_trials=8, reject_trials=(2, 5))
    result = epoch_raw(raw, cfg, subject=1, session="T", run=0)

    assert result.n_artifact_dropped == 2
    assert len(result.trial_meta) == 6
    assert result.artifact_annotations_found is True


def test_absent_artifact_annotations_are_distinguishable_from_a_clean_run(
    cfg: DictConfig,
) -> None:
    """No annotations at all is not the same fact as no trial being rejected."""
    result = epoch_raw(_build_raw(cfg, n_trials=4), cfg, subject=1, session="T", run=0)
    assert result.n_artifact_dropped == 0
    assert result.artifact_annotations_found is False


def test_flagged_trials_are_kept_when_dropping_is_disabled(cfg: DictConfig) -> None:
    keep = OmegaConf.merge(cfg, {"artifacts": {"drop_flagged": False}})
    assert isinstance(keep, DictConfig)
    result = epoch_raw(_build_raw(cfg, n_trials=8, reject_trials=(1,)), keep, subject=1,
                       session="T", run=0)
    assert len(result.trial_meta) == 8
    assert result.trial_meta["artifact"].sum() == 1


# --- preprocessing ---


def test_preprocess_drops_eog_and_keeps_channel_order(cfg: DictConfig) -> None:
    prepared = preprocess_raw(_build_raw(cfg, n_trials=4), cfg)
    assert tuple(prepared.ch_names) == EEG_NAMES


def test_preprocess_does_not_mutate_the_input(cfg: DictConfig) -> None:
    raw = _build_raw(cfg, n_trials=4)
    before = raw.get_data().copy()
    preprocess_raw(raw, cfg)
    np.testing.assert_array_equal(raw.get_data(), before)


def test_preprocess_raises_on_an_unexpected_channel_count(cfg: DictConfig) -> None:
    raw = _build_raw(cfg, n_trials=4).drop_channels([EEG_NAMES[0]])
    with pytest.raises(ValueError, match="expected 22 EEG channels"):
        preprocess_raw(raw, cfg)


def test_causal_filtering_is_the_default(cfg: DictConfig) -> None:
    """A causal and a zero-phase filter must not produce the same signal.

    Guards against `filtfilt` being read but not acted on, which would leak
    future samples into every window while the config claimed otherwise.
    """
    assert cfg.filter.filtfilt is False
    zero_phase = OmegaConf.merge(cfg, {"filter": {"filtfilt": True}})
    assert isinstance(zero_phase, DictConfig)

    raw = _build_raw(cfg, n_trials=4)
    causal = preprocess_raw(raw, cfg).get_data()
    acausal = preprocess_raw(raw, zero_phase).get_data()
    assert not np.allclose(causal, acausal)


# --- subject assembly ---


def _subject_data(cfg: DictConfig, *, n_trials: int = 8) -> EpochedData:
    raws = {
        "T": [_build_raw(cfg, n_trials=n_trials), _build_raw(cfg, n_trials=n_trials)],
        "E": [_build_raw(cfg, n_trials=n_trials)],
    }
    return epoch_subject(raws, cfg, subject=1)


def test_epoch_subject_concatenates_sessions_in_order(cfg: DictConfig) -> None:
    result = _subject_data(cfg)
    sessions = list(result.trial_meta["session"])
    assert sessions == ["T"] * 16 + ["E"] * 8
    assert result.X.shape == (24, len(EEG_NAMES), epoch_length(cfg, SFREQ))


def test_epoch_subject_numbers_trials_by_position(cfg: DictConfig) -> None:
    """`trial_id` and the frame index must agree, since splits returns positions."""
    result = _subject_data(cfg)
    np.testing.assert_array_equal(
        result.trial_meta["trial_id"].to_numpy(), np.arange(len(result.trial_meta))
    )
    assert result.trial_meta.index.equals(pd.RangeIndex(len(result.trial_meta)))


def test_epoch_subject_output_feeds_the_splitter(cfg: DictConfig) -> None:
    """End to end: what epoching produces is what splits.py accepts."""
    from micm.data.splits import train_test_indices

    result = _subject_data(cfg)
    train, test = train_test_indices(result.trial_meta)
    assert len(train) == 16
    assert len(test) == 8


def test_epoch_subject_requires_both_sessions(cfg: DictConfig) -> None:
    with pytest.raises(ValueError, match="missing session"):
        epoch_subject({"T": [_build_raw(cfg, n_trials=4)]}, cfg, subject=1)


def test_epoch_subject_rejects_a_changed_montage(cfg: DictConfig) -> None:
    other = _build_raw(cfg, n_trials=4)
    other.rename_channels({EEG_NAMES[0]: "RENAMED"})
    with pytest.raises(ValueError, match="channel order differs"):
        epoch_subject(
            {"T": [_build_raw(cfg, n_trials=4)], "E": [other]}, cfg, subject=1
        )
