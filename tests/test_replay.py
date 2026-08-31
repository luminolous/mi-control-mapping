"""Replay tests: window alignment, perturbation, latency, and the cache schema.

The alignment tests use a ramp signal whose every sample equals its own index,
so a cut window states out loud which samples it came from. That is the only way
to check `t_rel` without restating the implementation's own arithmetic back at
it, and `t_rel` is what every latency result in the paper is measured against.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from omegaconf import DictConfig

from micm.replay.cache import (
    META_KEY,
    SCHEMA,
    build_meta,
    posterior_filename,
    posterior_path,
    read_posteriors,
    validate_arrays,
    write_posteriors,
)
from micm.replay.latency import LatencyBuffer, available_mask
from micm.replay.perturb import (
    QUALITY_ABSOLUTE,
    QUALITY_MODES,
    burst_error,
    effective_accuracy,
    label_smoothing,
    lambda_for_accuracy,
    oracle_mixing,
    resolve_quality_levels,
    temporal_jitter,
)
from micm.replay.stream import (
    PROTOCOL_BURST,
    PROTOCOL_STITCHED,
    build_windows,
    burst_windows,
    extract_windows,
    round_half_up,
    stitched_windows,
    window_starts,
)
from micm.utils import load_config

SFREQ = 250.0
N_CHANNELS = 3
N_TIMES = 875  # 3.5 s, the real epoch length
WINDOW_S = 2.0
STRIDE_S = 0.25
WINDOW_SAMPLES = 500
TMIN, TMAX = 0.5, 4.0
N_CLASSES = 4


def ramp_trials(n_trials: int = 4) -> tuple[np.ndarray, pd.DataFrame]:
    """Trial `i` holds the values `i * N_TIMES + sample`, so a window names itself."""
    base = np.arange(n_trials * N_TIMES, dtype=np.float32).reshape(n_trials, N_TIMES)
    X = np.repeat(base[:, None, :], N_CHANNELS, axis=1)
    meta = pd.DataFrame(
        {
            "class": np.arange(n_trials, dtype=np.int8) % N_CLASSES,
            "onset_s": np.arange(n_trials, dtype=np.float32) * 8.0 + 3.0,
        }
    )
    return X, meta


# --- window geometry ---


@pytest.mark.parametrize(("value", "expected"), [(62.5, 63), (312.5, 313), (0.0, 0), (61.4, 61)])
def test_round_half_up_does_not_use_bankers_rounding(value: float, expected: int) -> None:
    """Python's round() sends both 62.5 and 312.5 down, making the step parity dependent."""
    assert round_half_up(value) == expected


def test_window_starts_cover_the_trial_without_running_past_it() -> None:
    starts = window_starts(N_TIMES, WINDOW_SAMPLES, STRIDE_S * SFREQ)
    assert len(starts) == 7  # 2.0 s to 3.5 s at 4 Hz
    assert starts[0] == 0
    assert int(starts[-1]) + WINDOW_SAMPLES <= N_TIMES
    assert int(starts[-1]) + WINDOW_SAMPLES > N_TIMES - int(STRIDE_S * SFREQ)


def test_window_starts_alternate_by_one_sample_at_a_fractional_stride() -> None:
    """62.5 samples cannot be an integer step; the long-run rate must still be 4 Hz."""
    starts = window_starts(N_TIMES, WINDOW_SAMPLES, STRIDE_S * SFREQ)
    steps = np.diff(starts)
    assert set(steps.tolist()) <= {62, 63}
    assert float(steps.mean()) == pytest.approx(62.5, abs=0.5)


@pytest.mark.parametrize(("window", "stride"), [(0, 10.0), (10, 0.0), (-1, 10.0)])
def test_window_starts_rejects_bad_geometry(window: int, stride: float) -> None:
    with pytest.raises(ValueError):
        window_starts(N_TIMES, window, stride)


# --- the alignment guarantee ---


def test_t_rel_marks_the_end_of_the_window() -> None:
    """`t_rel` is when the estimate becomes available, not when its window opened."""
    X, meta = ramp_trials()
    index = burst_windows(
        X, meta, sfreq=SFREQ, window_s=WINDOW_S, stride_s=STRIDE_S, tmin=TMIN, tmax=TMAX
    )
    expected = (index.start_sample + WINDOW_SAMPLES) / SFREQ
    np.testing.assert_allclose(index.t_rel, expected, rtol=0, atol=1e-6)
    assert float(index.t_rel.min()) == pytest.approx(WINDOW_S)


def test_extracted_window_holds_the_samples_t_rel_claims() -> None:
    """The ramp lets the signal itself confirm the index, not just the arithmetic."""
    X, meta = ramp_trials()
    index = burst_windows(
        X, meta, sfreq=SFREQ, window_s=WINDOW_S, stride_s=STRIDE_S, tmin=TMIN, tmax=TMAX
    )
    windows = extract_windows(
        X, index, window_samples=WINDOW_SAMPLES, protocol=PROTOCOL_BURST
    )

    for position in range(len(index)):
        trial = int(index.trial_index[position])
        last_value = float(windows[position, 0, -1])
        # The ramp encodes trial and sample, so the last value names the sample
        # the window ends on. It must equal t_rel * sfreq - 1 inside that trial.
        sample_in_trial = last_value - trial * N_TIMES
        assert sample_in_trial == pytest.approx(index.t_rel[position] * SFREQ - 1)


def test_burst_windows_never_cross_a_trial() -> None:
    X, meta = ramp_trials()
    index = burst_windows(
        X, meta, sfreq=SFREQ, window_s=WINDOW_S, stride_s=STRIDE_S, tmin=TMIN, tmax=TMAX
    )
    assert not index.boundary.any()
    np.testing.assert_array_equal(index.burst_id, index.trial_index)
    assert len(index) == 7 * len(X)


def test_burst_labels_follow_their_trial() -> None:
    X, meta = ramp_trials()
    index = burst_windows(
        X, meta, sfreq=SFREQ, window_s=WINDOW_S, stride_s=STRIDE_S, tmin=TMIN, tmax=TMAX
    )
    expected = meta["class"].to_numpy()[index.trial_index]
    np.testing.assert_array_equal(index.label, expected)


def test_burst_times_come_from_the_original_recording() -> None:
    """Provenance for the session timeline; the environment builds its own."""
    X, meta = ramp_trials()
    index = burst_windows(
        X, meta, sfreq=SFREQ, window_s=WINDOW_S, stride_s=STRIDE_S, tmin=TMIN, tmax=TMAX
    )
    np.testing.assert_allclose(index.burst_onset, meta["onset_s"].to_numpy() + TMIN, atol=1e-5)
    np.testing.assert_allclose(index.burst_offset, meta["onset_s"].to_numpy() + TMAX, atol=1e-5)


def test_a_window_longer_than_a_trial_raises() -> None:
    X, meta = ramp_trials()
    with pytest.raises(ValueError, match="no window would ever be emitted"):
        burst_windows(X, meta, sfreq=SFREQ, window_s=10.0, stride_s=STRIDE_S, tmin=TMIN, tmax=TMAX)


def test_padding_the_first_window_is_refused_with_the_remedy() -> None:
    """Padding fabricates signal. The honest fix is more real data, not more zeros."""
    X, meta = ramp_trials()
    with pytest.raises(ValueError, match=r"widen data\.epoch\.tmin"):
        burst_windows(
            X,
            meta,
            sfreq=SFREQ,
            window_s=WINDOW_S,
            stride_s=STRIDE_S,
            tmin=TMIN,
            tmax=TMAX,
            first_window="pad",
        )


# --- stitched ablation ---


def test_stitched_windows_cross_trial_boundaries_and_say_so() -> None:
    X, meta = ramp_trials()
    index = stitched_windows(
        X, meta, sfreq=SFREQ, window_s=WINDOW_S, stride_s=STRIDE_S, tmin=TMIN, tmax=TMAX
    )
    assert index.boundary.any()
    assert len(index) > 7 * len(X)  # no gaps, so more windows than the burst protocol


def test_a_boundary_window_takes_the_label_of_the_trial_it_ends_in() -> None:
    """The intent in force when the estimate lands is the one that matters."""
    X, meta = ramp_trials()
    index = stitched_windows(
        X, meta, sfreq=SFREQ, window_s=WINDOW_S, stride_s=STRIDE_S, tmin=TMIN, tmax=TMAX
    )
    classes = meta["class"].to_numpy()
    crossing = np.flatnonzero(index.boundary)
    assert crossing.size

    for position in crossing:
        last_sample = int(index.start_sample[position]) + WINDOW_SAMPLES - 1
        assert index.label[position] == classes[last_sample // N_TIMES]


def test_stitched_extraction_reads_the_concatenated_session() -> None:
    X, meta = ramp_trials()
    index = stitched_windows(
        X, meta, sfreq=SFREQ, window_s=WINDOW_S, stride_s=STRIDE_S, tmin=TMIN, tmax=TMAX
    )
    windows = extract_windows(
        X, index, window_samples=WINDOW_SAMPLES, protocol=PROTOCOL_STITCHED
    )
    for position in (0, len(index) // 2, len(index) - 1):
        start = int(index.start_sample[position])
        assert float(windows[position, 0, 0]) == pytest.approx(start)


def test_build_windows_dispatches_and_rejects_unknown_protocols() -> None:
    X, meta = ramp_trials()
    kwargs: dict[str, Any] = {
        "sfreq": SFREQ,
        "window_s": WINDOW_S,
        "stride_s": STRIDE_S,
        "tmin": TMIN,
        "tmax": TMAX,
    }
    assert len(build_windows(X, meta, protocol=PROTOCOL_BURST, **kwargs)) == 7 * len(X)
    assert build_windows(X, meta, protocol=PROTOCOL_STITCHED, **kwargs).boundary.any()
    with pytest.raises(ValueError, match="unknown protocol"):
        build_windows(X, meta, protocol="continuous", **kwargs)


def test_burst_is_the_configured_default() -> None:
    assert load_config("decode").replay.protocol == PROTOCOL_BURST


# --- perturbation ---


def synthetic_posteriors(
    n_bursts: int = 12, per_burst: int = 7, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A mediocre decoder: right more often than chance, wrong often enough to matter."""
    rng = np.random.default_rng(seed)
    labels = np.repeat(rng.integers(0, N_CLASSES, n_bursts), per_burst).astype(np.int8)
    burst_id = np.repeat(np.arange(n_bursts, dtype=np.int32), per_burst)

    logits = rng.normal(size=(len(labels), N_CLASSES))
    logits[np.arange(len(labels)), labels] += 1.2
    exponentiated = np.exp(logits - logits.max(axis=1, keepdims=True))
    posterior = (exponentiated / exponentiated.sum(axis=1, keepdims=True)).astype(np.float32)
    return posterior, labels, burst_id


def test_oracle_mixing_endpoints() -> None:
    p, y, _ = synthetic_posteriors()
    np.testing.assert_allclose(oracle_mixing(p, y, 0.0).posterior, p, atol=1e-6)
    assert oracle_mixing(p, y, 1.0).effective_accuracy == 1.0


def test_oracle_mixing_preserves_row_sums() -> None:
    p, y, _ = synthetic_posteriors()
    for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
        rows = oracle_mixing(p, y, lam).posterior.sum(axis=1)
        np.testing.assert_allclose(rows, 1.0, atol=1e-5)


def test_oracle_mixing_accuracy_is_non_decreasing_in_lambda() -> None:
    """The bisection in lambda_for_accuracy is only valid because of this."""
    p, y, _ = synthetic_posteriors()
    accuracies = [oracle_mixing(p, y, lam).effective_accuracy for lam in np.linspace(0, 1, 11)]
    assert all(b >= a - 1e-9 for a, b in itertools.pairwise(accuracies))


def test_lambda_for_accuracy_reaches_the_target() -> None:
    p, y, _ = synthetic_posteriors()
    for target in (0.6, 0.8, 0.95):
        lam = lambda_for_accuracy(p, y, target)
        assert oracle_mixing(p, y, lam).effective_accuracy >= target - 1e-9


def test_lambda_for_accuracy_returns_zero_below_the_baseline() -> None:
    p, y, _ = synthetic_posteriors()
    baseline = effective_accuracy(p, y)
    assert lambda_for_accuracy(p, y, baseline - 0.1) == 0.0


@pytest.mark.parametrize("lam", [-0.1, 1.1])
def test_oracle_mixing_rejects_lambda_outside_the_unit_interval(lam: float) -> None:
    p, y, _ = synthetic_posteriors()
    with pytest.raises(ValueError, match="lam must be in"):
        oracle_mixing(p, y, lam)


def test_label_smoothing_keeps_the_argmax_and_therefore_the_accuracy() -> None:
    """Confidence falls, correctness does not. That separation is the whole point."""
    p, y, _ = synthetic_posteriors()
    result = label_smoothing(p, y, 0.5)

    np.testing.assert_array_equal(result.posterior.argmax(axis=1), p.argmax(axis=1))
    assert result.effective_accuracy == pytest.approx(effective_accuracy(p, y))
    assert result.posterior.max(axis=1).mean() < p.max(axis=1).mean()


def test_label_smoothing_at_one_is_uniform() -> None:
    p, y, _ = synthetic_posteriors()
    result = label_smoothing(p, y, 1.0)
    np.testing.assert_allclose(result.posterior, 1.0 / N_CLASSES, atol=1e-6)


def test_temporal_jitter_delays_within_a_burst_only() -> None:
    """A shift across bursts would change the error rate, not just its timing."""
    p, y, burst = synthetic_posteriors()
    result = temporal_jitter(p, y, burst, shift_windows=2)

    for value in np.unique(burst):
        rows = np.flatnonzero(burst == value)
        block, shifted = p[rows], result.posterior[rows]
        # Positions 2 onwards hold what positions 0 onwards used to.
        np.testing.assert_allclose(shifted[2:], block[:-2], atol=1e-6)
        # Vacated positions repeat the burst's first estimate, the stalest one
        # genuinely available at that moment.
        np.testing.assert_allclose(shifted[0], block[0], atol=1e-6)


def test_temporal_jitter_with_no_shift_changes_nothing() -> None:
    p, y, burst = synthetic_posteriors()
    np.testing.assert_allclose(temporal_jitter(p, y, burst, 0).posterior, p, atol=1e-6)


def test_burst_error_produces_runs_rather_than_isolated_mistakes() -> None:
    """An i.i.d. error model at the same accuracy is far easier to average away."""
    p, y, burst = synthetic_posteriors(n_bursts=40)
    rng = np.random.default_rng(0)
    result = burst_error(p, y, burst, rng, p_flip=0.25, mean_len=4.0)

    assert result.effective_accuracy < effective_accuracy(p, y)

    wrong = result.posterior.argmax(axis=1) != y
    runs = np.diff(np.flatnonzero(np.diff(np.concatenate(([0], wrong.view(np.int8), [0])))))[::2]
    assert runs.size
    assert float(runs.mean()) > 1.5  # errors arrive in stretches


def test_burst_error_is_reproducible_from_its_generator() -> None:
    p, y, burst = synthetic_posteriors()
    first = burst_error(p, y, burst, np.random.default_rng(7), p_flip=0.2, mean_len=3.0)
    second = burst_error(p, y, burst, np.random.default_rng(7), p_flip=0.2, mean_len=3.0)
    np.testing.assert_array_equal(first.posterior, second.posterior)


def test_burst_error_never_flips_to_the_true_class() -> None:
    p, y, burst = synthetic_posteriors(n_bursts=30)
    result = burst_error(p, y, burst, np.random.default_rng(1), p_flip=1.0, mean_len=100.0)
    assert result.effective_accuracy == 0.0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"p_flip": 1.5, "mean_len": 3.0}, "p_flip"), ({"p_flip": 0.2, "mean_len": 0.0}, "mean_len")],
)
def test_burst_error_rejects_bad_parameters(kwargs: dict[str, float], match: str) -> None:
    p, y, burst = synthetic_posteriors()
    with pytest.raises(ValueError, match=match):
        burst_error(p, y, burst, np.random.default_rng(0), **kwargs)  # type: ignore[arg-type]


def test_every_perturbation_reports_its_achieved_accuracy() -> None:
    """A matched-accuracy condition that was assumed is not a matched-accuracy condition."""
    p, y, burst = synthetic_posteriors()
    results = [
        oracle_mixing(p, y, 0.3),
        label_smoothing(p, y, 0.2),
        temporal_jitter(p, y, burst, 1),
        burst_error(p, y, burst, np.random.default_rng(0), p_flip=0.2, mean_len=3.0),
    ]
    for result in results:
        assert 0.0 <= result.effective_accuracy <= 1.0
        assert result.effective_accuracy == pytest.approx(
            effective_accuracy(result.posterior, y)
        )
        assert result.posterior.dtype == np.float32


def test_perturbation_rejects_an_invalid_posterior() -> None:
    p, y, _ = synthetic_posteriors()
    broken = p.copy()
    broken[0] *= 2.0
    with pytest.raises(ValueError, match="must sum to 1"):
        oracle_mixing(broken, y, 0.5)


# --- latency ---


def test_latency_buffer_holds_a_value_until_its_time() -> None:
    buffer: LatencyBuffer[str] = LatencyBuffer(0.25)
    buffer.push(2.0, "first")

    assert buffer.latest(2.0) is None
    assert buffer.latest(2.24) is None
    assert buffer.latest(2.25) == "first"


def test_latency_buffer_returns_the_newest_available_value() -> None:
    buffer: LatencyBuffer[str] = LatencyBuffer(0.1)
    buffer.push(1.0, "a")
    buffer.push(1.5, "b")

    assert buffer.latest(1.2) == "a"
    assert buffer.latest(1.6) == "b"


def test_zero_latency_makes_a_value_available_immediately() -> None:
    buffer: LatencyBuffer[str] = LatencyBuffer(0.0)
    buffer.push(1.0, "a")
    assert buffer.latest(1.0) == "a"


def test_latency_buffer_reports_nothing_before_anything_arrives() -> None:
    """None means every mapping returns a zero command. That is the protocol working."""
    buffer: LatencyBuffer[str] = LatencyBuffer(0.25)
    assert buffer.latest(10.0) is None


def test_latency_buffer_resets_between_episodes() -> None:
    buffer: LatencyBuffer[str] = LatencyBuffer(0.0)
    buffer.push(1.0, "a")
    buffer.reset()
    assert len(buffer) == 0
    assert buffer.latest(5.0) is None


def test_latency_buffer_rejects_out_of_order_pushes() -> None:
    """Silently accepting them would return stale commands for the rest of the episode."""
    buffer: LatencyBuffer[str] = LatencyBuffer(0.0)
    buffer.push(2.0, "a")
    with pytest.raises(ValueError, match="time order"):
        buffer.push(1.0, "b")


def test_latency_buffer_rejects_a_negative_latency() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        LatencyBuffer(-0.1)


def test_available_mask_matches_the_buffer() -> None:
    t_rel = np.array([2.0, 2.25, 2.5, 2.75])
    np.testing.assert_array_equal(
        available_mask(t_rel, 0.25, 2.6), np.array([True, True, False, False])
    )


# --- cache ---


def cache_arrays() -> dict[str, np.ndarray]:
    p, y, burst = synthetic_posteriors(n_bursts=3, per_burst=7)
    return {
        "posterior": p,
        "label": y,
        "burst_id": burst,
        "t_rel": np.tile(np.linspace(2.0, 3.5, 7), 3).astype(np.float32),
        "burst_onset": np.array([3.0, 11.0, 19.0], dtype=np.float32),
        "burst_offset": np.array([6.5, 14.5, 22.5], dtype=np.float32),
    }


@pytest.fixture
def cfg() -> DictConfig:
    return load_config("decode")


def test_filename_is_deterministic_and_parseable() -> None:
    name = posterior_filename(
        subject=3, session="T", decoder="riemann", window_ms=2000, stride_ms=250, cfg_hash="abc12345"
    )
    assert name == "03_T_riemann_2000_250_abc12345.npz"


def test_posterior_path_lives_under_the_artifacts_root(tmp_path: Path) -> None:
    path = posterior_path(
        tmp_path,
        subject=1,
        session="E",
        decoder="fbcsp",
        window_ms=2000,
        stride_ms=250,
        cfg_hash="deadbeef",
    )
    assert path.parent == tmp_path / "posteriors"


def test_written_file_round_trips_with_its_schema(tmp_path: Path, cfg: DictConfig) -> None:
    arrays = cache_arrays()
    path = write_posteriors(tmp_path / "c.npz", **arrays, meta=build_meta(cfg))
    loaded = read_posteriors(path)

    for name, dtype in SCHEMA.items():
        np.testing.assert_allclose(getattr(loaded, name), arrays[name], rtol=1e-6)
        assert getattr(loaded, name).dtype == dtype
    assert loaded.n_windows == 21
    assert loaded.n_bursts == 3
    assert loaded.n_classes == N_CLASSES


def test_metadata_says_where_the_file_came_from(tmp_path: Path, cfg: DictConfig) -> None:
    """A file that cannot state its provenance will be misread later."""
    path = write_posteriors(tmp_path / "c.npz", **cache_arrays(), meta=build_meta(cfg))
    meta = read_posteriors(path).meta

    assert set(meta) >= {
        "config",
        "config_hash",
        "micm_version",
        "python",
        "numpy",
        "platform",
        "created_utc",
    }
    assert meta["config"]["data"]["name"] == "bci2a"


def test_a_cache_file_is_never_overwritten_in_place(tmp_path: Path, cfg: DictConfig) -> None:
    """The name carries the config hash, so a collision is a problem, not staleness."""
    arrays = cache_arrays()
    path = write_posteriors(tmp_path / "c.npz", **arrays, meta=build_meta(cfg))
    with pytest.raises(FileExistsError, match="already exists"):
        write_posteriors(path, **arrays, meta=build_meta(cfg))

    write_posteriors(path, **arrays, meta=build_meta(cfg), force=True)


def test_a_failed_write_leaves_no_partial_file(tmp_path: Path, cfg: DictConfig) -> None:
    arrays = cache_arrays()
    arrays["posterior"] = arrays["posterior"] * 2.0
    with pytest.raises(ValueError, match="must sum to 1"):
        write_posteriors(tmp_path / "c.npz", **arrays, meta=build_meta(cfg))
    assert list(tmp_path.iterdir()) == []


def test_reading_a_missing_cache_names_the_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no posterior cache at"):
        read_posteriors(tmp_path / "absent.npz")


def test_reading_a_file_with_a_missing_array_raises(tmp_path: Path, cfg: DictConfig) -> None:
    arrays = cache_arrays()
    del arrays["t_rel"]
    path = tmp_path / "broken.npz"
    np.savez_compressed(path, **arrays, **{META_KEY: json.dumps(build_meta(cfg))})
    with pytest.raises(KeyError, match="t_rel"):
        read_posteriors(path)


def test_validation_rejects_a_burst_id_outside_the_burst_table() -> None:
    arrays = cache_arrays()
    arrays["burst_id"] = arrays["burst_id"] + 10
    with pytest.raises(ValueError, match="but there are 3 bursts"):
        validate_arrays(**arrays)


def test_validation_rejects_t_rel_that_goes_backwards() -> None:
    """Out-of-order windows would make the latency buffer reject the sequence."""
    arrays = cache_arrays()
    arrays["t_rel"] = arrays["t_rel"][::-1].copy()
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_arrays(**arrays)


def test_validation_rejects_a_burst_that_ends_before_it_starts() -> None:
    arrays = cache_arrays()
    arrays["burst_offset"] = arrays["burst_onset"] - 1.0
    with pytest.raises(ValueError, match="ends before it starts"):
        validate_arrays(**arrays)


def test_validation_rejects_a_nan_posterior() -> None:
    arrays = cache_arrays()
    arrays["posterior"] = arrays["posterior"].copy()
    arrays["posterior"][0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or inf"):
        validate_arrays(**arrays)


# --- quality levels ---


def test_gap_fraction_levels_are_distinct_and_span_baseline_to_oracle() -> None:
    """A fixed lambda grid puts three of six points on the same perfect decoder."""
    p, y, _ = synthetic_posteriors(n_bursts=40)
    levels = resolve_quality_levels(p, y, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    achieved = [level.achieved_accuracy for level in levels]
    assert achieved[0] == pytest.approx(effective_accuracy(p, y))
    assert achieved[-1] == 1.0
    assert len(set(np.round(achieved, 6))) == len(levels)
    assert all(b >= a for a, b in itertools.pairwise(achieved))


def test_gap_fraction_targets_close_the_stated_share_of_the_gap() -> None:
    p, y, _ = synthetic_posteriors(n_bursts=40)
    baseline = effective_accuracy(p, y)
    for level in resolve_quality_levels(p, y, [0.0, 0.5, 1.0]):
        expected = baseline + level.level * (1.0 - baseline)
        assert level.target_accuracy == pytest.approx(expected)


def test_resolved_levels_report_what_they_achieved_not_what_they_asked_for() -> None:
    """Accuracy is a step function of lambda, so achieved can exceed the target."""
    p, y, _ = synthetic_posteriors(n_bursts=40)
    for level in resolve_quality_levels(p, y, [0.0, 0.3, 0.7, 1.0]):
        assert level.achieved_accuracy >= level.target_accuracy - 1e-9
        assert level.achieved_accuracy == pytest.approx(
            oracle_mixing(p, y, level.lam).effective_accuracy
        )


def test_absolute_mode_collapses_below_a_strong_baseline() -> None:
    """The behaviour that motivated the gap-fraction default, pinned so it is visible."""
    p, y, _ = synthetic_posteriors(n_bursts=40)
    baseline = effective_accuracy(p, y)
    levels = resolve_quality_levels(
        p, y, [baseline - 0.2, baseline - 0.1, 1.0], mode=QUALITY_ABSOLUTE
    )
    assert levels[0].lam == 0.0
    assert levels[1].lam == 0.0
    assert levels[0].achieved_accuracy == levels[1].achieved_accuracy


def test_lambda_saturates_well_before_one() -> None:
    """Why a fixed lambda grid wastes half its episodes on the oracle ceiling."""
    p, y, _ = synthetic_posteriors(n_bursts=40)
    saturation = lambda_for_accuracy(p, y, 1.0)
    assert saturation < 1.0
    assert oracle_mixing(p, y, saturation).effective_accuracy == 1.0


def test_quality_levels_reject_a_bad_mode_or_level() -> None:
    p, y, _ = synthetic_posteriors()
    with pytest.raises(ValueError, match="unknown quality mode"):
        resolve_quality_levels(p, y, [0.0, 1.0], mode="linear")
    with pytest.raises(ValueError, match=r"levels must lie in \[0, 1\]"):
        resolve_quality_levels(p, y, [0.0, 1.5])


@pytest.mark.parametrize("name", ["gap_fraction", "absolute"])
def test_quality_configs_declare_a_mode_and_six_levels(name: str) -> None:
    cfg = load_config("cache", overrides=(f"+quality={name}",))
    assert str(cfg.quality.mode) in QUALITY_MODES
    assert len(cfg.quality.levels) == 6
