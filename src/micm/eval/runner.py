"""The experiment runner: cached posteriors in, episode rows out.

Never trains a decoder and never touches raw EEG. It reads the `.npz` files the
caching script wrote, perturbs them, draws bursts to match what the task says the
user is trying to do, and simulates.

The burst loop is the part worth reading carefully. A burst is 3.5 s of trial
followed by a 2 s gap, but a command does not exist for the whole burst: the
first decoding window needs 2 s of data and the latency buffer holds the result
for another 250 ms, so the robot is driven for the remaining 1.25 s and sits
still the rest of the time. That duty cycle is what sets the speed limit and the
timeout in `configs/env/centerout.yaml`.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig

from micm.env.metrics import EpisodeMetrics, summarize
from micm.env.task import CenterOutTask, direction_table
from micm.mapping.registry import select_mapping
from micm.replay.cache import posterior_path, read_posteriors
from micm.replay.latency import LatencyBuffer
from micm.replay.perturb import (
    ERROR_BURST,
    ERROR_JITTER,
    ERROR_NONE,
    ERROR_SMOOTHING,
    burst_error,
    label_smoothing,
    oracle_mixing,
    resolve_quality_levels,
    temporal_jitter,
)
from micm.utils.logging import get_logger
from micm.utils.seeding import generator_for

logger = get_logger(__name__)


@dataclass(frozen=True)
class Burst:
    """One trial's worth of posteriors, drawn to match an intended class."""

    posterior: np.ndarray  # (n_windows, K) float32
    t_rel: np.ndarray  # (n_windows,) float32, seconds to the window END
    label: int
    burst_id: int


class TrialPool:
    """Draws bursts whose label matches what the task says the user intends.

    The link between the dataset and the task, per docs/decisions.md D1. The
    dataset fixes which class each trial belongs to; the task decides which class
    the user would need at each burst onset. Matching them is what lets an oracle
    posterior actually point at the target.

    Sampling is without replacement within a class, reshuffling only when that
    class runs out, so a subject's trials are spread across the episode rather
    than one strong trial being reused. The queues are built from a seeded
    generator, so the same intent sequence consumes the same trials for every
    mapping and the only difference between mappings is their own behaviour.
    """

    def __init__(
        self,
        posterior: np.ndarray,
        label: np.ndarray,
        burst_id: np.ndarray,
        t_rel: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        self._posterior = np.asarray(posterior, dtype=np.float32)
        self._t_rel = np.asarray(t_rel, dtype=np.float32)
        self._burst_id = np.asarray(burst_id, dtype=np.int32)
        self._rng = rng

        order = np.argsort(self._burst_id, kind="stable")
        self._rows: dict[int, np.ndarray] = {}
        self._label_of: dict[int, int] = {}
        for burst in np.unique(self._burst_id):
            rows = order[self._burst_id[order] == burst]
            self._rows[int(burst)] = rows
            self._label_of[int(burst)] = int(label[rows[0]])

        self._by_class: dict[int, list[int]] = {}
        for burst, cls in self._label_of.items():
            self._by_class.setdefault(cls, []).append(burst)

        self._queues: dict[int, list[int]] = {}
        self.consumed: list[int] = []

    def _refill(self, cls: int) -> None:
        available = self._by_class.get(cls)
        if not available:
            raise KeyError(
                f"no burst of class {cls} in this posterior file; the pool cannot match "
                "the task's intent, so the episode would silently use the wrong class"
            )
        shuffled = list(np.asarray(available)[self._rng.permutation(len(available))])
        self._queues[cls] = [int(burst) for burst in shuffled]

    def draw(self, cls: int) -> Burst:
        """Return the next unused burst of class `cls`.

        Raises:
            KeyError: if the file holds no burst of that class.
        """
        if not self._queues.get(cls):
            self._refill(cls)

        burst = self._queues[cls].pop()
        rows = self._rows[burst]
        self.consumed.append(burst)
        return Burst(
            posterior=self._posterior[rows],
            t_rel=self._t_rel[rows],
            label=self._label_of[burst],
            burst_id=burst,
        )


def synthetic_arrays(
    rng: np.random.Generator,
    *,
    n_bursts: int,
    n_windows: int,
    n_classes: int,
    accuracy: float,
    window_s: float,
    stride_s: float,
    confidence: float = 0.7,
) -> dict[str, np.ndarray]:
    """Posteriors with a known accuracy, for the smoke path and the sanity checks.

    No dataset and no cache file. `accuracy` is the fraction of windows whose
    argmax is the true class; the rest point at a uniformly chosen wrong class.
    `confidence` is the probability mass on the argmax, so a uniform posterior is
    `confidence = 1 / n_classes`.
    """
    if not 0.0 <= accuracy <= 1.0:
        raise ValueError(f"accuracy must be in [0, 1], got {accuracy}")

    # Every class appears, so the pool can always match the task's intent.
    labels = np.tile(np.arange(n_classes), n_bursts // n_classes + 1)[:n_bursts]
    labels = labels[rng.permutation(n_bursts)].astype(np.int8)

    posterior = np.empty((n_bursts * n_windows, n_classes), dtype=np.float32)
    correct = rng.random(n_bursts * n_windows) < accuracy
    window_label = np.repeat(labels, n_windows)

    rest = (1.0 - confidence) / (n_classes - 1)
    for row in range(len(posterior)):
        truth = int(window_label[row])
        if correct[row]:
            peak = truth
        else:
            others = [k for k in range(n_classes) if k != truth]
            peak = int(others[rng.integers(len(others))])
        posterior[row] = rest
        posterior[row, peak] = confidence

    return {
        "posterior": posterior,
        "label": window_label.astype(np.int8),
        "burst_id": np.repeat(np.arange(n_bursts, dtype=np.int32), n_windows),
        "t_rel": np.tile(
            window_s + np.arange(n_windows) * stride_s, n_bursts
        ).astype(np.float32),
        "burst_onset": (np.arange(n_bursts, dtype=np.float32) * 8.0),
        "burst_offset": (np.arange(n_bursts, dtype=np.float32) * 8.0 + window_s),
    }


def apply_error_structure(
    posterior: np.ndarray,
    labels: np.ndarray,
    burst_id: np.ndarray,
    rng: np.random.Generator,
    cfg: DictConfig,
    kind: str,
) -> tuple[np.ndarray, float]:
    """Apply one structured-error variant and report the accuracy it achieved.

    Raises:
        ValueError: on an unknown variant.
    """
    if kind == ERROR_NONE:
        from micm.replay.perturb import effective_accuracy

        return posterior, effective_accuracy(posterior, labels)
    if kind == ERROR_SMOOTHING:
        result = label_smoothing(posterior, labels, float(cfg.error_structure.smoothing_eps))
    elif kind == ERROR_JITTER:
        result = temporal_jitter(
            posterior, labels, burst_id, int(cfg.error_structure.jitter_windows)
        )
    elif kind == ERROR_BURST:
        result = burst_error(
            posterior,
            labels,
            burst_id,
            rng,
            p_flip=float(cfg.error_structure.burst_p_flip),
            mean_len=float(cfg.error_structure.burst_mean_len),
        )
    else:
        raise ValueError(f"unknown error structure {kind!r}")
    return result.posterior, result.effective_accuracy


@dataclass(frozen=True)
class Cell:
    """One combination of independent variables, plus its seed."""

    experiment: str
    subject: int
    decoder: str
    mapping: str
    quality_level: float
    protocol: str
    error_struct: str
    intent_mode: str
    # None for a mapping without an autonomy weight, which the episode row
    # records as NaN.
    alpha: float | None
    latency_ms: int
    seed: int

    def key(self) -> tuple[Any, ...]:
        """Identity used to derive this episode's random stream."""
        return (
            self.experiment,
            self.subject,
            self.decoder,
            self.mapping,
            self.quality_level,
            self.protocol,
            self.error_struct,
            self.intent_mode,
            self.alpha,
            self.latency_ms,
            self.seed,
        )


@dataclass
class EpisodeResult:
    """One row of `episodes.parquet`, before the writer types it."""

    cell: Cell
    metrics: EpisodeMetrics
    lam: float
    effective_acc: float
    kappa_offline: float
    direction_perm: str
    bursts_without_command: int
    wall_time_s: float
    trials_consumed: list[int] = field(default_factory=list)


def _decoded_direction(posterior: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Direction the posterior points in, independent of any mapping.

    The posterior-weighted mean of the class directions. Mapping-independent on
    purpose: UCI has to compare the same intent signal across S1 to S4, so it
    cannot be defined by the mapping under test.
    """
    blended = posterior @ directions
    norm = float(np.hypot(blended[0], blended[1]))
    return blended / norm if norm > 0.0 else np.zeros(2, dtype=np.float64)


def run_episode(
    cell: Cell,
    arrays: dict[str, np.ndarray],
    cfg: DictConfig,
    *,
    kappa_offline: float = float("nan"),
) -> EpisodeResult:
    """Simulate one episode and return its metrics.

    Raises:
        KeyError: if the posterior file holds no burst of a class the task asks
            for, which would otherwise mean silently replaying the wrong intent.
    """
    started = time.perf_counter()
    rng = generator_for(int(cfg.seed), *cell.key())

    permutation = rng.permutation(int(cfg.data.n_classes))
    directions = direction_table(permutation)

    quality = resolve_quality_levels(
        arrays["posterior"],
        arrays["label"],
        [cell.quality_level],
        mode=str(cfg.quality.mode),
    )[0]
    mixed = oracle_mixing(arrays["posterior"], arrays["label"], quality.lam).posterior
    perturbed, effective_acc = apply_error_structure(
        mixed, arrays["label"], arrays["burst_id"], rng, cfg, cell.error_struct
    )

    pool = TrialPool(perturbed, arrays["label"], arrays["burst_id"], arrays["t_rel"], rng)
    mapping = select_mapping(cfg, cell.mapping, directions=directions)
    mapping.reset(rng)

    env = cfg.env
    task = CenterOutTask(
        n_targets=int(env.n_targets),
        radius=float(env.radius),
        target_radius=float(env.target_radius),
        dwell_s=float(env.dwell_s),
        timeout_s=float(env.timeout_s),
        dt=float(env.dt),
        v_max=float(env.v_max),
        tau=float(env.tau),
        obstacle_positions=np.asarray(env.obstacles.positions, dtype=np.float64),
        obstacle_radius=float(env.obstacles.radius),
        obstacle_jitter=float(env.obstacles.jitter),
        reset_between_targets=bool(env.reset_between_targets),
        rng=rng,
        directions=directions,
    )
    task.reset()

    dt = float(env.dt)
    burst_s = float(cfg.data.epoch.tmax) - float(cfg.data.epoch.tmin)
    gap_s = float(cfg.replay.gap_s)
    burst_steps = round(burst_s / dt)
    gap_steps = round(gap_s / dt)
    latency_s = cell.latency_ms / 1000.0

    buffer: LatencyBuffer[np.ndarray] = LatencyBuffer(latency_s)
    bursts_without_command = 0

    # The task's own per-target timeout already bounds the episode, so this is a
    # guard against a non-terminating task rather than a budget. It is derived
    # rather than configured on purpose: a fixed cap that happens to bind would
    # end the episode with targets never attempted, and those would be counted
    # as failures, which is indistinguishable from genuine failure.
    # See docs/decisions.md D38.
    cycle_s = burst_s + gap_s
    burst_budget = int(np.ceil(int(env.n_targets) * float(env.timeout_s) / cycle_s)) + 2

    bursts_used = 0
    while not task.done:
        bursts_used += 1
        if bursts_used > burst_budget:
            raise RuntimeError(
                f"episode ran past {burst_budget} bursts without finishing, which the "
                f"per-target timeout of {float(env.timeout_s)}s should have made "
                "impossible; the task is not terminating"
            )

        burst = pool.draw(task.intended_class())
        buffer.reset()
        for row in range(len(burst.posterior)):
            buffer.push(float(burst.t_rel[row]), burst.posterior[row])

        commanded = False
        previous_id: int | None = None
        for step in range(burst_steps + gap_steps):
            now = step * dt
            # No command before the first window has filled and the latency
            # buffer has released it, and none at all during the gap.
            available = buffer.latest(now) if now < burst_s else None

            update = available is not None and id(available) != previous_id
            previous_id = id(available) if available is not None else None

            command = mapping.step(available, task.state, dt)
            if available is None:
                outcome = task.step(None, decoder_update=False)
            else:
                # A burst counts as commanded only if some non-zero command was
                # actually issued. S3 can hold a posterior without ever crossing
                # its threshold, which is legitimate and must be recorded rather
                # than smoothed over as a zero command.
                commanded = commanded or bool(np.any(command))
                outcome = task.step(
                    command,
                    decoded_intent=_decoded_direction(available, directions),
                    decoder_update=update,
                )
            # Break on what the step reported rather than re-reading task.done,
            # so the episode is never stepped after it has finished.
            if outcome.done:
                break

        if not commanded:
            bursts_without_command += 1

    metrics = summarize(
        task.trace.outcomes, task.trace.arrays(), n_targets=int(env.n_targets), dt=dt
    )
    return EpisodeResult(
        cell=cell,
        metrics=metrics,
        lam=quality.lam,
        effective_acc=effective_acc,
        kappa_offline=kappa_offline,
        direction_perm=",".join(str(int(value)) for value in permutation),
        bursts_without_command=bursts_without_command,
        wall_time_s=time.perf_counter() - started,
    )


def enumerate_cells(cfg: DictConfig) -> list[Cell]:
    """Every cell of the configured grid, in a deterministic order."""
    grid = cfg.grid
    cells: list[Cell] = []
    for subject in grid.subjects:
        for decoder in grid.decoders:
            for mapping in grid.mappings:
                for level in grid.quality_levels:
                    for protocol in grid.protocols:
                        for error_struct in grid.error_structures:
                            for intent_mode in grid.intent_modes:
                                for alpha in grid.alphas:
                                    for latency in grid.latencies_ms:
                                        for seed in range(int(grid.seeds)):
                                            cells.append(
                                                Cell(
                                                    experiment=str(cfg.name),
                                                    subject=int(subject),
                                                    decoder=str(decoder),
                                                    mapping=str(mapping),
                                                    quality_level=float(level),
                                                    protocol=str(protocol),
                                                    error_struct=str(error_struct),
                                                    intent_mode=str(intent_mode),
                                                    alpha=None
                                                    if alpha is None
                                                    else float(alpha),
                                                    latency_ms=int(latency),
                                                    seed=int(seed),
                                                )
                                            )
    return cells


def load_arrays(cell: Cell, cfg: DictConfig) -> tuple[dict[str, np.ndarray], float]:
    """Posteriors for a cell, from the cache or generated synthetically.

    Raises:
        FileNotFoundError: naming the cache file that was expected. There is no
            silent recomputation: a missing cache means the caching script has
            not been run for this configuration.
    """
    if bool(cfg.synthetic.enabled):
        rng = generator_for(int(cfg.seed), "synthetic", cell.subject, cell.decoder)
        arrays = synthetic_arrays(
            rng,
            n_bursts=int(cfg.synthetic.n_bursts),
            n_windows=int(cfg.synthetic.n_windows),
            n_classes=int(cfg.data.n_classes),
            accuracy=float(cfg.synthetic.accuracy),
            window_s=float(cfg.replay.window_s),
            stride_s=float(cfg.replay.stride_s),
            confidence=float(cfg.synthetic.confidence),
        )
        return arrays, float("nan")

    if cfg.posterior_hash is None:
        raise ValueError(
            "synthetic.enabled is false but posterior_hash is null; name the config hash "
            "of the cache to read, which scripts/02_cache_posteriors.py prints when it writes"
        )

    path = posterior_path(
        Path(cfg.paths.artifacts),
        subject=cell.subject,
        session=str(cfg.session),
        decoder=cell.decoder,
        window_ms=round(float(cfg.replay.window_s) * 1000),
        stride_ms=round(float(cfg.replay.stride_s) * 1000),
        cfg_hash=str(cfg.posterior_hash),
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"no posterior cache at {path}. Run scripts/02_cache_posteriors.py for "
            f"subject {cell.subject} with this configuration first"
        )
    cached = read_posteriors(path)
    return (
        {
            "posterior": cached.posterior,
            "label": cached.label,
            "burst_id": cached.burst_id,
            "t_rel": cached.t_rel,
        },
        float(cached.meta.get("kappa_window", float("nan"))),
    )


def iter_results(cfg: DictConfig) -> Iterator[EpisodeResult]:
    """Run every cell, yielding one result each.

    Fails fast: an exception in one episode aborts the run. A partially completed
    matrix silently missing cells is worse than no results at all.
    """
    cells = enumerate_cells(cfg)
    logger.info("%s: %d episodes", cfg.name, len(cells))

    cache: dict[tuple[int, str], tuple[dict[str, np.ndarray], float]] = {}
    every = int(cfg.progress_every)
    started = time.perf_counter()

    for index, cell in enumerate(cells, start=1):
        key = (cell.subject, cell.decoder)
        if key not in cache:
            cache[key] = load_arrays(cell, cfg)
        arrays, kappa = cache[key]

        yield run_episode(cell, arrays, cfg, kappa_offline=kappa)

        if every and index % every == 0:
            elapsed = time.perf_counter() - started
            remaining = elapsed / index * (len(cells) - index)
            logger.info(
                "%d/%d episodes, %.1fs elapsed, about %.0fs remaining",
                index,
                len(cells),
                elapsed,
                remaining,
            )


def run_all(cfg: DictConfig) -> list[EpisodeResult]:
    """Every episode of the configured experiment."""
    return list(iter_results(cfg))


def success_rates_at(
    cfg: DictConfig,
    *,
    accuracy: float,
    confidence: float,
    quality_level: float,
    n_episodes: int,
    mapping: str,
) -> list[float]:
    """Success rates from synthetic episodes at a stated decoder quality.

    Used by the sanity checks, which need a controlled decoder rather than
    whatever the cache happens to hold.
    """
    rates: list[float] = []
    for seed in range(n_episodes):
        cell = Cell(
            experiment=f"sanity_{accuracy}_{confidence}",
            subject=int(cfg.grid.subjects[0]),
            decoder="synthetic",
            mapping=mapping,
            quality_level=quality_level,
            protocol=str(cfg.replay.protocol),
            error_struct=ERROR_NONE,
            intent_mode="aware",
            alpha=None,
            latency_ms=int(cfg.grid.latencies_ms[0]),
            seed=seed,
        )
        rng = generator_for(int(cfg.seed), "sanity", accuracy, confidence, seed)
        arrays = synthetic_arrays(
            rng,
            n_bursts=int(cfg.synthetic.n_bursts),
            n_windows=int(cfg.synthetic.n_windows),
            n_classes=int(cfg.data.n_classes),
            accuracy=accuracy,
            window_s=float(cfg.replay.window_s),
            stride_s=float(cfg.replay.stride_s),
            confidence=confidence,
        )
        rates.append(run_episode(cell, arrays, cfg).metrics.success_rate)
    return rates


def sanity_block(cfg: DictConfig) -> dict[str, Any]:
    """The checks that go into `summary.json`, computed on synthetic decoders.

    Never suppressed and never omitted. A check that cannot run in the current
    configuration is written with `applicable: false` and a reason, so the block
    always states what was and was not verified.

    The oracle ceiling is checked for **every** mapping in the grid, not just the
    first. A mapping that slows down when the posterior is uncertain can fail the
    ceiling while another passes it, and a single-mapping check would hide that
    behind whichever mapping happened to be listed first.
    """
    n_targets = int(cfg.env.n_targets)
    n_classes = int(cfg.data.n_classes)
    oracle_threshold = float(cfg.sanity.oracle_threshold)

    per_mapping: dict[str, float] = {}
    for name in cfg.grid.mappings:
        per_mapping[str(name)] = float(
            np.mean(
                success_rates_at(
                    cfg,
                    accuracy=1.0,
                    confidence=1.0,
                    quality_level=1.0,
                    n_episodes=int(cfg.sanity.oracle_episodes),
                    mapping=str(name),
                )
            )
        )

    # The floor is a property of the environment rather than of any one mapping,
    # and argmax is the mapping most able to blunder into a target, so checking
    # it there is the strictest reading.
    chance = float(
        np.mean(
            success_rates_at(
                cfg,
                accuracy=1.0 / n_classes,
                confidence=1.0 / n_classes,
                quality_level=0.0,
                n_episodes=int(cfg.sanity.chance_episodes),
                mapping=str(cfg.grid.mappings[0]),
            )
        )
    )

    worst = min(per_mapping, key=lambda key: per_mapping[key])
    checks: dict[str, Any] = {
        "oracle_success": {
            "by_mapping": {name: round(value, 6) for name, value in per_mapping.items()},
            "worst_mapping": worst,
            "value": round(per_mapping[worst], 6),
            "threshold": oracle_threshold,
            "pass": per_mapping[worst] >= oracle_threshold,
            "applicable": True,
        },
        "uniform_posterior_chance": {
            "value": round(chance, 6),
            "expected": round(1.0 / n_targets, 6),
            "threshold": float(cfg.sanity.chance_threshold),
            "pass": chance <= float(cfg.sanity.chance_threshold),
            "applicable": True,
        },
        "alpha0_equals_s2": {
            "applicable": False,
            "reason": "requires mapping s4_shared, which lands in T11",
            "pass": None,
        },
    }
    checks["all_passed"] = all(
        entry["pass"]
        for entry in checks.values()
        if isinstance(entry, dict) and entry["applicable"]
    )
    return checks
