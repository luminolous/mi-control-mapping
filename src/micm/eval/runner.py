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

import os
import time
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Final

import numpy as np
from omegaconf import DictConfig, OmegaConf

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

# Blocks per worker. More than one so a worker that draws a run of cheap
# episodes can come back for more, fewer than many so the fixed cost of loading
# a posterior file is not paid repeatedly.
_BLOCKS_PER_WORKER: Final[int] = 4


def _worker_count(cfg: DictConfig) -> int:
    """How many processes to use, never more than the machine has cores.

    `n_workers` is excluded from the config hash because it cannot change a
    number, which is only true because episode seeds are derived from cell
    content. Clamping here rather than raising: a config written on an
    eight-core machine should still run on a four-core one.
    """
    requested = int(cfg.n_workers)
    if requested < 1:
        raise ValueError(f"n_workers must be at least 1, got {requested}")
    available = os.cpu_count() or 1
    if requested > available:
        logger.info("n_workers %d exceeds %d cores, using %d", requested, available, available)
    return min(requested, available)


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
class ReplayVariant:
    """One replay condition: a protocol, a window, and the cache holding it.

    Two ablations vary the protocol and the window length, and both of those
    change the posteriors themselves rather than anything the runner computes.
    A variant therefore names the cache file its condition lives in, and the
    runner reads nothing about replay from `cfg.replay`, which can only describe
    one condition.
    """

    key: str
    protocol: str
    window_s: float
    stride_s: float
    gap_s: float
    # One hash per decoder. The cache config hash covers the decoder block and
    # so differs between decoders, while a single value would send every decoder
    # of a three-decoder experiment looking for one file. The decoder name is
    # already in the filename; the hash disambiguates the rest of the config
    # that produced it. See docs/decisions.md D58.
    posterior_hash: dict[str, str | None]

    def hash_for(self, decoder: str) -> str:
        """The cache hash this decoder's posteriors were written under.

        Raises:
            KeyError: if the config names no hash for that decoder, or names a
                null one, which means the caching script has not been run for it.
        """
        if decoder not in self.posterior_hash:
            raise KeyError(
                f"replay variant {self.key!r} has no posterior_hash for decoder "
                f"{decoder!r}; the caching script prints one hash per decoder. "
                f"Present: {sorted(self.posterior_hash)}"
            )
        found = self.posterior_hash[decoder]
        if found is None:
            raise KeyError(
                f"replay variant {self.key!r} has a null posterior_hash for decoder "
                f"{decoder!r}; run scripts/02_cache_posteriors.py decoder={decoder} "
                "and write the hash it prints into the config"
            )
        return found


def variant_key(protocol: str, window_s: float) -> str:
    """The name a replay variant must be filed under.

    Derived rather than free text, so a variant cannot be reached by a grid
    combination it does not describe.
    """
    return f"{protocol}_w{round(float(window_s) * 1000)}"


def resolve_variants(cfg: DictConfig) -> dict[str, ReplayVariant]:
    """Read `cfg.replay_variants`, checking each entry is filed under its own key.

    Raises:
        KeyError: if the config has no `replay_variants` block.
        ValueError: if an entry sits under a key that does not match its own
            protocol and window, which would let a grid combination silently
            load the posteriors of a different condition.
    """
    if "replay_variants" not in cfg:
        raise KeyError(
            "the experiment config has no `replay_variants` block; the runner reads the "
            "protocol and window from there rather than from `replay`, because the "
            "ablations vary both"
        )

    variants: dict[str, ReplayVariant] = {}
    for name, node in cfg.replay_variants.items():
        expected = variant_key(str(node.protocol), float(node.window_s))
        if str(name) != expected:
            raise ValueError(
                f"replay variant {name!r} describes protocol {node.protocol!r} at "
                f"{float(node.window_s)}s, which belongs under {expected!r}"
            )
        variants[str(name)] = ReplayVariant(
            key=str(name),
            protocol=str(node.protocol),
            window_s=float(node.window_s),
            stride_s=float(node.stride_s),
            gap_s=float(node.gap_s),
            posterior_hash={
                str(decoder): None if value is None else str(value)
                for decoder, value in node.posterior_hash.items()
            },
        )
    return variants


def variant_for(cell: Cell, variants: dict[str, ReplayVariant]) -> ReplayVariant:
    """The replay condition a cell names.

    Raises:
        KeyError: if the grid reaches a protocol and window the config does not
            describe. The alternative, falling back on some other variant, would
            run a condition the episode row does not name.
    """
    key = variant_key(cell.protocol, cell.window_s)
    if key not in variants:
        raise KeyError(
            f"the grid asks for {key!r} but the config composes only "
            f"{sorted(variants)}; add it as "
            f"`- /replay@replay_variants.{key}: {cell.protocol}`"
        )
    return variants[key]


@dataclass(frozen=True)
class Cell:
    """One combination of independent variables, plus its seed."""

    experiment: str
    subject: int
    decoder: str
    mapping: str
    quality_level: float
    # The protocol and the window length select which cached posterior file the
    # episode reads. They are cell properties rather than config properties
    # because two ablations vary them; reading them from `cfg.replay` would make
    # every row of those ablations read the same cache while the column claimed
    # otherwise. See docs/decisions.md D45.
    protocol: str
    window_s: float
    error_struct: str
    # None for a mapping that has no such parameter, which the episode row
    # records as NaN or as a null string. Setting either on a grid that also
    # holds argmax would name a parameter argmax does not have.
    intent_mode: str | None
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
            self.window_s,
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
    command_accuracy: float
    commands_issued: int
    wall_time_s: float
    trials_consumed: list[int] = field(default_factory=list)


def commanded_class(command: np.ndarray, directions: np.ndarray) -> int | None:
    """Which class the issued command points at, or None if it points nowhere.

    The nearest class direction to the command actually sent, not the argmax of
    the posterior. Those differ, and the difference is the point: S3 commands
    only once its accumulated evidence is strong, and S4 has an autonomy term
    that pulls toward the target whatever the posterior said. Measuring the
    posterior instead would score every mapping identically and say nothing
    about the mapping under test.
    """
    norm = float(np.hypot(command[0], command[1]))
    if norm <= 0.0:
        return None
    return int(np.argmax(directions @ (command / norm)))


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
    rng_key: tuple[Any, ...] | None = None,
) -> EpisodeResult:
    """Simulate one episode and return its metrics.

    Args:
        rng_key: overrides the cell identity used to derive the random stream.
            The experiment never passes it: an episode's randomness must depend
            on which cell it belongs to, so that re-running a subset reproduces
            the full grid. The arbitration sanity checks do pass it, because
            comparing two mappings for an exact identity requires them to see the
            same direction permutation, target order and trial draws, and the
            cell key includes the mapping name.

    Raises:
        KeyError: if the posterior file holds no burst of a class the task asks
            for, which would otherwise mean silently replaying the wrong intent.
    """
    started = time.perf_counter()
    rng = generator_for(int(cfg.seed), *(rng_key if rng_key is not None else cell.key()))

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

    # After the task, because the autonomy term of S4 needs the episode geometry
    # and the obstacles are jittered per seed.
    mapping = select_mapping(
        cfg,
        cell.mapping,
        directions=directions,
        overrides={"alpha": cell.alpha, "intent_mode": cell.intent_mode},
    )
    mapping.bind_task(targets=task.targets, obstacles=task.obstacles)
    mapping.reset(rng)

    dt = float(env.dt)
    burst_s = float(cfg.data.epoch.tmax) - float(cfg.data.epoch.tmin)
    gap_s = variant_for(cell, resolve_variants(cfg)).gap_s
    burst_steps = round(burst_s / dt)
    gap_steps = round(gap_s / dt)
    latency_s = cell.latency_ms / 1000.0

    buffer: LatencyBuffer[np.ndarray] = LatencyBuffer(latency_s)
    bursts_without_command = 0
    # Sampled once per decoder update that produced a command, never per
    # environment step: at 100 Hz against a 4 Hz decoder the latter would count
    # each posterior about twenty-five times. The same reason UCI is sampled
    # that way. See docs/decisions.md D57.
    commands_issued = 0
    commands_correct = 0

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
        # The array itself rather than its id, for the reason in
        # micm/mapping/base.py: a freed object's address can be reused.
        previous: np.ndarray | None = None
        for step in range(burst_steps + gap_steps):
            now = step * dt
            # No command before the first window has filled and the latency
            # buffer has released it, and none at all during the gap.
            available = buffer.latest(now) if now < burst_s else None

            update = available is not None and available is not previous
            previous = available

            command = mapping.step(available, task.state, dt)
            if available is None:
                outcome = task.step(None, decoder_update=False)
            else:
                # A burst counts as commanded only if some non-zero command was
                # actually issued. S3 can hold a posterior without ever crossing
                # its threshold, which is legitimate and must be recorded rather
                # than smoothed over as a zero command.
                commanded = commanded or bool(np.any(command))
                if update:
                    # Against the class the burst was drawn for, which is what
                    # the user was imagining for its whole length, rather than
                    # the task's current target: acquiring a target mid-burst
                    # does not retroactively change what was intended.
                    pointed_at = commanded_class(command, directions)
                    if pointed_at is not None:
                        commands_issued += 1
                        commands_correct += int(pointed_at == burst.label)
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
        # NaN rather than zero when nothing was ever commanded: a mapping that
        # never committed has no command accuracy, and zero would read as one
        # that committed and was always wrong.
        command_accuracy=(
            commands_correct / commands_issued if commands_issued else float("nan")
        ),
        commands_issued=commands_issued,
        wall_time_s=time.perf_counter() - started,
    )


# Grid key to the Cell field it fills, in the order the product is taken. The
# order is the file order of `episodes.parquet` and therefore fixed: changing it
# changes nothing about any result but makes two runs of the same config produce
# byte-different Parquet files.
_GRID_AXES: Final[tuple[tuple[str, str], ...]] = (
    ("subjects", "subject"),
    ("decoders", "decoder"),
    ("mappings", "mapping"),
    ("quality_levels", "quality_level"),
    ("protocols", "protocol"),
    ("windows_s", "window_s"),
    ("error_structures", "error_struct"),
    ("intent_modes", "intent_mode"),
    ("alphas", "alpha"),
    ("latencies_ms", "latency_ms"),
)

def enumerate_cells(cfg: DictConfig) -> list[Cell]:
    """Every cell of the configured grid, in a deterministic order.

    Every combination the grid names must be reachable, so this also resolves
    the replay variant of each cell and raises if the config does not describe
    it. Finding that out here rather than on the first episode means a
    mis-specified twelve-hour matrix fails in the first second.

    Raises:
        KeyError: on a grid axis missing from the config, or a protocol and
            window combination with no replay variant.
    """
    grid = cfg.grid
    for name, _ in _GRID_AXES:
        if name not in grid:
            raise KeyError(f"the grid has no {name!r} axis; every axis must be stated explicitly")

    values = [list(grid[name]) for name, _ in _GRID_AXES]
    fields = [field_name for _, field_name in _GRID_AXES]
    variants = resolve_variants(cfg)

    cells: list[Cell] = []
    for combination in product(*values, range(int(grid.seeds))):
        # Read by name rather than unpacked positionally, so reordering the axes
        # cannot silently swap two columns of the same type.
        axis = dict(zip(fields, combination[:-1], strict=True))
        cell = Cell(
            experiment=str(cfg.name),
            subject=int(axis["subject"]),
            decoder=str(axis["decoder"]),
            mapping=str(axis["mapping"]),
            quality_level=float(axis["quality_level"]),
            protocol=str(axis["protocol"]),
            window_s=float(axis["window_s"]),
            error_struct=str(axis["error_struct"]),
            intent_mode=None if axis["intent_mode"] is None else str(axis["intent_mode"]),
            alpha=None if axis["alpha"] is None else float(axis["alpha"]),
            latency_ms=int(axis["latency_ms"]),
            seed=int(combination[-1]),
        )
        # Raises here rather than on the first episode, so a mis-specified
        # twelve-hour matrix fails in its first second.
        variant_for(cell, variants)
        cells.append(cell)
    return cells


def load_arrays(cell: Cell, cfg: DictConfig) -> tuple[dict[str, np.ndarray], float]:
    """Posteriors for a cell, from the cache or generated synthetically.

    Raises:
        FileNotFoundError: naming the cache file that was expected. There is no
            silent recomputation: a missing cache means the caching script has
            not been run for this configuration.
        KeyError: if the config names no cache hash for this cell's decoder.
    """
    variant = variant_for(cell, resolve_variants(cfg))

    if bool(cfg.synthetic.enabled):
        rng = generator_for(int(cfg.seed), "synthetic", cell.subject, cell.decoder, variant.key)
        arrays = synthetic_arrays(
            rng,
            n_bursts=int(cfg.synthetic.n_bursts),
            n_windows=int(cfg.synthetic.n_windows),
            n_classes=int(cfg.data.n_classes),
            accuracy=float(cfg.synthetic.accuracy),
            window_s=variant.window_s,
            stride_s=variant.stride_s,
            confidence=float(cfg.synthetic.confidence),
        )
        return arrays, float("nan")

    path = posterior_path(
        Path(cfg.paths.artifacts),
        subject=cell.subject,
        session=str(cfg.session),
        decoder=cell.decoder,
        window_ms=round(variant.window_s * 1000),
        stride_ms=round(variant.stride_s * 1000),
        cfg_hash=variant.hash_for(cell.decoder),
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


def cache_key(cell: Cell) -> tuple[Any, ...]:
    """Which posterior file a cell reads.

    Cells sharing this key share one loaded array set, which is the whole point
    of caching posteriors: every mapping, quality level and seed of one subject
    replays the same decoded windows.
    """
    return (cell.subject, cell.decoder, variant_key(cell.protocol, cell.window_s))


def run_cells(cfg: DictConfig, cells: Sequence[Cell]) -> list[EpisodeResult]:
    """Run a block of cells in this process, loading each posterior file once.

    The unit of work a worker receives. Cells are expected to be grouped by
    `cache_key`, but correctness does not depend on it: an ungrouped block only
    reloads more often.
    """
    cache: dict[tuple[Any, ...], tuple[dict[str, np.ndarray], float]] = {}
    results: list[EpisodeResult] = []
    for cell in cells:
        key = cache_key(cell)
        if key not in cache:
            cache[key] = load_arrays(cell, cfg)
        arrays, kappa = cache[key]
        results.append(run_episode(cell, arrays, cfg, kappa_offline=kappa))
    return results


def iter_results(cfg: DictConfig) -> Iterator[EpisodeResult]:
    """Run every cell in this process, yielding one result each, in grid order.

    Fails fast: an exception in one episode aborts the run. A partially completed
    matrix silently missing cells is worse than no results at all.
    """
    cells = enumerate_cells(cfg)
    logger.info("%s: %d episodes, serial", cfg.name, len(cells))

    cache: dict[tuple[Any, ...], tuple[dict[str, np.ndarray], float]] = {}
    every = int(cfg.progress_every)
    started = time.perf_counter()

    for index, cell in enumerate(cells, start=1):
        key = cache_key(cell)
        if key not in cache:
            cache[key] = load_arrays(cell, cfg)
        arrays, kappa = cache[key]

        yield run_episode(cell, arrays, cfg, kappa_offline=kappa)

        if every and index % every == 0:
            _log_progress(index, len(cells), started)


def _log_progress(done: int, total: int, started: float) -> None:
    elapsed = time.perf_counter() - started
    remaining = elapsed / done * (total - done)
    logger.info(
        "%d/%d episodes, %.1fs elapsed, about %.0fs remaining", done, total, elapsed, remaining
    )


def plan_chunks(cells: Sequence[Cell], n_workers: int) -> list[list[Cell]]:
    """Split cells into worker-sized blocks that share a posterior file.

    Grouped by `cache_key` first, so a worker loads one file rather than one per
    episode. Split further into several blocks per worker, because episode cost
    varies by an order of magnitude between a decoder that acquires targets and
    one that times out on every one of them, and one block per worker would
    leave seven workers idle behind the slowest.
    """
    if n_workers < 1:
        raise ValueError(f"n_workers must be at least 1, got {n_workers}")

    groups: dict[tuple[Any, ...], list[Cell]] = {}
    for cell in cells:
        groups.setdefault(cache_key(cell), []).append(cell)

    target_blocks = max(n_workers * _BLOCKS_PER_WORKER, 1)
    size = max(1, len(cells) // target_blocks)

    chunks: list[list[Cell]] = []
    for group in groups.values():
        for start in range(0, len(group), size):
            chunks.append(group[start : start + size])
    return chunks


def run_all(cfg: DictConfig) -> list[EpisodeResult]:
    """Every episode of the configured experiment, in grid order.

    Parallel across processes when `n_workers` is above one. The result is
    identical either way: an episode's random stream is derived from the content
    of its cell, not from its position in the queue or the worker that drew it,
    so `n_workers` cannot change a number. There is a test that asserts it.
    """
    n_workers = _worker_count(cfg)
    if n_workers <= 1:
        return list(iter_results(cfg))

    cells = enumerate_cells(cfg)
    chunks = plan_chunks(cells, n_workers)
    # No point starting a process that would have nothing to do; on Windows each
    # one re-imports the package before it can run a single episode.
    n_workers = min(n_workers, len(chunks))
    logger.info(
        "%s: %d episodes, %d workers, %d blocks", cfg.name, len(cells), n_workers, len(chunks)
    )

    every = int(cfg.progress_every)
    started = time.perf_counter()
    collected: list[EpisodeResult] = []
    logged = 0

    executor = ProcessPoolExecutor(max_workers=n_workers)
    futures: list[Future[list[EpisodeResult]]] = [
        executor.submit(run_cells, cfg, chunk) for chunk in chunks
    ]
    try:
        for future in as_completed(futures):
            # `.result()` re-raises whatever the worker raised. Nothing catches
            # it: a matrix missing cells is worse than no matrix.
            collected.extend(future.result())
            if every and len(collected) - logged >= every:
                logged = len(collected)
                _log_progress(logged, len(cells), started)
    except BaseException:
        # Do not wait for the survivors once a worker has failed; the answer is
        # not going to improve. On the way out of a successful run the pool is
        # joined instead, so no process is left for the interpreter to reap.
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    executor.shutdown(wait=True)

    # Sorted back into grid order rather than completion order, so two runs of
    # the same config produce identical Parquet contents. Chunk order is not
    # grid order: chunks are grouped by posterior file, and the protocol and
    # window axes vary inside the subject loop.
    position = {cell: index for index, cell in enumerate(cells)}
    collected.sort(key=lambda result: position[result.cell])
    return collected


def sanity_variant(cfg: DictConfig) -> ReplayVariant:
    """The replay condition the sanity checks run under: the grid's first.

    The checks verify the environment and the arbitration, neither of which is a
    property of the protocol, so one condition settles them. The first rather
    than an arbitrary one so the choice is stated by the config.
    """
    variants = resolve_variants(cfg)
    key = variant_key(str(cfg.grid.protocols[0]), float(cfg.grid.windows_s[0]))
    if key not in variants:
        raise KeyError(f"the grid's first replay condition {key!r} has no variant in the config")
    return variants[key]


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
    variant = sanity_variant(cfg)
    rates: list[float] = []
    for seed in range(n_episodes):
        cell = Cell(
            experiment=f"sanity_{accuracy}_{confidence}",
            subject=int(cfg.grid.subjects[0]),
            decoder="synthetic",
            mapping=mapping,
            quality_level=quality_level,
            protocol=variant.protocol,
            window_s=variant.window_s,
            error_struct=ERROR_NONE,
            intent_mode=None,
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
            window_s=variant.window_s,
            stride_s=variant.stride_s,
            confidence=confidence,
        )
        rates.append(run_episode(cell, arrays, cfg).metrics.success_rate)
    return rates


def _identity_config(cfg: DictConfig) -> DictConfig:
    """A two-target environment for the two arbitration identities.

    Both are exact identities rather than statistics: either the trajectories
    match or they do not, and two targets settle it as conclusively as eight.
    Running them on the full task would only make the smoke slower.

    Two rather than one because a one-choice task carries no information and the
    ITR metric refuses it, which is the metric being right rather than a limit.
    """
    # A short timeout as well: the identity holds or fails on the first
    # divergent step, so there is nothing to gain from letting the episode run
    # to completion.
    reduced = OmegaConf.merge(cfg, {"env": {"n_targets": 2, "timeout_s": 30.0}})
    assert isinstance(reduced, DictConfig)
    return reduced


def _sanity_episode(
    cfg: DictConfig,
    mapping: str,
    arrays: dict[str, np.ndarray],
    *,
    seed: int = 0,
    alpha: float | None = None,
    intent_mode: str | None = None,
) -> EpisodeMetrics:
    """One episode under a named mapping, for the arbitration sanity checks.

    Every call shares one random stream, so the only difference between two
    episodes is the mapping under test.
    """
    variant = sanity_variant(cfg)
    cell = Cell(
        experiment="sanity_arbitration",
        subject=int(cfg.grid.subjects[0]),
        decoder="synthetic",
        mapping=mapping,
        quality_level=0.0,
        protocol=variant.protocol,
        window_s=variant.window_s,
        error_struct=ERROR_NONE,
        intent_mode=intent_mode,
        alpha=alpha,
        latency_ms=int(cfg.grid.latencies_ms[0]),
        seed=seed,
    )
    return run_episode(cell, arrays, cfg, rng_key=("sanity_arbitration", seed)).metrics


def _sanity_arrays(cfg: DictConfig, accuracy: float, tag: str) -> dict[str, np.ndarray]:
    variant = sanity_variant(cfg)
    return synthetic_arrays(
        generator_for(int(cfg.seed), "sanity", tag, accuracy),
        n_bursts=int(cfg.synthetic.n_bursts),
        n_windows=int(cfg.synthetic.n_windows),
        n_classes=int(cfg.data.n_classes),
        accuracy=accuracy,
        window_s=variant.window_s,
        stride_s=variant.stride_s,
        confidence=float(cfg.synthetic.confidence),
    )


def alpha_zero_matches_s2(cfg: DictConfig) -> dict[str, Any]:
    """S4 at alpha 0 must reproduce S2 exactly, trajectory for trajectory.

    Catches an arbitration sign error, which would otherwise show up as S4 simply
    performing differently from S2 and be indistinguishable from a real effect.
    """
    reduced = _identity_config(cfg)
    arrays = _sanity_arrays(cfg, 0.75, "alpha0")
    shared = _sanity_episode(reduced, "s4_shared", arrays, alpha=0.0)
    weighted = _sanity_episode(reduced, "s2_weighted", arrays)

    difference = abs(shared.success_rate - weighted.success_rate) + abs(
        shared.episode_duration_s - weighted.episode_duration_s
    )
    return {
        "max_abs_diff": round(float(difference), 9),
        "s4_success": round(shared.success_rate, 6),
        "s2_success": round(weighted.success_rate, 6),
        "pass": difference == 0.0,
        "applicable": True,
    }


def alpha_one_ignores_the_decoder(cfg: DictConfig) -> dict[str, Any]:
    """S4 at alpha 1, intent-blind, must give the same episode for any decoder.

    Checked on the intent-blind variant only. Intent-aware chooses its attractive
    target from the posterior, so it is decoder-dependent at alpha 1 by
    construction; that dependence is the leakage the ablation exists to measure,
    not a fault to check for here. See docs/decisions.md D43.
    """
    reduced = _identity_config(cfg)
    weak = _sanity_arrays(cfg, 0.4, "alpha1_weak")
    strong = _sanity_arrays(cfg, 0.95, "alpha1_strong")

    first = _sanity_episode(reduced, "s4_shared", weak, alpha=1.0, intent_mode="blind")
    second = _sanity_episode(reduced, "s4_shared", strong, alpha=1.0, intent_mode="blind")

    difference = abs(first.success_rate - second.success_rate) + abs(
        first.episode_duration_s - second.episode_duration_s
    )
    return {
        "max_abs_diff": round(float(difference), 9),
        "weak_decoder_success": round(first.success_rate, 6),
        "strong_decoder_success": round(second.success_rate, 6),
        "pass": difference == 0.0,
        "applicable": True,
    }


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
        "alpha0_equals_s2": alpha_zero_matches_s2(cfg),
        "alpha1_independence": alpha_one_ignores_the_decoder(cfg),
    }
    checks["all_passed"] = all(
        entry["pass"]
        for entry in checks.values()
        if isinstance(entry, dict) and entry["applicable"]
    )
    return checks
