"""Run output: `episodes.parquet`, `summary.json`, `config.yaml`.

The schema is the contract in `agents/05-experiments-and-outputs.md` §3, asserted
column by column so a renamed or reordered column fails here rather than in the
analysis six weeks later.

A run directory is written atomically: everything goes into a temporary sibling
and is renamed on success. A crashed run therefore leaves no half-written
directory for a later analysis to read as complete.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

import micm
from micm.eval.aggregate import scores_block
from micm.eval.runner import EpisodeResult, resolve_variants, variant_for
from micm.utils.config import save_config
from micm.utils.hashing import config_hash
from micm.utils.logging import get_logger

logger = get_logger(__name__)

# Column name to pandas dtype, in the order the spec lists them. `quality_level`
# is an addition; see docs/decisions.md D35.
EPISODE_SCHEMA: Final[dict[str, str]] = {
    "experiment": "string",
    "run_id": "string",
    "subject": "int8",
    "decoder": "string",
    "mapping": "string",
    "quality_level": "float32",
    "lam": "float32",
    "alpha": "float32",
    "window_s": "float32",
    "stride_s": "float32",
    "latency_ms": "int16",
    "protocol": "string",
    "error_struct": "string",
    "intent_mode": "string",
    "seed": "int32",
    "direction_perm": "string",
    "kappa_offline": "float32",
    "effective_acc": "float32",
    # An addition to the contract, like `quality_level` (D35): the figure
    # `agents/05` §6 asks for plots latency against it, and no column carried
    # it. See docs/decisions.md D57.
    "command_accuracy": "float32",
    "success_rate": "float32",
    "n_success": "int8",
    "time_to_target_median": "float32",
    "n_timeouts": "int8",
    "path_efficiency": "float32",
    "direction_reversals": "float32",
    "collisions": "int16",
    "effective_itr": "float32",
    "user_contribution_index": "float32",
    "uci_excluded_frac": "float32",
    "bursts_without_command": "int16",
    "episode_duration_s": "float32",
    "wall_time_s": "float32",
}

@dataclass(frozen=True)
class RunResult:
    """Where a run was written and what it found."""

    directory: Path
    run_id: str
    config_hash: str
    episodes: pd.DataFrame
    summary: dict[str, Any]


def run_id_for(cfg_hash: str, *, now: datetime | None = None) -> str:
    """`{YYYYMMDD-HHMMSS}-{cfghash}`, the only place a timestamp appears in a name."""
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{cfg_hash}"


def episodes_frame(
    results: Sequence[EpisodeResult], cfg: DictConfig, run_id: str
) -> pd.DataFrame:
    """Build `episodes.parquet` from episode results.

    Flat, one row per episode, every independent variable its own column, so the
    file can be grouped without joining anything. `kappa_offline` and
    `effective_acc` sit in the same row as the closed-loop metrics, which is what
    makes the H1 regression a one-liner.

    Raises:
        ValueError: if the frame does not match `EPISODE_SCHEMA` exactly.
    """
    variants = resolve_variants(cfg)
    rows = []
    for result in results:
        cell, metrics = result.cell, result.metrics
        # From the cell's own replay variant, not from `cfg.replay`, which can
        # only describe one condition while two ablations vary it. Reading it
        # from the config would leave every row of those ablations claiming the
        # window the config happened to compose. See docs/decisions.md D45.
        variant = variant_for(cell, variants)
        rows.append(
            {
                "experiment": cell.experiment,
                "run_id": run_id,
                "subject": cell.subject,
                "decoder": cell.decoder,
                "mapping": cell.mapping,
                "quality_level": cell.quality_level,
                "lam": result.lam,
                "alpha": np.nan if cell.alpha is None else cell.alpha,
                "window_s": variant.window_s,
                "stride_s": variant.stride_s,
                "latency_ms": cell.latency_ms,
                "protocol": cell.protocol,
                "error_struct": cell.error_struct,
                "intent_mode": cell.intent_mode,  # None becomes <NA> in a string column
                "seed": cell.seed,
                "direction_perm": result.direction_perm,
                "kappa_offline": result.kappa_offline,
                "effective_acc": result.effective_acc,
                "command_accuracy": result.command_accuracy,
                "success_rate": metrics.success_rate,
                "n_success": metrics.n_success,
                "time_to_target_median": metrics.time_to_target_median,
                "n_timeouts": metrics.n_timeouts,
                "path_efficiency": metrics.path_efficiency,
                "direction_reversals": metrics.direction_reversals,
                "collisions": metrics.collisions,
                "effective_itr": metrics.effective_itr,
                "user_contribution_index": metrics.user_contribution_index,
                "uci_excluded_frac": metrics.uci_excluded_frac,
                "bursts_without_command": result.bursts_without_command,
                "episode_duration_s": metrics.episode_duration_s,
                "wall_time_s": result.wall_time_s,
            }
        )

    frame = pd.DataFrame(rows, columns=list(EPISODE_SCHEMA))
    validate_schema(frame)
    return frame.astype(EPISODE_SCHEMA)


def validate_schema(frame: pd.DataFrame) -> None:
    """Assert the frame carries exactly the contracted columns, in order.

    Raises:
        ValueError: on a missing, extra, or reordered column.
    """
    actual = list(frame.columns)
    expected = list(EPISODE_SCHEMA)
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ValueError(
            f"episodes schema mismatch: missing {missing}, extra {extra}, "
            f"or order differs from the contract"
        )



@dataclass(frozen=True)
class LoadedRun:
    """A run directory read back: the episodes, the config, and the summary.

    The reader lives beside the writer because they share one format. Both the
    analysis script and the figures read runs, and two loaders would drift.
    """

    directory: Path
    episodes: pd.DataFrame
    config: DictConfig
    summary: dict[str, Any]

    @property
    def experiment(self) -> str:
        return str(self.config.name)


def load_run(directory: Path) -> LoadedRun:
    """Read a run directory.

    Raises:
        FileNotFoundError: naming whichever of the three files is missing. A run
            directory is renamed into place only once it is complete, so a
            missing file means this is not a run directory rather than that the
            run stopped half way.
        ValueError: if the episode frame does not match the schema contract, so
            a figure or a model never sees a frame the writer would have
            rejected.
    """
    for name in ("episodes.parquet", "summary.json", "config.yaml"):
        if not (directory / name).is_file():
            raise FileNotFoundError(f"{directory} has no {name}; is it a run directory?")

    frame = pd.read_parquet(directory / "episodes.parquet")
    validate_schema(frame)

    config = OmegaConf.load(directory / "config.yaml")
    if not isinstance(config, DictConfig):
        raise TypeError(f"{directory / 'config.yaml'} did not resolve to a mapping")

    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    return LoadedRun(directory=directory, episodes=frame, config=config, summary=summary)


def write_summary(directory: Path, summary: dict[str, Any]) -> None:
    """Replace a run's `summary.json` atomically.

    Written to a sibling and renamed, so an interrupted rewrite leaves the
    previous summary rather than a file that parses as valid JSON up to the
    point it stops.
    """
    staging = directory / ".summary.json.partial"
    staging.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8"
    )
    staging.replace(directory / "summary.json")


def host_block() -> dict[str, Any]:
    """The machine the run executed on, per `agents/05` §3.

    The GPU is reported because the config records the device that was
    *requested* and `resolve_device` may have fallen back to CPU with a warning
    that only reached the log. A summary that says `device: cuda` while the run
    was on CPU is a wall-time figure nobody can interpret later.

    Torch is imported here rather than at module scope: it is the optional `dl`
    extra, and the runner half of this project installs without it. Whichever of
    the two reasons applies is written down rather than left as an absent key.
    """
    try:
        import torch
    except ImportError:
        gpu: str = "torch is not installed"
    else:
        gpu = (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else f"torch {torch.__version__} reports no CUDA device"
        )

    return {
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "gpu": gpu,
    }


def meta_block(
    cfg: DictConfig, frame: pd.DataFrame, *, run_id: str, cfg_hash: str, wall_time_s: float
) -> dict[str, Any]:
    return {
        "experiment": str(cfg.name),
        "run_id": run_id,
        "config_hash": cfg_hash,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "micm_version": micm.__version__,
        "python": sys.version.split()[0],
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "host": host_block(),
        # What the score intervals were resampled from. It lives here rather
        # than inside `scores`, which `agents/05` §3 fixes at three keys.
        "aggregate": {
            "n_boot": int(cfg.aggregate.n_boot),
            "ci": float(cfg.aggregate.ci),
            "min_subjects": int(cfg.aggregate.min_subjects),
            "resampled": "subjects",
        },
        "n_episodes": len(frame),
        "wall_time_s": round(wall_time_s, 6),
        "seed": int(cfg.seed),
        "synthetic": bool(cfg.synthetic.enabled),
    }


def write_run(
    results: Sequence[EpisodeResult],
    cfg: DictConfig,
    *,
    sanity: dict[str, Any],
    wall_time_s: float,
    stats: dict[str, Any] | None = None,
) -> RunResult:
    """Write one run directory atomically and return where it went.

    Raises:
        ValueError: if the episode frame does not match the schema contract.
    """
    cfg_hash = config_hash(cfg)
    run_id = run_id_for(cfg_hash)
    frame = episodes_frame(results, cfg, run_id)

    summary = {
        "meta": meta_block(cfg, frame, run_id=run_id, cfg_hash=cfg_hash, wall_time_s=wall_time_s),
        "scores": scores_block(
            frame,
            n_boot=int(cfg.aggregate.n_boot),
            ci=float(cfg.aggregate.ci),
            min_subjects=int(cfg.aggregate.min_subjects),
            seed=int(cfg.seed),
        ),
        "stats": stats
        if stats is not None
        else {"model": None, "note": "fitted by scripts/04_analyze.py, which lands in T13"},
        "sanity": sanity,
    }

    final = Path(cfg.paths.artifacts) / "runs" / str(cfg.name) / run_id
    staging = final.with_name(f".{final.name}.partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    frame.to_parquet(staging / "episodes.parquet", index=False)
    (staging / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8"
    )
    save_config(cfg, staging / "config.yaml")

    final.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(final)

    if not sanity.get("all_passed", False):
        logger.warning(
            "SANITY CHECKS FAILED for %s. The run was still written, and summary.json "
            "records which check failed; do not use these numbers until it is explained",
            run_id,
        )
    logger.info("wrote %d episodes to %s", len(frame), final)
    return RunResult(
        directory=final, run_id=run_id, config_hash=cfg_hash, episodes=frame, summary=summary
    )
