"""The posterior cache: the interface between the two halves of the project.

Everything upstream of this file reads EEG; everything downstream reads only
these `.npz` files. Getting the schema exactly right matters more than anything
else in this module, because a column that means something slightly different
from what the reader assumes produces plausible numbers rather than an error.

Two rules the writer enforces. A cache file is never overwritten in place: the
name carries the config hash, so a different result means a different name, and
a collision means something is wrong rather than something is stale. And every
file carries its own metadata, because a file that cannot say where it came from
will be misread six weeks later.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
from omegaconf import DictConfig

import micm
from micm.utils.config import resolved_container
from micm.utils.hashing import config_hash
from micm.utils.logging import get_logger

logger = get_logger(__name__)

# Every array a posterior file must contain, with the dtype it must have.
# Asserted on both write and read: the schema is the contract, so it is checked
# from both sides rather than trusted.
SCHEMA: Final[dict[str, np.dtype[Any]]] = {
    "posterior": np.dtype(np.float32),
    "label": np.dtype(np.int8),
    "burst_id": np.dtype(np.int32),
    "t_rel": np.dtype(np.float32),
    "burst_onset": np.dtype(np.float32),
    "burst_offset": np.dtype(np.float32),
}
META_KEY: Final[str] = "meta"
ROW_SUM_TOL: Final[float] = 1e-5


@dataclass(frozen=True)
class PosteriorFile:
    """The contents of one cached posterior file.

    Attributes:
        posterior: (n_windows, K) float32, rows sum to 1.
        label: (n_windows,) int8, ground-truth class of the parent burst.
        burst_id: (n_windows,) int32, which burst the window belongs to.
        t_rel: (n_windows,) float32, seconds from burst onset to the window END.
        burst_onset: (n_bursts,) float32, burst starts on the session timeline.
        burst_offset: (n_bursts,) float32, burst ends on that timeline.
        meta: the JSON metadata, already parsed.
        path: where it was read from.
    """

    posterior: np.ndarray
    label: np.ndarray
    burst_id: np.ndarray
    t_rel: np.ndarray
    burst_onset: np.ndarray
    burst_offset: np.ndarray
    meta: dict[str, Any]
    path: Path

    @property
    def n_windows(self) -> int:
        return len(self.posterior)

    @property
    def n_classes(self) -> int:
        return int(self.posterior.shape[1])

    @property
    def n_bursts(self) -> int:
        return len(self.burst_onset)


def posterior_filename(
    *, subject: int, session: str, decoder: str, window_ms: int, stride_ms: int, cfg_hash: str
) -> str:
    """Deterministic, parseable name. No timestamp: the hash identifies the run."""
    return f"{subject:02d}_{session}_{decoder}_{window_ms}_{stride_ms}_{cfg_hash}.npz"


def posterior_path(
    artifacts_root: Path,
    *,
    subject: int,
    session: str,
    decoder: str,
    window_ms: int,
    stride_ms: int,
    cfg_hash: str,
) -> Path:
    """Full path of a cached posterior file under `artifacts/posteriors/`."""
    return (
        Path(artifacts_root)
        / "posteriors"
        / posterior_filename(
            subject=subject,
            session=session,
            decoder=decoder,
            window_ms=window_ms,
            stride_ms=stride_ms,
            cfg_hash=cfg_hash,
        )
    )


def build_meta(cfg: DictConfig, **extra: Any) -> dict[str, Any]:
    """Metadata embedded in every cache file.

    Carries the fully resolved config rather than a reference to it, so the file
    stays interpretable after the config on disk has moved on.
    """
    return {
        "config": resolved_container(cfg),
        "config_hash": config_hash(cfg),
        "micm_version": micm.__version__,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **extra,
    }


def validate_arrays(
    *,
    posterior: np.ndarray,
    label: np.ndarray,
    burst_id: np.ndarray,
    t_rel: np.ndarray,
    burst_onset: np.ndarray,
    burst_offset: np.ndarray,
) -> None:
    """Check the schema invariants that downstream code relies on.

    Raises:
        ValueError: on a shape mismatch, a bad row sum, a NaN, a burst id
            outside the burst table, or windows that are not ordered by time.
    """
    if posterior.ndim != 2:
        raise ValueError(f"posterior must be (n_windows, K), got {posterior.shape}")

    n_windows = len(posterior)
    for name, array in (("label", label), ("burst_id", burst_id), ("t_rel", t_rel)):
        if array.shape != (n_windows,):
            raise ValueError(f"{name} must be ({n_windows},), got {array.shape}")

    if burst_onset.shape != burst_offset.shape:
        raise ValueError(
            f"burst_onset {burst_onset.shape} and burst_offset {burst_offset.shape} differ"
        )
    if (burst_offset < burst_onset).any():
        raise ValueError("a burst ends before it starts")

    if not np.isfinite(posterior).all():
        raise ValueError("posterior contains NaN or inf")
    if (posterior < 0.0).any():
        raise ValueError("posterior contains negative probabilities")
    sums = posterior.sum(axis=1)
    if n_windows and not np.allclose(sums, 1.0, atol=ROW_SUM_TOL):
        raise ValueError(
            f"posterior rows must sum to 1, worst deviation {float(np.max(np.abs(sums - 1.0))):.3e}"
        )

    if n_windows:
        n_bursts = len(burst_onset)
        if int(burst_id.min()) < 0 or int(burst_id.max()) >= n_bursts:
            raise ValueError(
                f"burst_id spans [{int(burst_id.min())}, {int(burst_id.max())}] but there "
                f"are {n_bursts} bursts"
            )
        # Windows are ordered by time within a burst; t_rel must not go backwards
        # inside one, or the latency buffer would reject the sequence.
        for burst in np.unique(burst_id):
            times = t_rel[burst_id == burst]
            if (np.diff(times) <= 0).any():
                raise ValueError(f"t_rel is not strictly increasing within burst {int(burst)}")


def write_posteriors(
    path: Path,
    *,
    posterior: np.ndarray,
    label: np.ndarray,
    burst_id: np.ndarray,
    t_rel: np.ndarray,
    burst_onset: np.ndarray,
    burst_offset: np.ndarray,
    meta: dict[str, Any],
    force: bool = False,
) -> Path:
    """Write one posterior cache file, atomically.

    Writes to a temporary name in the destination directory and renames on
    success, so a crash leaves no half-written cache file for a later run to
    read as valid.

    Args:
        force: overwrite an existing file. Only ever set from the deliberate
            `--force` flag of the caching script.

    Raises:
        FileExistsError: if the file exists and `force` is false. The name
            carries the config hash, so an existing file with the same name was
            produced by the same config; recomputing it is a waste, and
            overwriting it silently would hide a hash collision.
        ValueError: if the arrays fail `validate_arrays`.
    """
    arrays = {
        "posterior": np.asarray(posterior, dtype=np.float32),
        "label": np.asarray(label, dtype=np.int8),
        "burst_id": np.asarray(burst_id, dtype=np.int32),
        "t_rel": np.asarray(t_rel, dtype=np.float32),
        "burst_onset": np.asarray(burst_onset, dtype=np.float32),
        "burst_offset": np.asarray(burst_offset, dtype=np.float32),
    }
    validate_arrays(**arrays)

    path = Path(path)
    if path.exists() and not force:
        raise FileExistsError(
            f"{path} already exists. Its name carries the config hash, so the same "
            "config produced it; pass force=True only to recompute deliberately"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    payload: dict[str, Any] = {**arrays, META_KEY: json.dumps(meta, sort_keys=True)}
    np.savez_compressed(temporary, **payload)

    # np.savez appends .npz when the name lacks it; ours already has one inside.
    written = temporary if temporary.exists() else temporary.with_suffix(temporary.suffix + ".npz")
    written.replace(path)

    logger.info("wrote %d windows to %s", len(arrays["posterior"]), path.name)
    return path


def read_posteriors(path: Path) -> PosteriorFile:
    """Read and validate one posterior cache file.

    Validation runs on read as well as on write. A file may have been produced by
    an older version of this code, and the reader is the last place to catch that
    before the numbers reach an experiment.

    Raises:
        FileNotFoundError: if `path` does not exist.
        KeyError: if an array or the metadata is missing.
        ValueError: if the dtypes or the invariants do not hold.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no posterior cache at {path}")

    with np.load(path, allow_pickle=False) as handle:
        missing = (set(SCHEMA) | {META_KEY}) - set(handle.files)
        if missing:
            raise KeyError(f"{path.name} is missing {sorted(missing)}")

        arrays = {name: handle[name] for name in SCHEMA}
        for name, expected in SCHEMA.items():
            if arrays[name].dtype != expected:
                raise ValueError(
                    f"{path.name}: {name} has dtype {arrays[name].dtype}, expected {expected}"
                )
        meta = json.loads(str(handle[META_KEY]))

    validate_arrays(**arrays)
    return PosteriorFile(**arrays, meta=meta, path=path)
