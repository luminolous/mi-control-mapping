"""Aggregation of `episodes.parquet` into the `scores` block.

Two rules from `agents/05` §4 shape everything here.

**Resample subjects, never episodes.** The ninety episodes of one subject share
a decoder, a posterior file and that subject's own aptitude. Treating them as
ninety independent observations produces an interval several times narrower than
the evidence supports, and the effects in this project are small enough for that
to decide conclusions.

**Never report an average across subjects without the breakdown beside it.**
Between-subject variance on IV-2a exceeds most effects of interest, so a mean
alone invites reading a subject effect as a condition effect.

A consequence of the first rule: with fewer than a handful of subjects the
interval is not defined, and this module says so rather than returning a
zero-width interval that looks like precision.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd

from micm.utils.logging import get_logger
from micm.utils.seeding import generator_for

logger = get_logger(__name__)

# Metrics summarised in the scores block.
#
# `effective_acc` is not a closed-loop outcome but the achieved posterior
# accuracy of the cell. It is here because the main figure plots success rate
# against it, and `agents/05` §6 requires a figure to read a number rather than
# derive one: without it the x axis would have to be recomputed from the
# Parquet by every figure that wants it.
SCORE_COLUMNS: Final[tuple[str, ...]] = (
    "effective_acc",
    "command_accuracy",
    "success_rate",
    "path_efficiency",
    "time_to_target_median",
    "direction_reversals",
    "effective_itr",
    "user_contribution_index",
)

SUBJECT: Final[str] = "subject"


@dataclass(frozen=True)
class Estimate:
    """A mean with an interval, or a mean with a stated reason for having none."""

    mean: float
    # Both over the per-subject means. `n` counts episodes for context only.
    sd: float
    n: int
    n_subjects: int
    ci95: tuple[float, float] | None
    ci_note: str | None

    def as_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "mean": round(self.mean, 6),
            "sd": round(self.sd, 6),
            "n": self.n,
            "n_subjects": self.n_subjects,
            "ci95": None if self.ci95 is None else [round(self.ci95[0], 6), round(self.ci95[1], 6)],
        }
        if self.ci_note is not None:
            payload["ci_note"] = self.ci_note
        return payload


def subject_means(frame: pd.DataFrame, column: str) -> pd.Series:
    """One value per subject: the mean of that subject's episodes.

    The unit of analysis. Everything downstream resamples these, so a subject
    with more episodes than another does not get more say.
    """
    clean = frame[[SUBJECT, column]].dropna()
    return clean.groupby(SUBJECT)[column].mean()


def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    *,
    n_boot: int,
    ci: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Percentile interval from resampling `values` with replacement.

    `values` is one number per subject. Resampling a list of subject means is
    the whole point; passing episode-level values here would defeat it.

    Raises:
        ValueError: on fewer than two values, where a bootstrap interval is a
            statement about a single number.
    """
    array = np.asarray(values, dtype=np.float64)
    if len(array) < 2:
        raise ValueError(f"a bootstrap interval needs at least two values, got {len(array)}")

    draws = rng.integers(0, len(array), size=(n_boot, len(array)))
    means = array[draws].mean(axis=1)
    lower = (1.0 - ci) / 2.0
    return float(np.quantile(means, lower)), float(np.quantile(means, 1.0 - lower))


def estimate(
    frame: pd.DataFrame,
    column: str,
    *,
    n_boot: int,
    ci: float,
    min_subjects: int,
    rng: np.random.Generator,
) -> Estimate | None:
    """Mean and subject-bootstrapped interval for one metric, or None if all NaN.

    Returns None rather than a NaN-filled block when the metric never had a
    value: `time_to_target_median` is NaN for an episode that acquired nothing,
    and a group of only such episodes has no median time to report.
    """
    clean = frame[[SUBJECT, column]].dropna()
    if clean.empty:
        return None

    per_subject = subject_means(frame, column)
    values = clean[column].to_numpy(dtype=np.float64)
    # Both over subject means, not over episodes. The subject is the unit of
    # analysis here, and a mean taken one way beside a spread taken the other
    # would describe two different quantities under one heading.
    mean = float(per_subject.mean())
    sd = float(per_subject.std(ddof=1)) if len(per_subject) > 1 else 0.0

    if len(per_subject) < max(min_subjects, 2):
        return Estimate(
            mean=mean,
            sd=sd,
            n=len(values),
            n_subjects=len(per_subject),
            ci95=None,
            ci_note=(
                f"{len(per_subject)} subject(s); a between-subject interval needs "
                f"at least {max(min_subjects, 2)}"
            ),
        )

    return Estimate(
        mean=mean,
        sd=sd,
        n=len(values),
        n_subjects=len(per_subject),
        ci95=bootstrap_ci(per_subject.to_numpy(), n_boot=n_boot, ci=ci, rng=rng),
        ci_note=None,
    )


def _plain(value: Any) -> Any:
    """A grouping key as something `json.dumps` will accept.

    NaN becomes None rather than a bare `nan`: an `alpha` of NaN means the
    mapping has no autonomy weight, and the summary should say so in JSON's own
    vocabulary rather than in one JSON cannot parse back.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return value.item() if hasattr(value, "item") else value


def _group_entries(
    frame: pd.DataFrame,
    by: Sequence[str],
    *,
    n_boot: int,
    ci: float,
    min_subjects: int,
    seed: int,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for key, group in frame.groupby(list(by), dropna=False, observed=True):
        keys = key if isinstance(key, tuple) else (key,)
        entry: dict[str, Any] = {
            name: _plain(value) for name, value in zip(by, keys, strict=True)
        }
        entry["n_episodes"] = len(group)
        for column in SCORE_COLUMNS:
            # A generator per group and metric, so adding a metric or reordering
            # the groups does not move the intervals of the others.
            rng = generator_for(seed, "bootstrap", *[str(value) for value in keys], column)
            found = estimate(
                group,
                column,
                n_boot=n_boot,
                ci=ci,
                min_subjects=min_subjects,
                rng=rng,
            )
            entry[column] = None if found is None else found.as_json()
        entries.append(entry)
    return entries


def scores_block(
    frame: pd.DataFrame,
    *,
    n_boot: int = 1000,
    ci: float = 0.95,
    min_subjects: int = 3,
    seed: int = 1337,
) -> dict[str, Any]:
    """The `scores` block of `summary.json`.

    Exactly the three keys `agents/05` §3 lists. What the intervals were
    computed from is recorded in the `meta` block instead, so this one keeps the
    shape the contract gives it.

    `by_cell` is grouped by the axes that define a condition rather than by
    every column, so a grid that does not vary an axis still produces one entry
    per condition. `by_subject` is always present next to `overall`.
    """
    overall_rng = generator_for(seed, "bootstrap", "overall")
    return {
        "by_cell": _group_entries(
            frame, _cell_axes(frame), n_boot=n_boot, ci=ci, min_subjects=min_subjects, seed=seed
        ),
        "by_subject": _group_entries(
            frame, [SUBJECT], n_boot=n_boot, ci=ci, min_subjects=min_subjects, seed=seed
        ),
        "overall": {
            column: (
                None
                if (
                    found := estimate(
                        frame,
                        column,
                        n_boot=n_boot,
                        ci=ci,
                        min_subjects=min_subjects,
                        rng=overall_rng,
                    )
                )
                is None
                else found.as_json()
            )
            for column in SCORE_COLUMNS
        },
    }


# Axes that can define a condition. Only those that actually vary in the frame
# are used to group, so `main` groups by mapping and decoder while
# `ablation_latency` also groups by latency, without either config having to say
# which axes it varies.
_CANDIDATE_AXES: Final[tuple[str, ...]] = (
    "mapping",
    "decoder",
    "quality_level",
    "alpha",
    "intent_mode",
    "protocol",
    "window_s",
    "latency_ms",
    "error_struct",
)


def _cell_axes(frame: pd.DataFrame) -> list[str]:
    """Which axes vary in this run, in a fixed order.

    Always at least the mapping: a grid with one mapping and one decoder would
    otherwise group by nothing and report a single cell identical to `overall`.
    An axis the frame does not carry is skipped rather than raising, so a
    hand-built frame can be aggregated without carrying every column of the
    episode schema.
    """
    varying = [
        axis
        for axis in _CANDIDATE_AXES
        if axis in frame.columns and frame[axis].nunique(dropna=False) > 1
    ]
    return varying or ["mapping"]
