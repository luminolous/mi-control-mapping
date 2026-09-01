"""Models for H1, H2 and H3, fitted with statsmodels.

Three things in here are deliberate and worth reading before trusting a number.

**A degenerate response is refused, not fitted.** `success_rate` is the primary
metric `agents/05` §5 names, and it is also the one most likely to saturate: with
a timeout generous enough to be fair to the slower mappings, every mapping can
eventually reach every target (docs/decisions.md D42b). A slope fitted through a
column of identical values is not a small effect, it is no measurement at all, so
the fit raises and names the alternatives instead. Every run reports the spread
of each candidate response, so the choice is made from data and recorded.

**Every hypothesis block can be inapplicable.** H2 needs the accuracy dial to
move and H3 needs alpha to move, and only `lambda_sweep` and `alpha_sweep`
respectively do that. A block that cannot be fitted says so with a reason rather
than being omitted, so `summary.json` always states what was and was not tested.

**Bootstrap resamples subjects.** Episodes within a subject share a decoder, a
posterior file and that subject's own aptitude. See `eval/aggregate.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from omegaconf import DictConfig

from micm.eval.aggregate import SUBJECT
from micm.utils.logging import get_logger
from micm.utils.seeding import generator_for

logger = get_logger(__name__)

IDENTITY: Final[str] = "identity"
LOGIT: Final[str] = "logit"
LOG: Final[str] = "log"
TRANSFORMS: Final[tuple[str, str, str]] = (IDENTITY, LOGIT, LOG)

# Column the transformed response is fitted under, so a formula never has to
# name the transform and the coefficient table stays readable.
RESPONSE: Final[str] = "response"


# statsmodels files the random-intercept variance among the parameters under
# this name.
GROUP_VAR: Final[str] = "Group Var"


class NotFittableError(ValueError):
    """The data cannot support the model, with the reason in the message."""


def _finite(value: Any) -> float | None:
    """A float for JSON, or None where the model could not produce one.

    `summary.json` is written with NaN allowed, but a NaN there is not valid
    JSON for a strict reader, and `null` says the same thing in a vocabulary
    every parser has.
    """
    number = float(value)
    return None if not np.isfinite(number) else round(number, 6)


# --- transforms ---


def half_count_logit(values: np.ndarray, n_trials: int) -> np.ndarray:
    """Logit of a proportion measured from `n_trials`, with the half-count correction.

        p' = (p * n + 0.5) / (n + 1)

    A proportion out of eight targets hits 0 and 1 often, and the logit of
    either is infinite. The usual fix is to clamp to `[eps, 1 - eps]`, which
    makes eps decide how far those piled-up ends sit from everything else; for a
    metric that saturates, that choice is most of the effect. The half-count
    correction ties the adjustment to the number of trials the proportion came
    from, which is a fact about the experiment rather than a knob.
    See docs/decisions.md D49.

    Raises:
        ValueError: on a non-positive trial count or values outside [0, 1].
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be at least 1, got {n_trials}")
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size and (finite.min() < 0.0 or finite.max() > 1.0):
        raise ValueError(
            f"a logit response must lie in [0, 1], got [{finite.min()}, {finite.max()}]; "
            "an unbounded metric wants transform 'log' or 'identity'"
        )
    adjusted = (array * n_trials + 0.5) / (n_trials + 1.0)
    return np.log(adjusted / (1.0 - adjusted))


def apply_transform(values: np.ndarray, kind: str, *, n_trials: int) -> np.ndarray:
    """Transform a response before fitting.

    Raises:
        ValueError: on an unknown transform, or a non-positive value under `log`.
    """
    array = np.asarray(values, dtype=np.float64)
    if kind == IDENTITY:
        return array
    if kind == LOGIT:
        return half_count_logit(array, n_trials)
    if kind == LOG:
        finite = array[np.isfinite(array)]
        if finite.size and finite.min() <= 0.0:
            raise ValueError(
                f"transform 'log' needs positive values, got a minimum of {finite.min()}"
            )
        return np.log(array)
    raise ValueError(f"unknown transform {kind!r}, expected one of {list(TRANSFORMS)}")


# --- what the data can support ---


def spread(values: pd.Series) -> float:
    """Standard deviation over the non-null values, or 0 when there are none."""
    clean = values.dropna()
    return float(clean.std(ddof=1)) if len(clean) > 1 else 0.0


def response_diagnostics(frame: pd.DataFrame, candidates: list[str]) -> list[dict[str, Any]]:
    """Spread of every candidate response, reported on every run.

    Cheap, and it is what turns "the headline metric saturated" from something
    discovered during the write-up into something visible in `summary.json`.
    """
    rows: list[dict[str, Any]] = []
    for column in candidates:
        clean = frame[column].dropna()
        rows.append(
            {
                "column": column,
                "n": len(clean),
                "n_distinct": int(clean.nunique()),
                "mean": None if clean.empty else round(float(clean.mean()), 6),
                "sd": round(spread(frame[column]), 6),
                "min": None if clean.empty else round(float(clean.min()), 6),
                "max": None if clean.empty else round(float(clean.max()), 6),
            }
        )
    return rows


def require_fittable(
    frame: pd.DataFrame,
    *,
    response: str,
    predictor: str,
    group: str,
    min_response_spread: float,
    min_subjects: int,
    candidates: list[str],
) -> None:
    """Refuse a model the data cannot support, naming what is missing.

    Raises:
        NotFittableError: on a response with no spread, a predictor with no spread,
            or too few groups for a random intercept.
    """
    n_groups = int(frame[group].nunique())
    if n_groups < min_subjects:
        raise NotFittableError(
            f"a random intercept over {group} needs at least {min_subjects} levels, "
            f"this run has {n_groups}"
        )

    if spread(frame[response]) < min_response_spread:
        alternatives = ", ".join(
            f"{row['column']} (sd {row['sd']})"
            for row in response_diagnostics(frame, candidates)
            if row["column"] != response and row["sd"] >= min_response_spread
        )
        raise NotFittableError(
            f"the response {response!r} has a spread of {spread(frame[response]):.3g}, which is "
            f"below {min_response_spread:g}: it is saturated and cannot carry a slope. "
            f"Set stats.response to one that still varies. Candidates: {alternatives or 'none'}"
        )

    if spread(frame[predictor]) < min_response_spread:
        raise NotFittableError(
            f"the predictor {predictor!r} does not vary in this run "
            f"(spread {spread(frame[predictor]):.3g}); this experiment cannot test that model"
        )


# --- fitting ---


@dataclass(frozen=True)
class ModelFit:
    """A fitted mixed model, reduced to what goes into `summary.json`."""

    formula: str
    n_obs: int
    n_groups: int
    converged: bool
    coefficients: list[dict[str, Any]]
    group_var: float | None
    params: dict[str, float]

    def as_json(self) -> dict[str, Any]:
        return {
            "model": self.formula,
            "n_obs": self.n_obs,
            "n_groups": self.n_groups,
            "converged": self.converged,
            "coefficients": self.coefficients,
            "group_var": self.group_var,
        }


def prepare(
    frame: pd.DataFrame, *, response: str, transform: str, n_trials: int
) -> pd.DataFrame:
    """Drop rows the model cannot use and attach the transformed response.

    Rows are dropped only for a missing response, and the count is logged: an
    episode that acquired nothing has no median time to target, and silently
    fitting the rest without saying so would understate the failure rate.
    """
    usable = frame.dropna(subset=[response]).copy()
    dropped = len(frame) - len(usable)
    if dropped:
        logger.info("%d of %d rows have no %s and are dropped", dropped, len(frame), response)
    usable[RESPONSE] = apply_transform(
        usable[response].to_numpy(dtype=np.float64), transform, n_trials=n_trials
    )
    return usable


def fit_mixed(frame: pd.DataFrame, formula: str, *, group: str, ci: float) -> ModelFit:
    """Fit one mixed model with a random intercept per group.

    Raises:
        NotFittableError: if statsmodels cannot fit it at all.
    """
    try:
        result = smf.mixedlm(formula, frame, groups=frame[group]).fit()
    except Exception as error:
        raise NotFittableError(f"could not fit {formula!r}: {error}") from error

    lower, upper = result.conf_int(alpha=1.0 - ci).to_numpy().T
    # `Group Var` is the random-intercept variance, not a fixed effect: it has
    # no standard error or p-value, and leaving it among the coefficients would
    # invite reading a variance component as an estimated slope.
    coefficients = [
        {
            "term": str(term),
            "estimate": _finite(result.params[term]),
            "se": _finite(result.bse[term]),
            "ci95": [_finite(lower[index]), _finite(upper[index])],
            "p": _finite(result.pvalues[term]),
        }
        for index, term in enumerate(result.params.index)
        if str(term) != GROUP_VAR
    ]
    return ModelFit(
        formula=formula,
        n_obs=int(result.nobs),
        n_groups=int(frame[group].nunique()),
        converged=bool(result.converged),
        coefficients=coefficients,
        group_var=_finite(result.params.get(GROUP_VAR, np.nan)),
        params={str(term): float(value) for term, value in result.params.items()},
    )


def resample_subjects(
    frame: pd.DataFrame, rng: np.random.Generator, *, group: str = SUBJECT
) -> pd.DataFrame:
    """One bootstrap replicate: subjects drawn with replacement.

    A subject drawn twice is relabelled, so the model sees two groups rather
    than one group of double size. Without that the random intercept would be
    shared between the two copies and the interval would come out too narrow,
    which is the failure this bootstrap exists to avoid.
    """
    levels = np.sort(frame[group].unique())
    drawn = rng.integers(0, len(levels), size=len(levels))
    parts = []
    for copy, index in enumerate(drawn):
        part = frame[frame[group] == levels[index]].copy()
        part[group] = f"{levels[index]}_r{copy}"
        parts.append(part)
    replicate: pd.DataFrame = pd.concat(parts, ignore_index=True)
    return replicate


# --- H1 ---


def h1_block(frame: pd.DataFrame, cfg: DictConfig) -> dict[str, Any]:
    """Does the kappa-to-performance relationship depend on the mapping?

    The interaction term is the hypothesis; the main effects are not.
    """
    spec = cfg.h1
    formula = str(spec.formula).format(
        response=RESPONSE, predictor=str(spec.predictor), strategy=str(spec.strategy)
    )
    require_fittable(
        frame,
        response=RESPONSE,
        predictor=str(spec.predictor),
        group=str(spec.group),
        min_response_spread=float(cfg.min_response_spread),
        min_subjects=int(cfg.bootstrap.min_subjects),
        candidates=[str(name) for name in cfg.response_candidates],
    )
    fit = fit_mixed(frame, formula, group=str(spec.group), ci=float(cfg.bootstrap.ci))
    interactions = [row for row in fit.coefficients if ":" in row["term"]]
    return {
        "applicable": True,
        **fit.as_json(),
        "interaction_terms": [row["term"] for row in interactions],
        "note": (
            "H1 is the interaction, not the main effects. With nine subjects the power is "
            "limited; read the intervals, not the p-values."
        ),
    }


# --- H2 ---


def _trade_ratios(fit: ModelFit, *, predictor: str, strategy: str, min_slope: float) -> dict[str, Any]:
    """Mapping offset divided by the accuracy slope, per non-reference mapping.

    The units are accuracy: a ratio of 0.08 says switching away from the
    reference mapping bought as much as raising decoder accuracy by 8 points.
    """
    slope = fit.params.get(predictor)
    if slope is None:
        raise NotFittableError(f"the fit has no {predictor!r} term to divide by")

    ratios: dict[str, Any] = {}
    prefix = f"C({strategy})[T."
    for term, value in fit.params.items():
        if not term.startswith(prefix) or ":" in term:
            continue
        name = term[len(prefix) :].rstrip("]")
        ratios[name] = {
            "offset": round(value, 6),
            "slope": round(slope, 6),
            "estimate": round(value / slope, 6) if abs(slope) >= min_slope else None,
            "unstable": abs(slope) < min_slope,
        }
    return ratios


def h2_block(frame: pd.DataFrame, cfg: DictConfig, seed: int) -> dict[str, Any]:
    """The paper's headline: is a mapping change worth more than a better decoder?

    Reported as the trade ratio, the fitted mapping offset over the fitted slope
    along the accuracy dial, with a subject-bootstrapped interval.
    """
    spec = cfg.h2
    predictor, strategy, group = str(spec.predictor), str(spec.strategy), str(spec.group)
    if spread(frame[predictor]) < float(cfg.min_response_spread):
        return {
            "applicable": False,
            "reason": (
                f"{predictor!r} does not vary in this run, so there is no decoder axis to "
                "trade against; H2 is tested by the lambda_sweep experiment"
            ),
        }
    if frame[strategy].nunique() < 2:
        return {
            "applicable": False,
            "reason": f"only one {strategy} in this run, so there is no mapping change to price",
        }

    require_fittable(
        frame,
        response=RESPONSE,
        predictor=predictor,
        group=group,
        min_response_spread=float(cfg.min_response_spread),
        min_subjects=int(cfg.bootstrap.min_subjects),
        candidates=[str(name) for name in cfg.response_candidates],
    )

    reference = str(spec.reference)
    if reference not in set(frame[strategy]):
        raise NotFittableError(
            f"the reference {strategy} {reference!r} is absent from this run; every offset is "
            f"measured against it. Present: {sorted(set(frame[strategy]))}"
        )
    ordered = frame.copy()
    ordered[strategy] = pd.Categorical(
        ordered[strategy],
        categories=[reference, *sorted(set(ordered[strategy]) - {reference})],
    )

    formula = f"{RESPONSE} ~ {predictor} + C({strategy})"
    fit = fit_mixed(ordered, formula, group=group, ci=float(cfg.bootstrap.ci))
    point = _trade_ratios(
        fit, predictor=predictor, strategy=strategy, min_slope=float(spec.min_slope)
    )

    rng = generator_for(seed, "h2_bootstrap")
    draws: dict[str, list[float]] = {name: [] for name in point}
    failures = 0
    for _ in range(int(cfg.bootstrap.n_boot)):
        replicate = resample_subjects(ordered, rng, group=group)
        try:
            replicate_fit = fit_mixed(replicate, formula, group=group, ci=float(cfg.bootstrap.ci))
            values = _trade_ratios(
                replicate_fit,
                predictor=predictor,
                strategy=strategy,
                min_slope=float(spec.min_slope),
            )
        except NotFittableError:
            failures += 1
            continue
        for name, entry in values.items():
            if entry["estimate"] is not None and name in draws:
                draws[name].append(float(entry["estimate"]))

    lower = (1.0 - float(cfg.bootstrap.ci)) / 2.0
    for name, entry in point.items():
        sample = np.asarray(draws[name], dtype=np.float64)
        entry["n_boot_used"] = int(sample.size)
        entry["ci95"] = (
            None
            if sample.size < 2
            else [
                round(float(np.quantile(sample, lower)), 6),
                round(float(np.quantile(sample, 1.0 - lower)), 6),
            ]
        )

    return {
        "applicable": True,
        **fit.as_json(),
        "reference": reference,
        "trade_ratio": point,
        "interpretation": (
            "mapping offset divided by the slope along the accuracy dial, in units of "
            "posterior accuracy; a ratio of 0.08 means the mapping change bought as much "
            "as eight points of decoder accuracy"
        ),
        "bootstrap": {
            "n_boot": int(cfg.bootstrap.n_boot),
            "failed_refits": failures,
            "resampled": "subjects",
        },
    }


# --- H3 ---


def h3_block(frame: pd.DataFrame, cfg: DictConfig) -> dict[str, Any]:
    """Where does decoder quality stop moving the outcome, and at what cost?

    Two numbers, and the second is the guard. Alpha* is the lowest autonomy
    level at which the kappa slope's interval first includes zero. The UCI
    regression says whether the user was still contributing there, because an
    alpha that makes every decoder look equal by ignoring all of them is not the
    finding it resembles.
    """
    spec = cfg.h3
    autonomy, predictor = str(spec.autonomy), str(spec.predictor)
    levels = sorted(value for value in frame[autonomy].dropna().unique())
    if len(levels) < 2:
        return {
            "applicable": False,
            "reason": (
                f"{autonomy!r} does not vary in this run; H3 is tested by the alpha_sweep "
                "experiment"
            ),
        }

    ci = float(cfg.bootstrap.ci)
    per_level: list[dict[str, Any]] = []
    alpha_star: float | None = None
    for level in levels:
        subset = frame[frame[autonomy] == level]
        entry: dict[str, Any] = {"alpha": round(float(level), 6), "n": len(subset)}
        try:
            fit = fit_mixed(
                subset, f"{RESPONSE} ~ {predictor}", group=str(cfg.h1.group), ci=ci
            )
        except NotFittableError as error:
            entry["slope"] = None
            entry["note"] = str(error)
            per_level.append(entry)
            continue

        row = next(item for item in fit.coefficients if item["term"] == predictor)
        if row["ci95"][0] is None or row["ci95"][1] is None:
            # An interval the model could not produce is not an interval that
            # includes zero. Recorded and skipped rather than counted either way.
            entry["slope"] = row["estimate"]
            entry["ci95"] = row["ci95"]
            entry["note"] = "the model produced no interval for this level"
            per_level.append(entry)
            continue
        includes_zero = row["ci95"][0] <= 0.0 <= row["ci95"][1]
        entry.update(
            {"slope": row["estimate"], "ci95": row["ci95"], "includes_zero": includes_zero}
        )
        per_level.append(entry)
        if includes_zero and alpha_star is None:
            alpha_star = float(level)

    guard = str(spec.guard)
    guard_fit = fit_mixed(
        frame.dropna(subset=[guard]).assign(**{RESPONSE: lambda data: data[guard]}),
        f"{RESPONSE} ~ {autonomy}",
        group=str(cfg.h1.group),
        ci=ci,
    )

    return {
        "applicable": True,
        "alpha_star": alpha_star,
        "estimator": "grid",
        "grid": [round(float(level), 6) for level in levels],
        "by_alpha": per_level,
        "guard": {"metric": guard, **guard_fit.as_json()},
        "note": (
            "alpha_star is the lowest level on the alpha grid whose kappa slope interval "
            "includes zero. It is an estimate from that grid, not a solved threshold: the "
            "true crossing lies somewhere between it and the level below, and a finer grid "
            "would move it. Read it alongside the guard: an autonomy level that flattens "
            "the decoder effect by ignoring the user is not shared control working."
        ),
    }


# --- the block ---


def stats_block(
    frame: pd.DataFrame, cfg: DictConfig, *, n_targets: int, seed: int
) -> dict[str, Any]:
    """The `stats` block of `summary.json`.

    Raises:
        NotFittableError: if the primary model cannot be fitted. H2 and H3 are
            reported as inapplicable when the experiment does not vary their
            axis, which is normal; H1 failing is not.
    """
    response, transform = str(cfg.response), str(cfg.transform)
    candidates = [str(name) for name in cfg.response_candidates]
    prepared = prepare(frame, response=response, transform=transform, n_trials=n_targets)

    return {
        "response": response,
        "transform": transform,
        "n_trials": n_targets,
        "response_diagnostics": response_diagnostics(frame, candidates),
        "h1": h1_block(prepared, cfg),
        "h2": h2_block(prepared, cfg, seed),
        "h3": h3_block(prepared, cfg),
        "notes": (
            "Effect sizes with intervals throughout. Nine subjects is thin, and the "
            "write-up says so rather than leaning on p-values."
        ),
    }
