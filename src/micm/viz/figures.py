"""The six figures of `agents/05` §6, one function each.

Every function takes runs that have already been read and returns a matplotlib
figure. None of them simulates anything and none of them computes a metric: the
numbers come from `summary.json` where the summary has them and from
`episodes.parquet` where it does not, and if a number is in neither then the
pipeline is missing a column and that is the finding, not something for a figure
to paper over.

The one thing figures do compute is a straight line through a scatter, which is
a drawing aid rather than a result. The fitted model that carries H1 lives in
`summary.json`, and the figures that show a line say so.

**Nothing raises for want of data.** A figure asked for an axis this run does not
vary draws the panel and writes the reason across it. The smoke run varies almost
nothing and must still produce all six files, because a figure that only renders
once the real matrix exists is a figure nobody has run. What is refused is a
missing *column*, which means the frame is not what the contract says.
"""

from __future__ import annotations

from typing import Any, Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from micm.eval.writer import LoadedRun
from micm.utils.logging import get_logger
from micm.viz.theme import (
    REFERENCE_LINE,
    RIBBON_ALPHA,
    SEQUENCE,
    apply_theme,
    label_for,
    style_for,
)

logger = get_logger(__name__)

# Axes a `by_cell` entry may carry. Mirrors the candidate list in
# `eval/aggregate.py`; anything else in an entry is a metric block.
_AXIS_KEYS: Final[frozenset[str]] = frozenset(
    {
        "mapping",
        "decoder",
        "quality_level",
        "alpha",
        "intent_mode",
        "protocol",
        "window_s",
        "latency_ms",
        "error_struct",
        "subject",
        "n_episodes",
    }
)

MISSING_STYLE: Final[dict[str, Any]] = {
    "ha": "center",
    "va": "center",
    "fontsize": 8,
    "color": "#7F7F7F",
    "wrap": True,
}


def cell_table(run: LoadedRun, block: str = "by_cell") -> pd.DataFrame:
    """Flatten a `scores` block into a frame of axes and `mean/lo/hi` columns.

    Reading, not recomputing: the means and the intervals were produced by
    `eval/aggregate.py` when the run was written or last analysed, and a figure
    that recomputed them could disagree with the summary it sits beside.

    Raises:
        KeyError: if the run's summary has no such block, which means the run
            predates the scores contract rather than that it has no data.
    """
    entries = run.summary.get("scores", {}).get(block)
    if entries is None:
        raise KeyError(
            f"{run.directory} has no scores.{block}; re-run scripts/04_analyze.py on it"
        )

    rows = []
    for entry in entries:
        row: dict[str, Any] = {}
        for key, value in entry.items():
            if key in _AXIS_KEYS:
                row[key] = value
            elif isinstance(value, dict):
                row[f"{key}_mean"] = value.get("mean")
                interval = value.get("ci95")
                row[f"{key}_lo"] = None if interval is None else interval[0]
                row[f"{key}_hi"] = None if interval is None else interval[1]
            elif value is None:
                # A metric no episode in this cell recorded.
                row[f"{key}_mean"] = None
        rows.append(row)
    return pd.DataFrame(rows)


def note(axes: Axes, reason: str) -> None:
    """Write across an empty panel why it is empty.

    A blank panel and a panel whose data was all zero look the same on paper.
    This is the difference between them.
    """
    axes.text(0.5, 0.5, reason, transform=axes.transAxes, **MISSING_STYLE)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.grid(False)
    # Spines too: an empty pair of axes reads as a plot whose data fell outside
    # the limits, which is a different problem from the one that happened.
    for spine in axes.spines.values():
        spine.set_visible(False)


def _has_spread(values: pd.Series) -> bool:
    return values.dropna().nunique() > 1


def _ribbon(axes: Axes, frame: pd.DataFrame, x: str, metric: str, color: str) -> None:
    """Shade the interval where the summary has one, and skip where it does not."""
    low, high = frame.get(f"{metric}_lo"), frame.get(f"{metric}_hi")
    if low is None or high is None or low.isna().all():
        return
    axes.fill_between(frame[x], low, high, color=color, alpha=RIBBON_ALPHA, linewidth=0)


def _fit_line(axes: Axes, x: np.ndarray, y: np.ndarray, color: str) -> None:
    """A least-squares line through a scatter, as a drawing aid.

    Not the model. `summary.json` carries the mixed model that H1 is read from,
    which pools subjects with a random intercept; this line does not, and the
    two can differ. The caption says so.
    """
    usable = np.isfinite(x) & np.isfinite(y)
    if usable.sum() < 3 or np.ptp(x[usable]) == 0.0:
        return
    slope, intercept = np.polyfit(x[usable], y[usable], 1)
    span = np.linspace(x[usable].min(), x[usable].max(), 2)
    axes.plot(span, slope * span + intercept, color=color, linewidth=1.2, alpha=0.8)


# --- 1. the main figure ---


def figure_quality_sweep(run: LoadedRun) -> Figure:
    """Success rate against achieved posterior accuracy, one line per mapping.

    The figure the paper is built on: if the mapping lines are separated by more
    than the accuracy axis moves them, a mapping change is worth more than a
    better decoder, which is H2 stated as a picture. The number and its interval
    are in `summary.json`; this shows the shape the number summarises.
    """
    apply_theme()
    figure, axes = plt.subplots(figsize=(5.2, 3.6))
    table = cell_table(run)

    if "effective_acc_mean" not in table or "success_rate_mean" not in table:
        note(axes, "this run's summary carries no accuracy or success rate")
        return figure

    for mapping, group in table.groupby("mapping", dropna=False):
        style = style_for(str(mapping))
        ordered = group.sort_values("effective_acc_mean")
        axes.plot(
            ordered["effective_acc_mean"],
            ordered["success_rate_mean"],
            color=style.color,
            linestyle=style.linestyle,
            marker=style.marker,
            label=style.label,
        )
        _ribbon(axes, ordered, "effective_acc_mean", "success_rate", style.color)

    if not _has_spread(table["effective_acc_mean"]):
        axes.set_title("one accuracy level: no sweep in this run", fontsize=8, color="#7F7F7F")

    axes.set_xlabel(label_for("effective_acc"))
    axes.set_ylabel(label_for("success_rate"))
    axes.legend(loc="best")
    figure.suptitle("Closed-loop success against decoder quality", y=1.0)
    return figure


# --- 2. offline against closed loop ---


def figure_kappa_against_success(run: LoadedRun) -> Figure:
    """One panel per mapping: offline kappa against closed-loop success.

    H1 asks whether the relationship between the two depends on the mapping, so
    the panels are the hypothesis. Each point is one subject and decoder, which
    is the level at which kappa is defined; the line through each panel is a
    plain least-squares fit and not the mixed model in `summary.json`.
    """
    apply_theme()
    frame = run.episodes
    mappings = sorted(frame["mapping"].dropna().unique())
    figure, panels = plt.subplots(
        1, max(len(mappings), 1), figsize=(2.4 * max(len(mappings), 1), 2.9), squeeze=False,
        sharey=True,
    )

    if frame["kappa_offline"].dropna().empty:
        note(panels[0][0], "no offline kappa in this run\n(synthetic posteriors carry none)")
        for axes in panels[0][1:]:
            note(axes, "")
        figure.suptitle("Offline decoder quality against closed-loop success", y=1.02)
        return figure

    # One point per subject and decoder: kappa is a property of that pair, and
    # plotting episodes would draw ten identical x values per point.
    grouped = (
        frame.groupby(["mapping", "subject", "decoder"], dropna=False, observed=True)[
            ["kappa_offline", "success_rate"]
        ]
        .mean()
        .reset_index()
    )

    for axes, mapping in zip(panels[0], mappings, strict=False):
        style = style_for(str(mapping))
        subset = grouped[grouped["mapping"] == mapping]
        for index, decoder in enumerate(sorted(subset["decoder"].unique())):
            points = subset[subset["decoder"] == decoder]
            axes.scatter(
                points["kappa_offline"],
                points["success_rate"],
                color=SEQUENCE[index % len(SEQUENCE)],
                s=18,
                label=str(decoder),
                edgecolor="white",
                linewidth=0.4,
            )
        _fit_line(
            axes,
            subset["kappa_offline"].to_numpy(dtype=float),
            subset["success_rate"].to_numpy(dtype=float),
            style.color,
        )
        axes.set_title(style.label)
        axes.set_xlabel(label_for("kappa_offline"))

    panels[0][0].set_ylabel(label_for("success_rate"))
    panels[0][-1].legend(loc="best", title="decoder")
    figure.suptitle(
        "Offline decoder quality against closed-loop success (lines are least squares, "
        "not the fitted model)",
        y=1.04,
        fontsize=9,
    )
    return figure


# --- 3. the autonomy sweep ---


def figure_alpha_sweep(run: LoadedRun) -> Figure:
    """Success rate and user contribution against autonomy, one pair per decoder.

    Three decoders rather than one averaged line, because H3 is the question of
    whether they converge: if the strong and the weak decoder end up on the same
    success rate as alpha rises, decoder quality has stopped mattering, and that
    is visible as three lines meeting. Averaging them would erase the hypothesis
    into a single curve.

    The dashed lines are the guard. Success rising with alpha is only a result
    while the user is still contributing; if UCI falls to zero at the same alpha,
    the robot is driving itself and the decoders were made irrelevant rather than
    assisted. Alpha* comes from `summary.json` and is marked as the grid estimate
    it is.
    """
    apply_theme()
    figure, axes = plt.subplots(figsize=(5.8, 3.8))
    table = cell_table(run)

    if "alpha" not in table or table["alpha"].dropna().empty:
        note(axes, "this run does not vary alpha")
        figure.suptitle("Autonomy sweep", y=1.0)
        return figure

    guard = axes.twinx()
    guard.grid(False)

    # One line per decoder where the run varies it. `by_cell` groups by every
    # varying axis, so a single line here would join three decoders at each
    # alpha and read as a sawtooth rather than three curves.
    groups = (
        table.groupby("decoder", dropna=False)
        if "decoder" in table
        else [("all decoders", table)]
    )
    for index, (decoder, group) in enumerate(groups):
        color = SEQUENCE[index % len(SEQUENCE)]
        ordered = group.sort_values("alpha")
        axes.plot(
            ordered["alpha"],
            ordered["success_rate_mean"],
            color=color,
            marker="o",
            label=str(decoder),
        )
        _ribbon(axes, ordered, "alpha", "success_rate", color)
        if "user_contribution_index_mean" in ordered:
            guard.plot(
                ordered["alpha"],
                ordered["user_contribution_index_mean"],
                color=color,
                linestyle="--",
                marker="s",
                markersize=3,
                alpha=0.7,
            )

    axes.set_xlabel(label_for("alpha"))
    axes.set_ylabel(f"{label_for('success_rate')} (solid)")
    guard.set_ylabel(f"{label_for('user_contribution_index')} (dashed)")

    alpha_star = run.summary.get("stats", {}).get("h3", {}).get("alpha_star")
    if alpha_star is not None:
        axes.axvline(float(alpha_star), **REFERENCE_LINE)
        axes.annotate(
            # Raw: in a plain f-string the backslash-a is a bell character, and
            # mathtext then refuses the whole label.
            rf"$\alpha^*$ = {float(alpha_star):.3g} (grid estimate)",
            xy=(float(alpha_star), axes.get_ylim()[0]),
            xytext=(4, 6),
            textcoords="offset points",
            fontsize=8,
            color="#7F7F7F",
        )

    axes.legend(loc="best", title="decoder")
    figure.suptitle("Autonomy against performance, with the user contribution guard", y=1.0)
    return figure


# --- 4. what waiting for evidence costs ---


def figure_command_tradeoff(run: LoadedRun) -> Figure:
    """What a mapping pays for the commands it declines to issue.

    **Not the figure `agents/05` §6 asks for.** That one is an S3 Pareto front of
    latency against command accuracy, and neither quantity is in the episode
    schema: there is no `command_accuracy` column and no S3 threshold sweep in
    the experiment matrix. Rather than invent either, this plots the two columns
    that do exist and carry the same trade: `bursts_without_command`, which
    counts the bursts a mapping let pass without committing, against the success
    rate it bought by waiting. S3 is the mapping that can move along this axis;
    the others are shown for scale. See docs/decisions.md D55.
    """
    apply_theme()
    figure, axes = plt.subplots(figsize=(5.2, 3.6))
    frame = run.episodes

    grouped = (
        frame.groupby(["mapping", "latency_ms"], dropna=False, observed=True)[
            ["bursts_without_command", "success_rate"]
        ]
        .mean()
        .reset_index()
    )

    for mapping, group in grouped.groupby("mapping", dropna=False):
        style = style_for(str(mapping))
        ordered = group.sort_values("latency_ms")
        axes.plot(
            ordered["bursts_without_command"],
            ordered["success_rate"],
            color=style.color,
            linestyle=style.linestyle,
            marker=style.marker,
            label=style.label,
        )
        for _, row in ordered.iterrows():
            axes.annotate(
                f"{int(row['latency_ms'])} ms",
                xy=(row["bursts_without_command"], row["success_rate"]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
                color="#7F7F7F",
            )

    axes.set_xlabel(label_for("bursts_without_command"))
    axes.set_ylabel(label_for("success_rate"))
    axes.legend(loc="best")
    figure.suptitle("What declining to command costs, by feedback latency", y=1.0)
    return figure


# --- 5. the ablations ---


ABLATION_AXES: Final[tuple[tuple[str, str], ...]] = (
    ("ablation_window", "window_s"),
    ("ablation_latency", "latency_ms"),
    ("ablation_protocol", "protocol"),
    ("ablation_errstruct", "error_struct"),
)


def figure_ablation_grid(runs: dict[str, LoadedRun], metric: str = "success_rate") -> Figure:
    """Small multiples: one panel per ablation, all four on one page.

    Four separate runs, so this is the one figure that reads more than one run
    directory. A panel whose run was not supplied says which run is missing
    rather than being dropped, so the figure keeps its shape and the gap is
    visible instead of being mistaken for an ablation that was never planned.
    """
    apply_theme()
    figure, panels = plt.subplots(2, 2, figsize=(7.4, 5.4))
    flat = panels.flatten()

    for axes, (experiment, axis) in zip(flat, ABLATION_AXES, strict=True):
        run = runs.get(experiment)
        if run is None:
            note(axes, f"{experiment}\nnot supplied")
            continue

        table = cell_table(run)
        column = f"{metric}_mean"
        if axis not in table or column not in table:
            note(axes, f"{experiment}\nhas no {axis} axis in its summary")
            continue

        for mapping, group in table.groupby("mapping", dropna=False):
            style = style_for(str(mapping))
            ordered = group.sort_values(axis)
            positions = (
                ordered[axis]
                if pd.api.types.is_numeric_dtype(ordered[axis])
                else np.arange(len(ordered))
            )
            axes.plot(
                positions,
                ordered[column],
                color=style.color,
                linestyle=style.linestyle,
                marker=style.marker,
                label=style.label,
            )
            if not pd.api.types.is_numeric_dtype(ordered[axis]):
                axes.set_xticks(np.arange(len(ordered)))
                axes.set_xticklabels([str(value) for value in ordered[axis]])

        axes.set_xlabel(label_for(axis))
        axes.set_ylabel(label_for(metric))
        axes.set_title(experiment.replace("ablation_", ""))

    handles, labels = flat[0].get_legend_handles_labels()
    for axes in flat:
        if not handles:
            handles, labels = axes.get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="lower center", ncol=len(labels))
        figure.subplots_adjust(bottom=0.16)
    figure.suptitle("Ablations", y=1.0)
    return figure


# --- 6. the honesty check on S4 ---


def figure_intent_ablation(run: LoadedRun) -> Figure:
    """Intent-aware against intent-blind S4, across autonomy.

    Intent-aware reads the decoder twice, once through the user term and again
    through its choice of attractive target, so part of any advantage it shows is
    the decoder informing the autonomy rather than arbitration doing work.
    Intent-blind attracts toward the active task target whatever the posterior
    says. The gap between the two lines is that leakage, drawn.
    """
    apply_theme()
    figure, axes = plt.subplots(figsize=(5.2, 3.6))
    table = cell_table(run)

    if "intent_mode" not in table or table["intent_mode"].dropna().nunique() < 2:
        note(axes, "this run does not compare intent-aware with intent-blind")
        figure.suptitle("Intent-aware against intent-blind S4", y=1.0)
        return figure

    for index, (mode, group) in enumerate(table.groupby("intent_mode", dropna=False)):
        ordered = group.sort_values("alpha") if "alpha" in group else group
        color = SEQUENCE[index % len(SEQUENCE)]
        axes.plot(
            ordered["alpha"] if "alpha" in ordered else np.arange(len(ordered)),
            ordered["success_rate_mean"],
            color=color,
            marker="o" if mode == "aware" else "s",
            linestyle="-" if mode == "aware" else "--",
            label=f"intent-{mode}",
        )
        _ribbon(axes, ordered, "alpha", "success_rate", color)

    axes.set_xlabel(label_for("alpha"))
    axes.set_ylabel(label_for("success_rate"))
    axes.legend(loc="best")
    figure.suptitle("The gap between the two lines is the decoder leaking into the autonomy", y=1.0)
    return figure
