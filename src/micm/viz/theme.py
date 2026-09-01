"""One palette, one set of type sizes, one way of saving a figure.

Every figure in the paper is read next to the others, so a mapping that is blue
in one panel and orange in another costs the reader more than any amount of
polish buys back. The palette lives here and nothing else defines a colour.

Colours are from Okabe and Ito's colourblind-safe set, and the four mappings are
also given distinct line styles and markers, so the figures survive being read in
greyscale or printed badly. Colour alone is never the only channel carrying a
distinction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import matplotlib as mpl

# A non-interactive backend, chosen before pyplot is imported. These figures are
# only ever saved, never shown, and a default backend that wants a display makes
# the figure step fail on a headless machine for no reason.
mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from micm.utils.logging import get_logger

logger = get_logger(__name__)

# Okabe-Ito, which stays distinguishable under the common colour deficiencies.
BLUE: Final[str] = "#0072B2"
ORANGE: Final[str] = "#E69F00"
GREEN: Final[str] = "#009E73"
VERMILLION: Final[str] = "#D55E00"
PURPLE: Final[str] = "#CC79A7"
SKY: Final[str] = "#56B4E9"
YELLOW: Final[str] = "#F0E442"
BLACK: Final[str] = "#000000"
GREY: Final[str] = "#7F7F7F"


@dataclass(frozen=True)
class Style:
    """How one mapping is drawn, in every figure it appears in."""

    label: str
    color: str
    linestyle: str
    marker: str


# Keyed by the `mapping` column, which is the config name rather than the class
# name: S2 with and without entropy scaling are different conditions.
MAPPING_STYLES: Final[dict[str, Style]] = {
    "s1_argmax": Style("S1 argmax", BLUE, "-", "o"),
    "s2_weighted": Style("S2 weighted", ORANGE, "--", "s"),
    "s2_weighted_entropy": Style("S2 weighted + entropy", VERMILLION, "-.", "D"),
    "s3_evidence": Style("S3 evidence", GREEN, ":", "^"),
    "s4_shared": Style("S4 shared", PURPLE, "-", "v"),
}

# Anything not in the table, so an unexpected condition is visibly grey rather
# than silently sharing a colour with a mapping it is not.
UNKNOWN_STYLE: Final[Style] = Style("unknown", GREY, "-", "x")

# For axes that are not the mapping: subjects, decoders, latencies. Ordered so
# the first few are the most distinguishable.
SEQUENCE: Final[tuple[str, ...]] = (BLUE, ORANGE, GREEN, VERMILLION, PURPLE, SKY, YELLOW, BLACK)

DECODER_LABELS: Final[dict[str, str]] = {
    "fbcsp": "FBCSP",
    "riemann": "Riemannian",
    "eegnet": "EEGNet",
    "synthetic": "synthetic",
}

METRIC_LABELS: Final[dict[str, str]] = {
    "success_rate": "success rate",
    "effective_acc": "posterior accuracy",
    "kappa_offline": "offline $\\kappa$",
    "path_efficiency": "path efficiency",
    "time_to_target_median": "median time to target (s)",
    "direction_reversals": "direction reversals per target",
    "effective_itr": "effective ITR (bits/min)",
    "user_contribution_index": "user contribution index",
    "bursts_without_command": "bursts with no command",
    "command_accuracy": "command accuracy",
    "latency_ms": "feedback latency (ms)",
    "window_s": "decoding window (s)",
    "alpha": "autonomy level $\\alpha$",
    "quality_level": "quality level",
}

RIBBON_ALPHA: Final[float] = 0.18
REFERENCE_LINE: Final[dict[str, Any]] = {"color": GREY, "linewidth": 0.8, "linestyle": (0, (3, 3))}


def style_for(mapping: str) -> Style:
    """How to draw one mapping. Unknown names come back grey, never recoloured."""
    return MAPPING_STYLES.get(mapping, UNKNOWN_STYLE)


def label_for(column: str) -> str:
    """Axis label for a column, falling back to the column name itself.

    The fallback is the raw name on purpose: an unlabelled axis reading
    `uci_excluded_frac` tells the reader which column to go and look at, where a
    guessed prose label would not.
    """
    return METRIC_LABELS.get(column, column)


def apply_theme() -> None:
    """Set the rcParams every figure is drawn under.

    Called once by the figure module rather than by each function, so a figure
    cannot be drawn under whatever the importing process happened to leave set.
    """
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#E6E6E6",
            "grid.linewidth": 0.6,
            "lines.linewidth": 1.6,
            "lines.markersize": 4.5,
            "legend.frameon": False,
            # Type 42 keeps the text as text in the PDF, so the publisher can
            # search and re-flow it rather than receiving outlines.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(figure: Figure, directory: Path, name: str) -> list[Path]:
    """Write one figure as both PDF and PNG, and return the paths.

    Vector for the paper, raster for looking at quickly. Both, always: a figure
    that exists in only one format is the one somebody needs in the other at
    four in the morning.
    """
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in ("pdf", "png"):
        path = directory / f"{name}.{suffix}"
        figure.savefig(path)
        written.append(path)
    plt.close(figure)
    logger.info("wrote %s.{pdf,png}", directory / name)
    return written
