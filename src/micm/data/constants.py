"""Fixed identifiers for BCI Competition IV-2a.

The class ordering is defined here and nowhere else. Every downstream array
indexed by class, including the posterior columns and the class-to-direction
table, inherits this order. Duplicating it somewhere else is how a silent label
permutation gets introduced.
"""

from __future__ import annotations

from typing import Final

# GDF event codes for the four cued motor-imagery classes, in class-index order.
# 769 left hand, 770 right hand, 771 feet, 772 tongue.
EVENT_CODE_TO_CLASS: Final[dict[int, int]] = {769: 0, 770: 1, 771: 2, 772: 3}

CLASS_NAMES: Final[tuple[str, ...]] = ("left_hand", "right_hand", "feet", "tongue")

# MOABB reports the same four classes under these names. Used to translate the
# annotation descriptions returned by the loader into class indices.
MOABB_LABEL_TO_CLASS: Final[dict[str, int]] = {
    "left_hand": 0,
    "right_hand": 1,
    "feet": 2,
    "tongue": 3,
}

N_CLASSES: Final[int] = len(CLASS_NAMES)

SESSION_TRAIN: Final[str] = "T"
SESSION_TEST: Final[str] = "E"
SESSIONS: Final[tuple[str, str]] = (SESSION_TRAIN, SESSION_TEST)

# Columns of the trial table produced by epoching, in order.
TRIAL_META_COLUMNS: Final[tuple[str, ...]] = (
    "trial_id",
    "subject",
    "session",
    "run",
    "class",
    "onset_s",
    "artifact",
)


def class_name(index: int) -> str:
    """Human-readable name of a class index.

    Raises:
        IndexError: if `index` is not a valid class index.
    """
    if not 0 <= index < N_CLASSES:
        raise IndexError(f"class index must be in 0..{N_CLASSES - 1}, got {index}")
    return CLASS_NAMES[index]
