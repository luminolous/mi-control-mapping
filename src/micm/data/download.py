"""Dataset fetching and verification.

Everything goes through MOABB. The evaluation-session labels of BCI IV-2a live
in a separate file from the recordings, and joining them by hand is a known
source of silent label misalignment, so the GDF files are never parsed directly.

Nothing here runs as an import side effect. `scripts/00_download_data.py` is the
entry point, and the user runs it.

The MOABB dataset object is injected rather than imported at module scope, so
this module stays importable, and testable, without paying for the MOABB import
or reaching the network.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import mne
from omegaconf import DictConfig

from micm.data.constants import MOABB_LABEL_TO_CLASS, SESSIONS
from micm.utils.logging import get_logger

logger = get_logger(__name__)

# Only datasets the project actually uses. A plain dict, no dynamic import and
# no getattr on a config string, so an unknown name fails loudly and early.
SUPPORTED_DATASETS: tuple[str, ...] = ("BNCI2014_001",)


class MoabbDataset(Protocol):
    """The slice of the MOABB dataset interface this module relies on."""

    def get_data(
        self, subjects: Sequence[int]
    ) -> Mapping[int, Mapping[str, Mapping[str, mne.io.BaseRaw]]]: ...


@dataclass(frozen=True)
class SubjectReport:
    """Verification outcome for one subject."""

    subject: int
    runs_per_session: dict[str, int]
    trials_per_session: dict[str, int]
    problems: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.problems


@dataclass(frozen=True)
class DownloadReport:
    """Verification outcome for every requested subject."""

    subjects: tuple[SubjectReport, ...]

    @property
    def ok(self) -> bool:
        return all(report.ok for report in self.subjects)

    def as_table(self) -> str:
        """Human-readable summary, one line per subject."""
        header = f"{'subj':>4}  {'runs T/E':>9}  {'trials T/E':>11}  status"
        lines = [header, "-" * len(header)]
        for report in self.subjects:
            runs = "/".join(str(report.runs_per_session.get(s, 0)) for s in SESSIONS)
            trials = "/".join(str(report.trials_per_session.get(s, 0)) for s in SESSIONS)
            status = "ok" if report.ok else "; ".join(report.problems)
            lines.append(f"{report.subject:>4}  {runs:>9}  {trials:>11}  {status}")
        return "\n".join(lines)


def build_dataset(cfg: DictConfig) -> MoabbDataset:
    """Instantiate the MOABB dataset named in the config.

    Imported here rather than at module scope: MOABB pulls in a large dependency
    tree, and the rest of this module does not need it.

    Raises:
        KeyError: if the configured dataset is not one this project supports.
    """
    name = str(cfg.data.moabb_dataset)
    if name not in SUPPORTED_DATASETS:
        raise KeyError(f"unsupported dataset {name!r}, expected one of {list(SUPPORTED_DATASETS)}")

    from moabb.datasets import BNCI2014_001

    # MOABB defaults to "ignore", which drops the source artifact flags without
    # saying so. The mode is in config because it changes which trials the
    # decoder is fitted on.
    dataset: MoabbDataset = BNCI2014_001(
        artifact_handling=str(cfg.data.artifacts.moabb_handling)
    )
    return dataset


def set_download_dir(path: Path) -> None:
    """Point MOABB at the project's own raw data directory.

    Without this MOABB writes into the user's global MNE data directory, which
    makes it unclear which copy of the data a result came from.
    """
    from moabb.utils import set_download_dir as moabb_set_download_dir

    path.mkdir(parents=True, exist_ok=True)
    moabb_set_download_dir(str(path))


def normalize_sessions(
    sessions: Mapping[str, Mapping[str, mne.io.BaseRaw]], cfg: DictConfig
) -> dict[str, list[mne.io.BaseRaw]]:
    """Translate MOABB's session and run keys into this project's naming.

    Runs are ordered by their MOABB key, which is what puts them back in
    recording order. The session key mapping is verified against what the
    dataset actually returned, because those keys changed between MOABB major
    versions and a wrong guess would silently swap train and test.

    Raises:
        ValueError: if the returned session keys are not exactly the configured
            ones.
    """
    mapping = {str(k): str(v) for k, v in cfg.data.session_keys.items()}
    returned = set(sessions)
    if returned != set(mapping):
        raise ValueError(
            f"dataset returned sessions {sorted(returned)} but the config maps "
            f"{sorted(mapping)}; update data.session_keys rather than assuming an order"
        )

    out: dict[str, list[mne.io.BaseRaw]] = {}
    for moabb_key, session in mapping.items():
        runs = sessions[moabb_key]
        out[session] = [runs[key] for key in sorted(runs)]
    return out


def load_subject_raws(
    dataset: MoabbDataset, cfg: DictConfig, subject: int
) -> dict[str, list[mne.io.BaseRaw]]:
    """Return `{session: [run raws in order]}` for one subject.

    Raises:
        KeyError: if the dataset returns nothing for `subject`.
        ValueError: if the session keys do not match the config.
    """
    fetched = dataset.get_data([subject])
    if subject not in fetched:
        raise KeyError(f"dataset returned no data for subject {subject}")
    return normalize_sessions(fetched[subject], cfg)


def _count_trials(raw: mne.io.BaseRaw) -> int:
    return sum(
        1 for description in raw.annotations.description if str(description) in MOABB_LABEL_TO_CLASS
    )


def verify_subject(
    dataset: MoabbDataset, cfg: DictConfig, subject: int
) -> SubjectReport:
    """Fetch one subject and check its run and trial counts against the config.

    Fetching is what triggers the download, so this doubles as the download step.
    MOABB skips files it already has, which is what makes the script idempotent.
    """
    raws = load_subject_raws(dataset, cfg, subject)

    runs_per_session = {session: len(raws[session]) for session in SESSIONS}
    trials_per_session = {
        session: sum(_count_trials(raw) for raw in raws[session]) for session in SESSIONS
    }

    problems: list[str] = []
    expected_runs = int(cfg.data.expected.runs_per_session)
    expected_trials = int(cfg.data.expected.trials_per_session)
    for session in SESSIONS:
        if runs_per_session[session] != expected_runs:
            problems.append(
                f"session {session}: {runs_per_session[session]} runs, expected {expected_runs}"
            )
        if trials_per_session[session] != expected_trials:
            problems.append(
                f"session {session}: {trials_per_session[session]} trials, "
                f"expected {expected_trials}"
            )

    return SubjectReport(
        subject=subject,
        runs_per_session=runs_per_session,
        trials_per_session=trials_per_session,
        problems=tuple(problems),
    )


def download_and_verify(
    cfg: DictConfig,
    *,
    subjects: Sequence[int] | None = None,
    dataset: MoabbDataset | None = None,
) -> DownloadReport:
    """Download every requested subject and verify what arrived.

    Idempotent: MOABB serves already-downloaded files from its cache, so a second
    call verifies without refetching.

    Args:
        cfg: composed config carrying `data` and `paths`.
        subjects: defaults to `cfg.data.subjects`.
        dataset: injected for testing; built from the config when omitted.
    """
    if dataset is None:
        set_download_dir(Path(cfg.paths.data_raw))
        dataset = build_dataset(cfg)

    wanted = list(subjects if subjects is not None else cfg.data.subjects)
    reports: list[SubjectReport] = []
    for subject in wanted:
        logger.info("fetching subject %d", subject)
        report = verify_subject(dataset, cfg, subject)
        if not report.ok:
            logger.warning("subject %d failed verification: %s", subject, "; ".join(report.problems))
        reports.append(report)

    return DownloadReport(subjects=tuple(reports))
