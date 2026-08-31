"""Download and verification tests.

MOABB is never imported and the network is never touched. The dataset object is
injected, which is the reason `download.py` takes one instead of building its
own. What is under test is the session-key translation and the count
verification, and both are exactly where a silent mistake would be expensive:
a swapped session mapping would train on the evaluation half without raising.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import mne
import numpy as np
import pytest
from omegaconf import DictConfig

from micm.data.download import (
    DownloadReport,
    build_dataset,
    download_and_verify,
    load_subject_raws,
    normalize_sessions,
    verify_subject,
)
from micm.utils import load_config

RUNS_PER_SESSION = 6
TRIALS_PER_RUN = 48
LABELS = ("left_hand", "right_hand", "feet", "tongue")


@pytest.fixture
def cfg() -> DictConfig:
    return load_config("download")


def _raw(n_trials: int, *, marker: float = 0.0) -> mne.io.RawArray:
    """One-second recording carrying `n_trials` cue annotations.

    The verification path reads annotations only, so the signal can be a stub.
    `marker` makes each run distinguishable, which is how run ordering is checked.
    """
    info = mne.create_info(ch_names=["EEG-00"], sfreq=250.0, ch_types=["eeg"])
    raw = mne.io.RawArray(np.full((1, 250), marker), info, verbose=False)
    raw.set_annotations(
        mne.Annotations(
            onset=[i * 0.001 for i in range(n_trials)],
            duration=[0.0] * n_trials,
            description=[LABELS[i % len(LABELS)] for i in range(n_trials)],
        ),
        verbose=False,
    )
    return raw


class FakeDataset:
    """Stands in for a MOABB dataset. Records which subjects were asked for."""

    def __init__(
        self,
        *,
        session_keys: tuple[str, str] = ("0train", "1test"),
        n_runs: int = RUNS_PER_SESSION,
        n_trials: int = TRIALS_PER_RUN,
        subjects: tuple[int, ...] = (1, 2),
    ) -> None:
        self.session_keys = session_keys
        self.n_runs = n_runs
        self.n_trials = n_trials
        self.subjects = subjects
        self.requested: list[int] = []

    def get_data(
        self, subjects: Sequence[int]
    ) -> Mapping[int, Mapping[str, Mapping[str, mne.io.RawArray]]]:
        self.requested.extend(subjects)
        return {
            subject: {
                session: {
                    str(run): _raw(self.n_trials, marker=float(run))
                    for run in range(self.n_runs)
                }
                for session in self.session_keys
            }
            for subject in subjects
            if subject in self.subjects
        }


def test_session_keys_are_translated(cfg: DictConfig) -> None:
    raws = load_subject_raws(FakeDataset(), cfg, subject=1)
    assert set(raws) == {"T", "E"}
    assert len(raws["T"]) == RUNS_PER_SESSION


def test_runs_are_ordered_by_key(cfg: DictConfig) -> None:
    """Recording order carries slow drift, so runs must not come back shuffled."""
    raws = load_subject_raws(FakeDataset(), cfg, subject=1)
    markers = [float(raw.get_data()[0, 0]) for raw in raws["T"]]
    assert markers == sorted(markers)
    assert markers == [float(i) for i in range(RUNS_PER_SESSION)]


def test_unexpected_session_keys_raise(cfg: DictConfig) -> None:
    """A MOABB version that renames its sessions must stop the pipeline, not be guessed at."""
    dataset = FakeDataset(session_keys=("session_T", "session_E"))
    with pytest.raises(ValueError, match=r"update data.session_keys"):
        load_subject_raws(dataset, cfg, subject=1)


def test_partial_session_keys_raise(cfg: DictConfig) -> None:
    sessions: dict[str, dict[str, mne.io.RawArray]] = {"0train": {"0": _raw(4)}}
    with pytest.raises(ValueError, match=r"update data.session_keys"):
        normalize_sessions(sessions, cfg)


def test_missing_subject_raises(cfg: DictConfig) -> None:
    with pytest.raises(KeyError, match="no data for subject"):
        load_subject_raws(FakeDataset(subjects=(1,)), cfg, subject=9)


def test_verify_subject_accepts_the_expected_counts(cfg: DictConfig) -> None:
    report = verify_subject(FakeDataset(), cfg, subject=1)
    assert report.ok
    assert report.runs_per_session == {"T": RUNS_PER_SESSION, "E": RUNS_PER_SESSION}
    assert report.trials_per_session == {"T": 288, "E": 288}


def test_verify_subject_reports_a_short_session(cfg: DictConfig) -> None:
    report = verify_subject(FakeDataset(n_runs=5), cfg, subject=1)
    assert not report.ok
    assert any("5 runs, expected 6" in problem for problem in report.problems)


def test_verify_subject_reports_missing_trials(cfg: DictConfig) -> None:
    report = verify_subject(FakeDataset(n_trials=47), cfg, subject=1)
    assert not report.ok
    assert any("282 trials, expected 288" in problem for problem in report.problems)


def test_download_and_verify_covers_every_requested_subject(cfg: DictConfig) -> None:
    dataset = FakeDataset(subjects=(1, 2, 3))
    report = download_and_verify(cfg, subjects=[1, 2, 3], dataset=dataset)
    assert report.ok
    assert dataset.requested == [1, 2, 3]
    assert [r.subject for r in report.subjects] == [1, 2, 3]


def test_download_and_verify_defaults_to_the_configured_subjects(cfg: DictConfig) -> None:
    dataset = FakeDataset(subjects=tuple(cfg.data.subjects))
    report = download_and_verify(cfg, dataset=dataset)
    assert [r.subject for r in report.subjects] == list(cfg.data.subjects)


def test_report_table_lists_every_subject(cfg: DictConfig) -> None:
    report = download_and_verify(cfg, subjects=[1, 2], dataset=FakeDataset(n_runs=5))
    table = report.as_table()
    assert not report.ok
    assert table.count("\n") == 3  # header, rule, two subjects
    assert "expected 6" in table


def test_empty_report_is_vacuously_ok() -> None:
    assert DownloadReport(subjects=()).ok


def test_build_dataset_rejects_an_unknown_name(cfg: DictConfig) -> None:
    """No getattr on a config string: an unsupported dataset fails before any download."""
    from omegaconf import OmegaConf

    other = OmegaConf.merge(cfg, {"data": {"moabb_dataset": "NotADataset"}})
    assert isinstance(other, DictConfig)
    with pytest.raises(KeyError, match="unsupported dataset"):
        build_dataset(other)
