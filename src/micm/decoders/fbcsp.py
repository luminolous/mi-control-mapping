"""Filter-bank CSP with an LDA classifier.

The weak end of the decoder range, and the one the literature treats as the
standard baseline. Expected Cohen's kappa on BCI IV-2a is roughly 0.40 to 0.55.
A result far outside that range means preprocessing is wrong; tuning the
classifier to compensate would hide the real problem.

Pipeline: split each trial into 4 Hz bands, learn CSP spatial filters per band,
take log-variance features, keep the most informative ones by mutual
information, and classify with shrinkage LDA. LDA gives calibrated posteriors
directly, so no separate calibration step is needed.

The band filters are zero-phase. That is not a contradiction of the causal
preprocessing in `data/epoching.py`: those filters run over the continuous
recording and would reach past the end of the current window, whereas these run
inside a trial window that is already complete at decision time. See
docs/decisions.md D14.
"""

from __future__ import annotations

import numpy as np
from mne.decoding import CSP
from scipy.signal import butter, sosfiltfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import mutual_info_classif

from micm.decoders.base import BaseDecoder
from micm.utils.logging import get_logger

logger = get_logger(__name__)


def band_edges(low: float, high: float, width: float) -> tuple[tuple[float, float], ...]:
    """Contiguous bands of `width` Hz covering [low, high].

    Raises:
        ValueError: if the range is empty or does not divide into whole bands.
    """
    if not 0.0 < low < high:
        raise ValueError(f"need 0 < low < high, got low={low}, high={high}")
    if width <= 0.0:
        raise ValueError(f"width must be positive, got {width}")

    span = high - low
    n_bands = round(span / width)
    if n_bands < 1 or abs(n_bands * width - span) > 1e-9:
        raise ValueError(
            f"band range {low} to {high} Hz does not divide into whole {width} Hz bands"
        )
    return tuple((low + i * width, low + (i + 1) * width) for i in range(n_bands))


def bandpass(X: np.ndarray, low: float, high: float, *, sfreq: float, order: int) -> np.ndarray:
    """Zero-phase Butterworth bandpass along the last axis.

    Raises:
        ValueError: if the band is not below the Nyquist frequency.
    """
    nyquist = sfreq / 2.0
    if not 0.0 < low < high < nyquist:
        raise ValueError(f"band {low}-{high} Hz is not inside (0, {nyquist}) Hz")

    sos = butter(order, [low, high], btype="bandpass", fs=sfreq, output="sos")
    return np.asarray(sosfiltfilt(sos, X, axis=-1))


class FBCSPDecoder(BaseDecoder):
    """Filter-bank CSP, mutual-information selection, shrinkage LDA."""

    name = "fbcsp"

    def __init__(
        self,
        *,
        seed: int,
        sfreq: float,
        band_low: float,
        band_high: float,
        band_width: float,
        filter_order: int,
        n_components: int,
        csp_reg: str | float | None,
        n_features: int,
        lda_solver: str,
        lda_shrinkage: str | float | None,
    ) -> None:
        super().__init__(seed=seed)
        self.sfreq = float(sfreq)
        self.bands = band_edges(band_low, band_high, band_width)
        self.filter_order = int(filter_order)
        self.n_components = int(n_components)
        self.csp_reg = csp_reg
        self.n_features = int(n_features)
        self.lda_solver = str(lda_solver)
        self.lda_shrinkage = lda_shrinkage

        self._csps: list[CSP] = []
        self._selected: np.ndarray | None = None
        self._lda: LinearDiscriminantAnalysis | None = None

    def _band_features(self, X: np.ndarray, *, fit: bool, y: np.ndarray | None = None) -> np.ndarray:
        """Log-variance CSP features, (n_trials, n_bands * n_components).

        `fit=True` learns one CSP per band; otherwise the stored ones are applied.
        """
        blocks: list[np.ndarray] = []
        for index, (low, high) in enumerate(self.bands):
            filtered = bandpass(
                X, low, high, sfreq=self.sfreq, order=self.filter_order
            )
            if fit:
                csp = CSP(
                    n_components=self.n_components,
                    reg=self.csp_reg,
                    log=True,
                    transform_into="average_power",
                    norm_trace=False,
                )
                blocks.append(csp.fit_transform(filtered, y))
                self._csps.append(csp)
            else:
                blocks.append(self._csps[index].transform(filtered))
        return np.concatenate(blocks, axis=1)

    def _fit(self, X: np.ndarray, y: np.ndarray, n_classes: int) -> None:
        self._csps = []
        features = self._band_features(X, fit=True, y=y)

        n_available = features.shape[1]
        if self.n_features > n_available:
            raise ValueError(
                f"n_features={self.n_features} exceeds the {n_available} features produced "
                f"by {len(self.bands)} bands x {self.n_components} components"
            )

        scores = mutual_info_classif(features, y, random_state=self.seed)
        # Ties broken by index so the selection is reproducible across platforms.
        self._selected = np.sort(np.argsort(-scores, kind="stable")[: self.n_features])

        self._lda = LinearDiscriminantAnalysis(
            solver=self.lda_solver, shrinkage=self.lda_shrinkage
        )
        self._lda.fit(features[:, self._selected], y)
        logger.debug(
            "fbcsp fitted: %d bands, %d features selected of %d, %d classes",
            len(self.bands),
            self.n_features,
            n_available,
            n_classes,
        )

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._lda is None or self._selected is None:  # pragma: no cover - guarded by base
            raise RuntimeError("fbcsp is not fitted")
        features = self._band_features(X, fit=False)
        return np.asarray(self._lda.predict_proba(features[:, self._selected]))
