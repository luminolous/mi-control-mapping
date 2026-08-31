"""Riemannian tangent-space decoder.

The strong end of the decoder range. Expected Cohen's kappa on BCI IV-2a is
roughly 0.55 to 0.70.

Pipeline: spatial covariance per trial with OAS shrinkage, projection into the
tangent space at the Riemannian mean of the training set, then multinomial
logistic regression. Logistic regression returns calibrated posteriors directly.

The Riemannian mean is a fitted parameter. It is estimated inside `_fit` and
reused unchanged at predict time, so it never sees evaluation trials. Refitting
it on the combined set is a common and completely silent form of leakage: it
raises nothing, it improves every score, and it is invisible in the output.
"""

from __future__ import annotations

import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression

from micm.decoders.base import BaseDecoder
from micm.utils.logging import get_logger

logger = get_logger(__name__)


class RiemannDecoder(BaseDecoder):
    """Covariance, tangent space, multinomial logistic regression."""

    name = "riemann"

    def __init__(
        self,
        *,
        seed: int,
        cov_estimator: str,
        metric: str,
        logreg_c: float,
        max_iter: int,
    ) -> None:
        super().__init__(seed=seed)
        self.cov_estimator = str(cov_estimator)
        self.metric = str(metric)
        self.logreg_c = float(logreg_c)
        self.max_iter = int(max_iter)

        self._covariances: Covariances | None = None
        self._tangent: TangentSpace | None = None
        self._logreg: LogisticRegression | None = None

    def _fit(self, X: np.ndarray, y: np.ndarray, n_classes: int) -> None:
        self._covariances = Covariances(estimator=self.cov_estimator)
        covariances = self._covariances.fit_transform(X)

        # Fitted on train covariances only, then frozen. See the module docstring.
        self._tangent = TangentSpace(metric=self.metric)
        features = self._tangent.fit_transform(covariances)

        self._logreg = LogisticRegression(
            C=self.logreg_c,
            max_iter=self.max_iter,
            random_state=self.seed,
        )
        self._logreg.fit(features, y)
        logger.debug(
            "riemann fitted: %d tangent features, %d classes", features.shape[1], n_classes
        )

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        if (
            self._covariances is None or self._tangent is None or self._logreg is None
        ):  # pragma: no cover - guarded by base
            raise RuntimeError("riemann is not fitted")
        covariances = self._covariances.transform(X)
        features = self._tangent.transform(covariances)
        return np.asarray(self._logreg.predict_proba(features))
