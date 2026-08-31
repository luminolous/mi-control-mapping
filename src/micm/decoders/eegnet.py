"""EEGNet through braindecode, trained with a plain PyTorch loop.

The architecture comes from braindecode; the training loop is written here
rather than using skorch. The loop is thirty lines, and owning it is what makes
the two properties this project needs achievable: a fit that is reproducible
from its seed, and an early-stopping split that provably cannot reach session E.

Determinism is handled by `torch.random.fork_rng`, which seeds the global torch
generators for the duration of the fit and restores whatever was there before.
Seeding torch globally and leaving it that way would make every later component
in the process depend on whether a decoder happened to be fitted first.

Device selection honours the config. A requested CUDA device that is not
available falls back to CPU with a warning, never silently: a run that was meant
to use the GPU and quietly did not is a run whose wall-time numbers mean nothing.

`batch_norm_momentum` is a config parameter and not braindecode's default of
0.01. That default is faithful to the original Keras EEGNet, which trained on
far more batches per epoch. Here a subject contributes about 200 training trials,
so an epoch is a handful of batches and a running estimate with a 0.01 momentum
lags the weights by roughly 25 epochs. Validation runs in eval mode and therefore
reads those stale statistics, so the validation loss sits at ln(K) forever, early
stopping selects an untrained model, and the decoder reports chance while its
training loss falls normally. See docs/decisions.md D20.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import torch
from braindecode.models import EEGNet
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from micm.decoders.base import BaseDecoder
from micm.utils.logging import get_logger

logger = get_logger(__name__)


def resolve_device(requested: str) -> torch.device:
    """Return the torch device to use, warning loudly on a CPU fallback.

    Raises:
        ValueError: on a device string this project does not support.
    """
    name = str(requested).lower()
    if name.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(name)
        logger.warning(
            "config asked for device=%r but torch reports no CUDA device (torch %s). "
            "Falling back to CPU. Training will be slower and any wall-time figure "
            "recorded from this run describes CPU, not GPU",
            requested,
            torch.__version__,
        )
        return torch.device("cpu")
    if name == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unsupported device {requested!r}, expected 'cpu' or 'cuda[:n]'")


class EEGNetDecoder(BaseDecoder):
    """EEGNet v4, AdamW with a cosine schedule, early stopping on an inner split."""

    name = "eegnet"

    def __init__(
        self,
        *,
        seed: int,
        device: str,
        sfreq: float,
        f1: int,
        depth_multiplier: int,
        f2: int,
        kernel_length: int,
        drop_prob: float,
        batch_norm_momentum: float,
        n_epochs: int,
        batch_size: int,
        lr: float,
        weight_decay: float,
        val_fraction: float,
        patience: int,
    ) -> None:
        super().__init__(seed=seed)
        if not 0.0 < val_fraction < 1.0:
            raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
        if n_epochs < 1:
            raise ValueError(f"n_epochs must be at least 1, got {n_epochs}")
        if patience < 1:
            raise ValueError(f"patience must be at least 1, got {patience}")

        self.device = str(device)
        self.sfreq = float(sfreq)
        self.f1 = int(f1)
        self.depth_multiplier = int(depth_multiplier)
        self.f2 = int(f2)
        self.kernel_length = int(kernel_length)
        self.drop_prob = float(drop_prob)
        self.batch_norm_momentum = float(batch_norm_momentum)
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.val_fraction = float(val_fraction)
        self.patience = int(patience)

        self._model: nn.Module | None = None
        self.n_times_: int | None = None
        self.best_epoch_: int | None = None
        self.best_val_loss_: float | None = None

    # --- persistence -------------------------------------------------------

    def __getstate__(self) -> dict[str, Any]:
        """Serialise the weights, not the module object.

        braindecode's EEGNet applies max-norm constraints through
        `torch.nn.utils.parametrize`, and a parametrized module cannot be
        pickled: PyTorch refuses it outright rather than writing something that
        fails to load later. So the checkpoint carries a CPU `state_dict` plus
        the three shapes needed to rebuild the architecture, which also makes a
        GPU-fitted checkpoint open on a machine without a GPU.
        """
        state = self.__dict__.copy()
        model = state.pop("_model", None)
        state["_model"] = None
        state["_model_state"] = (
            None
            if model is None
            else {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        )
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        weights = state.pop("_model_state", None)
        self.__dict__.update(state)
        self._model = None

        if weights is None:
            return
        if self.n_channels_ is None or self.n_times_ is None or self.n_classes_ is None:
            raise ValueError("checkpoint carries weights but not the shapes to rebuild them")

        model = self._build_model(self.n_channels_, self.n_times_, self.n_classes_)
        model.load_state_dict(weights)
        self._model = model.to(resolve_device(self.device)).eval()

    # --- internals ---------------------------------------------------------

    def _build_model(self, n_channels: int, n_times: int, n_classes: int) -> nn.Module:
        model: nn.Module = EEGNet(
            n_chans=n_channels,
            n_outputs=n_classes,
            n_times=n_times,
            F1=self.f1,
            D=self.depth_multiplier,
            F2=self.f2,
            kernel_length=self.kernel_length,
            drop_prob=self.drop_prob,
            batch_norm_momentum=self.batch_norm_momentum,
            sfreq=self.sfreq,
        )
        return model

    def _inner_split(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Stratified train/validation split of the data passed to `fit`.

        The caller passes session T only, so the validation half is drawn from
        session T by construction. There is no path here that could reach the
        evaluation session.
        """
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=self.val_fraction, random_state=self.seed
        )
        inner_train, inner_val = next(splitter.split(np.zeros_like(y), y))
        return inner_train, inner_val

    def _run_epoch(
        self,
        model: nn.Module,
        loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        criterion: nn.Module,
        device: torch.device,
        optimiser: torch.optim.Optimizer | None,
    ) -> float:
        """One pass. Trains when `optimiser` is given, evaluates otherwise."""
        training = optimiser is not None
        model.train(training)

        total = 0.0
        n_seen = 0
        with torch.set_grad_enabled(training):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)

                if optimiser is not None:
                    optimiser.zero_grad(set_to_none=True)
                    loss.backward()
                    optimiser.step()

                total += float(loss.detach()) * len(batch_y)
                n_seen += len(batch_y)
        return total / max(n_seen, 1)

    def _fit(self, X: np.ndarray, y: np.ndarray, n_classes: int) -> None:
        device = resolve_device(self.device)
        inner_train, inner_val = self._inner_split(y)

        features = torch.from_numpy(np.asarray(X, dtype=np.float32))
        targets = torch.from_numpy(np.asarray(y, dtype=np.int64))

        # fork_rng seeds the global torch generators for this fit and restores
        # the previous state on exit, so fitting a decoder does not change the
        # random stream of anything else in the process.
        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            torch.manual_seed(self.seed)

            model = self._build_model(X.shape[1], X.shape[2], n_classes).to(device)
            criterion = nn.CrossEntropyLoss()
            optimiser = torch.optim.AdamW(
                model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimiser, T_max=self.n_epochs
            )

            shuffle_rng = torch.Generator()
            shuffle_rng.manual_seed(self.seed)
            train_loader: DataLoader[Any] = DataLoader(
                TensorDataset(features[inner_train], targets[inner_train]),
                batch_size=min(self.batch_size, len(inner_train)),
                shuffle=True,
                generator=shuffle_rng,
                drop_last=False,
            )
            val_loader: DataLoader[Any] = DataLoader(
                TensorDataset(features[inner_val], targets[inner_val]),
                batch_size=min(self.batch_size, len(inner_val)),
                shuffle=False,
            )

            best_state = copy.deepcopy(model.state_dict())
            best_loss = math.inf
            best_epoch = 0
            since_improved = 0

            for epoch in range(self.n_epochs):
                self._run_epoch(model, train_loader, criterion, device, optimiser)
                scheduler.step()
                val_loss = self._run_epoch(model, val_loader, criterion, device, None)

                if val_loss < best_loss:
                    best_loss, best_epoch, since_improved = val_loss, epoch, 0
                    best_state = copy.deepcopy(model.state_dict())
                else:
                    since_improved += 1
                    if since_improved >= self.patience:
                        logger.debug(
                            "early stop at epoch %d, best %d (val loss %.4f)",
                            epoch,
                            best_epoch,
                            best_loss,
                        )
                        break

            model.load_state_dict(best_state)

        self._model = model.eval()
        self.n_times_ = int(X.shape[2])
        self.best_epoch_ = best_epoch
        self.best_val_loss_ = float(best_loss)

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:  # pragma: no cover - guarded by base
            raise RuntimeError("eegnet is not fitted")

        device = resolve_device(self.device)
        self._model.to(device).eval()

        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                batch = torch.from_numpy(
                    np.asarray(X[start : start + self.batch_size], dtype=np.float32)
                ).to(device)
                logits = self._model(batch)
                outputs.append(torch.softmax(logits, dim=1).cpu().numpy())

        # float64 so the row sums land inside the posterior tolerance after the
        # float32 softmax.
        return np.concatenate(outputs, axis=0).astype(np.float64)
