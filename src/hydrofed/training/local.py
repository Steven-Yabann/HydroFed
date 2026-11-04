"""Local training utilities shared by baseline and federated modes.

In HydroFed, every simulated hydropower plant trains on its own local
windows. This module contains the basic PyTorch pieces needed for that:
a Dataset wrapper and a simple training loop.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from hydrofed.models.lstm_autoencoder import LSTMAutoencoder


class PlantDataset(Dataset):
    """Wrap pre-windowed plant telemetry for PyTorch.

    Expected input shape:
        (num_windows, window_length, n_features)

    Each item returned by the dataset is one telemetry window.
    """

    def __init__(self, windows: np.ndarray) -> None:
        if windows.ndim != 3:
            raise ValueError(
                "windows must have shape "
                "(num_windows, window_length, n_features)"
            )

        if windows.shape[0] == 0:
            raise ValueError("windows must contain at least one window")

        self.windows = torch.as_tensor(windows, dtype=torch.float32)

    def __len__(self) -> int:
        return self.windows.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.windows[index]


def train_epochs(
    model: LSTMAutoencoder,
    train_windows: np.ndarray,
    epochs: int = 1,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    proximal_weights: list[np.ndarray] | None = None,
    proximal_mu: float = 0.0,
) -> float:
    """Train an autoencoder on normal telemetry windows.

    With ``proximal_weights`` and a positive ``proximal_mu``, the loss adds
    FedProx's term mu/2 * ||w - w_global||^2, which keeps a client's update
    close to the global model it started from.

    Returns:
        The average reconstruction loss from the final epoch.
    """

    if epochs <= 0:
        raise ValueError("epochs must be positive")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if proximal_mu < 0:
        raise ValueError("proximal_mu must be non-negative")

    dataset = PlantDataset(train_windows)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    anchors = []
    if proximal_weights is not None and proximal_mu > 0:
        # Copy the anchors: arrays from get_parameters_numpy share memory with
        # the live weights, so an uncopied anchor would move with them.
        weights = dict(zip(model.state_dict().keys(), proximal_weights))
        anchors = [
            (parameter, torch.tensor(weights[name], dtype=parameter.dtype, device=device))
            for name, parameter in model.named_parameters()
        ]

    final_epoch_loss = 0.0

    for _ in range(epochs):
        total_loss = 0.0
        total_examples = 0

        for batch in loader:
            batch = batch.to(device)

            optimizer.zero_grad()

            reconstruction = model(batch)

            loss = criterion(reconstruction, batch)
            objective = loss
            if anchors:
                objective = loss + proximal_mu / 2 * sum(
                    ((parameter - anchor) ** 2).sum() for parameter, anchor in anchors
                )

            objective.backward()

            optimizer.step()

            total_loss += loss.item() * batch.shape[0]
            total_examples += batch.shape[0]

        final_epoch_loss = total_loss / total_examples

    return float(final_epoch_loss)


def reconstruction_errors(
    model: LSTMAutoencoder,
    windows: np.ndarray,
    batch_size: int = 256,
    device: str = "cpu",
) -> np.ndarray:
    """Score windows without building an autograd graph."""
    loader = DataLoader(PlantDataset(windows), batch_size=batch_size, shuffle=False)
    model.to(device)
    model.eval()
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            scores.append(model.reconstruction_errors(batch.to(device)).cpu().numpy())
    return np.concatenate(scores).astype(np.float64)


def reconstruction_absolute_errors(
    model: LSTMAutoencoder,
    windows: np.ndarray,
    batch_size: int = 256,
    device: str = "cpu",
) -> np.ndarray:
    """Return the SKAB-style absolute reconstruction score for each sequence.

    For each sequence, SKAB averages absolute error across time for every
    sensor, then sums those sensor-level errors into one anomaly score.
    """
    loader = DataLoader(PlantDataset(windows), batch_size=batch_size, shuffle=False)
    model.to(device)
    model.eval()
    scores: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            reconstruction = model(batch)
            score = torch.abs(batch - reconstruction).mean(dim=1).sum(dim=1)
            scores.append(score.cpu().numpy())

    return np.concatenate(scores).astype(np.float64)


def reconstruction_signed_residuals(
    model: LSTMAutoencoder,
    windows: np.ndarray,
    batch_size: int = 256,
    device: str = "cpu",
) -> np.ndarray:
    """Return each sequence's mean signed residual per sensor.

    Keeping the sign and the sensors separate matters for SKAB faults, which
    often move one sensor only. An absolute error summed across sensors can
    dilute that movement, or shrink when a fault cancels an existing offset.
    """
    loader = DataLoader(PlantDataset(windows), batch_size=batch_size, shuffle=False)
    model.to(device)
    model.eval()
    residuals: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            residuals.append((batch - model(batch)).mean(dim=1).cpu().numpy())

    return np.concatenate(residuals).astype(np.float64)


def calibrate_threshold(errors: np.ndarray, percentile: float = 95.0) -> float:
    """Choose a threshold from normal validation reconstruction errors."""
    values = np.asarray(errors, dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("errors must contain finite values")
    if not 0.0 < percentile <= 100.0:
        raise ValueError("percentile must be in (0, 100]")
    return float(np.percentile(values, percentile))
