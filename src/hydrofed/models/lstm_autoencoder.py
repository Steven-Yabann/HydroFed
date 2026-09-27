"""LSTM autoencoder for time-series anomaly detection.

The model learns to reconstruct normal telemetry windows. Later, if a
window reconstructs poorly, we treat its reconstruction error as evidence
that the window may be anomalous.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
from torch import nn


class LSTMAutoencoder(nn.Module):
    """Sequence-to-sequence autoencoder built with LSTM layers.

    Expected input shape:
        (batch_size, window_length, n_features)

    Output shape:
        (batch_size, window_length, n_features)

    The encoder reads the full telemetry window and compresses it into one
    latent vector. The decoder then repeats that latent vector across the
    window length and tries to reconstruct the original sequence.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        latent_size: int = 16,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if n_features <= 0:
            raise ValueError("n_features must be positive")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if latent_size <= 0:
            raise ValueError("latent_size must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        self.n_features = n_features
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.to_latent = nn.Linear(hidden_size, latent_size)

        self.decoder = nn.LSTM(
            input_size=latent_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output_layer = nn.Linear(hidden_size, n_features)

    def _validate_batch(self, batch: torch.Tensor) -> None:
        """Check that the input tensor matches the model contract."""

        if batch.ndim != 3:
            raise ValueError(
                "Expected batch with shape "
                "(batch_size, window_length, n_features)"
            )

        if batch.shape[2] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, "
                f"but received {batch.shape[2]}"
            )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Reconstruct a batch of telemetry windows."""

        self._validate_batch(batch)

        _, (hidden, _) = self.encoder(batch)

        final_hidden_state = hidden[-1]

        latent = self.to_latent(final_hidden_state)

        repeated_latent = latent.unsqueeze(1).expand(-1, batch.shape[1], -1)

        decoded, _ = self.decoder(repeated_latent)

        reconstruction = self.output_layer(decoded)

        return reconstruction

    def reconstruction_errors(self, batch: torch.Tensor) -> torch.Tensor:
        """Return one mean squared reconstruction error per window."""

        reconstruction = self(batch)

        squared_error = (batch - reconstruction) ** 2

        return squared_error.mean(dim=(1, 2))

    def get_parameters_numpy(self) -> list[np.ndarray]:
        """Return model weights as NumPy arrays.

        Flower exchanges model parameters as NumPy arrays, so this method
        converts the PyTorch state dictionary into Flower-friendly values.
        """

        return [value.detach().cpu().numpy() for value in self.state_dict().values()]

    def set_parameters_numpy(self, parameters: list[np.ndarray]) -> None:
        """Load model weights from NumPy arrays."""

        current_state = self.state_dict()

        if len(parameters) != len(current_state):
            raise ValueError(
                f"Expected {len(current_state)} parameter arrays, "
                f"but received {len(parameters)}"
            )

        new_state = OrderedDict()

        for key, value in zip(current_state.keys(), parameters):
            current_tensor = current_state[key]
            new_state[key] = torch.as_tensor(
                value,
                dtype=current_tensor.dtype,
                device=current_tensor.device,
            )

        self.load_state_dict(new_state, strict=True)

    def count_trainable_parameters(self) -> int:
        """Return the number of trainable parameters in the model."""

        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def build_model(n_features: int, model_config: dict | None = None) -> LSTMAutoencoder:
    """Create an LSTM autoencoder from a config dictionary."""

    model_config = model_config or {}

    return LSTMAutoencoder(
        n_features=n_features,
        hidden_size=int(model_config.get("hidden_size", 64)),
        latent_size=int(model_config.get("latent_size", 16)),
        num_layers=int(model_config.get("num_layers", 2)),
        dropout=float(model_config.get("dropout", 0.1)),
    )