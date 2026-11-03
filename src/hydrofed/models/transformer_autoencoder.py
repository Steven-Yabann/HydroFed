"""Transformer autoencoder for time-series anomaly detection.

This is the attention-based counterpart to the LSTM autoencoder. It keeps the
same contract and the same bottleneck: the encoder compresses a window into
one latent vector, and the decoder rebuilds the window from it. Without the
bottleneck, self-attention could copy each time step to the output and
reconstruct faults as well as normal data.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from hydrofed.models.parameters import NumpyParametersMixin


def sinusoidal_positions(length: int, d_model: int) -> torch.Tensor:
    """Return fixed sine and cosine position encodings of shape (length, d_model)."""
    positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10_000.0) / d_model)
    )
    encoding = torch.zeros(length, d_model)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    encoding[:, 1::2] = torch.cos(positions * frequencies[: d_model // 2])
    return encoding


class TransformerAutoencoder(NumpyParametersMixin, nn.Module):
    """Sequence autoencoder built with transformer encoder layers.

    Expected input and output shape:
        (batch_size, window_length, n_features)
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        feedforward_size: int = 64,
        latent_size: int = 4,
        dropout: float = 0.1,
        max_length: int = 256,
    ) -> None:
        super().__init__()

        if n_features <= 0:
            raise ValueError("n_features must be positive")
        if d_model <= 0 or d_model % 2 or d_model % num_heads:
            raise ValueError("d_model must be positive, even and divisible by num_heads")
        if latent_size <= 0:
            raise ValueError("latent_size must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        self.n_features = n_features
        self.latent_size = latent_size

        def stack():
            layer = nn.TransformerEncoderLayer(
                d_model, num_heads, feedforward_size, dropout, batch_first=True,
            )
            return nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)

        self.input_projection = nn.Linear(n_features, d_model)
        self.encoder = stack()
        self.to_latent = nn.Linear(d_model, latent_size)
        self.from_latent = nn.Linear(latent_size, d_model)
        self.decoder = stack()
        self.output_layer = nn.Linear(d_model, n_features)
        # Fixed encodings are not weights, so FedAvg does not exchange them.
        self.register_buffer("positions", sinusoidal_positions(max_length, d_model), persistent=False)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Reconstruct a batch of telemetry windows."""
        if batch.ndim != 3 or batch.shape[2] != self.n_features:
            raise ValueError(
                f"Expected batch with shape (batch_size, window_length, {self.n_features})"
            )
        length = batch.shape[1]
        if length > len(self.positions):
            raise ValueError(f"Windows longer than {len(self.positions)} steps are not supported")

        positions = self.positions[:length]
        encoded = self.encoder(self.input_projection(batch) + positions)
        latent = self.to_latent(encoded.mean(dim=1))
        repeated = self.from_latent(latent).unsqueeze(1).expand(-1, length, -1)
        return self.output_layer(self.decoder(repeated + positions))

    def reconstruction_errors(self, batch: torch.Tensor) -> torch.Tensor:
        """Return one mean squared reconstruction error per window."""
        return ((batch - self(batch)) ** 2).mean(dim=(1, 2))


def build_transformer_model(n_features: int, model_config: dict | None = None) -> TransformerAutoencoder:
    """Create a transformer autoencoder from a config dictionary."""
    model_config = model_config or {}
    return TransformerAutoencoder(
        n_features=n_features,
        d_model=int(model_config.get("d_model", 32)),
        num_heads=int(model_config.get("num_heads", 4)),
        num_layers=int(model_config.get("num_layers", 2)),
        feedforward_size=int(model_config.get("feedforward_size", 64)),
        latent_size=int(model_config.get("latent_size", 4)),
        dropout=float(model_config.get("dropout", 0.1)),
    )
