"""Small script for inspecting the LSTM autoencoder.

This is not part of training yet. It simply helps us see the tensor shapes
that move through the model.
"""

from __future__ import annotations

import torch

from hydrofed.models.lstm_autoencoder import LSTMAutoencoder


def main() -> None:
    batch_size = 4
    window_length = 32
    n_features = 8

    model = LSTMAutoencoder(
        n_features=n_features,
        hidden_size=64,
        latent_size=16,
        num_layers=2,
        dropout=0.1,
    )

    batch = torch.randn(batch_size, window_length, n_features)

    reconstruction = model(batch)
    errors = model.reconstruction_errors(batch)

    print("Input shape:          ", tuple(batch.shape))
    print("Reconstruction shape: ", tuple(reconstruction.shape))
    print("Errors shape:         ", tuple(errors.shape))
    print("Trainable parameters: ", model.count_trainable_parameters())


if __name__ == "__main__":
    main()