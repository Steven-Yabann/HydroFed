"""Save a federated global model as a reusable PyTorch checkpoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from hydrofed.models.lstm_autoencoder import build_model


def save_global_checkpoint(
    parameters: list[np.ndarray],
    n_features: int,
    output_path: str | Path,
    model_config: dict | None = None,
) -> Path:
    """Convert final Flower parameters into a saved PyTorch model checkpoint."""

    model = build_model(n_features, model_config)
    model.set_parameters_numpy(parameters)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_features": n_features,
            "model_config": model_config or {},
        },
        path,
    )

    return path