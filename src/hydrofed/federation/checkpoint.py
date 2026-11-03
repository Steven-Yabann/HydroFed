"""Save a federated global model as a reusable PyTorch checkpoint."""

from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import torch

from hydrofed.models.lstm_autoencoder import build_model


def _checkpoint_value(value):
    """Convert NumPy values into safe PyTorch checkpoint metadata."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _checkpoint_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_checkpoint_value(item) for item in value]
    return value


def save_global_checkpoint(
    parameters: list[np.ndarray],
    n_features: int,
    output_path: str | Path,
    model_config: dict | None = None,
    preprocessing: dict | None = None,
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
            "preprocessing": _checkpoint_value(preprocessing or {}),
        },
        path,
    )

    return path


def load_global_checkpoint(path: str | Path) -> dict:
    """Load a HydroFed checkpoint, including older local NumPy metadata.

    Current checkpoints are safe to load with ``weights_only=True``. The
    fallback exists only for earlier checkpoints written by HydroFed before
    preprocessing metadata was converted to Python lists.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        return torch.load(path, map_location="cpu", weights_only=False)
