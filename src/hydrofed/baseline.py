"""Independent per-plant baseline training."""
from pathlib import Path
import numpy as np
import torch
from hydrofed.data.pipeline import load_client_partition
from hydrofed.models.lstm_autoencoder import build_model
from hydrofed.training.local import train_epochs, reconstruction_errors, calibrate_threshold
from hydrofed.evaluation import compute_metrics
from hydrofed.storage import ExperimentStore

def train_baselines(partition_dir, output_dir, epochs=1, batch_size=64, model_config=None, db_path=None):
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True); results = {}
    store = ExperimentStore(db_path) if db_path else None
    for file in sorted(Path(partition_dir).glob("plant_*.npz")):
        windows, labels = load_client_partition(file)
        normal = windows[labels == 0]
        split = max(1, int(.8 * len(normal)))
        model = build_model(windows.shape[2], model_config)
        loss = train_epochs(model, normal[:split], epochs, batch_size)
        calibration = reconstruction_errors(model, normal[split:] if len(normal[split:]) else normal[:split])
        threshold = calibrate_threshold(calibration)
        metrics = compute_metrics(labels, reconstruction_errors(model, windows), threshold) | {"loss": loss}
        torch.save({"state_dict": model.state_dict(), "n_features": windows.shape[2], "threshold": threshold}, output / f"{file.stem}.pt")
        results[file.stem] = metrics
        if store: store.record("baseline", "evaluation", metrics, file.stem)
    return results
