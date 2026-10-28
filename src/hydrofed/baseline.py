"""Independent per-plant baseline training."""
from pathlib import Path
import numpy as np
import torch
from hydrofed.data.pipeline import load_client_partition
from hydrofed.models.lstm_autoencoder import build_model
from hydrofed.training.local import train_epochs, reconstruction_errors, calibrate_threshold
from hydrofed.evaluation import evaluate_model
from hydrofed.storage import ExperimentStore
from hydrofed.data.splits import split_plant_windows
from hydrofed.runtime import resolve_device
from hydrofed.reproducibility import set_random_seed

def train_baselines(
    partition_dir,
    output_dir,
    epochs=1,
    batch_size=64,
    model_config=None,
    db_path=None,
    device: str = "auto",
    seed: int = 42,
):
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True); results = {}
    set_random_seed(seed)
    device = resolve_device(device)
    store = ExperimentStore(db_path) if db_path else None
    run_id = None

    if store:
        run_id = store.create_run(
            "baseline",
            {
                "partition_dir": str(partition_dir),
                "output_dir": str(output_dir),
                "epochs": epochs,
                "batch_size": batch_size,
                "model_config": model_config or {},
                "device": str(device),
                "seed": seed,
            },
        )

    try:
        for file in sorted(Path(partition_dir).glob("plant_*.npz")):
            windows, labels = load_client_partition(file)
            plant_split = split_plant_windows(windows, labels)

            model = build_model(windows.shape[2], model_config)

            loss = train_epochs(
                model,
                plant_split.train_windows,
                epochs,
                batch_size,
                device=device,
            )

            calibration_errors = reconstruction_errors(
                model,
                plant_split.calibration_windows,
                device=device,
            )

            threshold = calibrate_threshold(calibration_errors)

            metrics = evaluate_model(
                model,
                plant_split.test_windows,
                plant_split.test_labels,
                threshold,
                device=device,
            ) | {"loss": loss}

            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "n_features": windows.shape[2],
                    "threshold": threshold,
                },
                output / f"{file.stem}.pt",
            )

            results[file.stem] = metrics

            if store:
                store.record(
                    "baseline",
                    "evaluation",
                    metrics,
                    client_id=file.stem,
                    run_id=run_id,
                )
    except Exception as error:
        if store and run_id:
            store.complete_run(
                run_id,
                status="failed",
                error_message=str(error),
            )
        raise
    else:
        if store and run_id:
            store.complete_run(run_id)

    return results