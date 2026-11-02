#!/usr/bin/env python3
import argparse
import time
from pathlib import Path
import flwr as fl
import json

from hydrofed.evaluation import (
    calibrate_recording_thresholds,
    evaluate_model,
    evaluate_skab_model,
)
from hydrofed.storage import ExperimentStore
from hydrofed.training.local import (
    calibrate_threshold,
    reconstruction_absolute_errors,
    reconstruction_errors,
)
from flwr.common import ndarrays_to_parameters
from hydrofed.data.pipeline import (
    load_client_partition,
    load_client_partition_metadata,
)
from hydrofed.federation.client import HydroFedClient
from hydrofed.federation.strategy import TrackingFedAvg
from hydrofed.models.lstm_autoencoder import build_model
from hydrofed.monitoring import serve, F1
from hydrofed.federation.checkpoint import save_global_checkpoint
from hydrofed.data.splits import split_plant_windows, split_skab_plant_windows
from hydrofed.data.scaling import (
    fit_shared_standard_window_scaler,
    fit_shared_window_scaler,
    scale_plant_split,
)
from hydrofed.runtime import resolve_device
from hydrofed.reproducibility import set_random_seed

p = argparse.ArgumentParser()

p.add_argument("--partitions", default="data/partitions")
p.add_argument("--rounds", type=int, default=2)
p.add_argument("--epochs", type=int, default=1)
p.add_argument("--batch-size", type=int, default=64)
p.add_argument("--learning-rate", type=float, default=1e-3)
p.add_argument("--hidden-size", type=int, default=64)
p.add_argument("--latent-size", type=int, default=16)
p.add_argument("--num-layers", type=int, default=2)
p.add_argument("--dropout", type=float, default=0.1)
p.add_argument("--db", default="artifacts/experiments.db")
p.add_argument("--keep-metrics-seconds", type=int, default=0)
p.add_argument("--output", default="artifacts/federated/global_model.pt")
p.add_argument("--evaluate", action="store_true")
p.add_argument("--threshold-percentile", type=float, default=95.0)
p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
p.add_argument("--seed", type=int, default=42)
p.add_argument(
    "--evaluation-protocol",
    choices=("window", "skab"),
    default="window",
    help="Use existing window labels or SKAB-aligned point endpoint labels",
)
p.add_argument("--skab-train-size", type=int, default=400)
p.add_argument("--skab-threshold-percentile", type=float, default=99.0)
p.add_argument("--skab-threshold-scale", type=float, default=1.5)
p.add_argument("--skab-persistence", type=int, default=10)
p.add_argument(
    "--skab-threshold-scope",
    choices=("plant", "recording"),
    default="plant",
    help="Calibrate one SKAB threshold per plant or per recording",
)

a=p.parse_args(); files=sorted(Path(a.partitions).glob("plant_*.npz"))

if a.batch_size <= 0:
    p.error("--batch-size must be positive")
if a.learning_rate <= 0:
    p.error("--learning-rate must be positive")
if a.hidden_size <= 0 or a.latent_size <= 0 or a.num_layers <= 0:
    p.error("model sizes and --num-layers must be positive")

model_config = {
    "hidden_size": a.hidden_size,
    "latent_size": a.latent_size,
    "num_layers": a.num_layers,
    "dropout": a.dropout,
}

set_random_seed(a.seed)
device = resolve_device(a.device)
print(f"Using device: {device}")

if not files: 
    raise SystemExit("No plant partitions found; run make partition-synthetic")

n_features=load_client_partition(files[0])[0].shape[2]\


def build_plant_split(file):
    windows, labels = load_client_partition(file)

    if a.evaluation_protocol == "window":
        return split_plant_windows(windows, labels)

    metadata = load_client_partition_metadata(file)
    required_fields = {
        "recording_ids",
        "endpoint_labels",
        "window_start_indices",
    }
    missing = required_fields - metadata.keys()
    if missing:
        raise SystemExit(
            "SKAB protocol requires a temporal partition prepared with the "
            "latest prepare_data.py. Missing: " + ", ".join(sorted(missing))
        )
    return split_skab_plant_windows(
        windows,
        metadata["endpoint_labels"],
        metadata["recording_ids"],
        metadata["window_start_indices"],
        train_size=a.skab_train_size,
    )


raw_plant_splits = {}
for file in files:
    raw_plant_splits[file] = build_plant_split(file)

if a.evaluation_protocol == "skab":
    shared_scaler = fit_shared_standard_window_scaler(
        plant_split.train_windows
        for plant_split in raw_plant_splits.values()
    )
    scaling_name = "shared_skab_train_standard"
else:
    shared_scaler = fit_shared_window_scaler(
        plant_split.train_windows
        for plant_split in raw_plant_splits.values()
    )
    scaling_name = "shared_normal_train_minmax"

run_store = ExperimentStore(a.db)
run_id = run_store.create_run(
    "federated",
    {
        "partitions": str(a.partitions),
        "rounds": a.rounds,
        "local_epochs": a.epochs,
        "batch_size": a.batch_size,
        "learning_rate": a.learning_rate,
        "model": model_config,
        "client_count": len(files),
        "device": str(device),
        "seed": a.seed,
        "evaluate": a.evaluate,
        "threshold_percentile": a.threshold_percentile,
        "scaling": scaling_name,
        "evaluation_protocol": a.evaluation_protocol,
        "skab_train_size": a.skab_train_size,
        "skab_threshold_percentile": a.skab_threshold_percentile,
        "skab_threshold_scale": a.skab_threshold_scale,
        "skab_persistence": a.skab_persistence,
        "skab_threshold_scope": a.skab_threshold_scope,
    },
)

print(f"Started federated run: {run_id}")

try:
    serve(8000)

    def client_fn(cid):
        return HydroFedClient(
            cid,
            files[int(cid)],
            epochs=a.epochs,
            batch_size=a.batch_size,
            learning_rate=a.learning_rate,
            device=device,
            seed=a.seed + int(cid),
            scaler=shared_scaler,
            evaluation_protocol=a.evaluation_protocol,
            skab_train_size=a.skab_train_size,
            model_config=model_config,
        ).to_client()

    strategy = TrackingFedAvg(
        db_path=a.db,
        run_id=run_id,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=len(files),
        initial_parameters=ndarrays_to_parameters(
            build_model(n_features, model_config).get_parameters_numpy()
        ),
    )

    client_resources = {"num_cpus": 1}

    if device == "cuda":
        client_resources["num_gpus"] = 1.0

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(files),
        config=fl.server.ServerConfig(num_rounds=a.rounds),
        strategy=strategy,
        client_resources=client_resources,
    )

    final_parameters = strategy.get_final_parameters_numpy()

    checkpoint_path = save_global_checkpoint(
        final_parameters,
        n_features,
        a.output,
        model_config=model_config,
        preprocessing={
            "scaler": scaling_name,
            **(
                {
                    "means": shared_scaler.means,
                    "standard_deviations": shared_scaler.standard_deviations,
                }
                if a.evaluation_protocol == "skab"
                else {
                    "minimums": shared_scaler.minimums,
                    "spans": shared_scaler.spans,
                }
            ),
        },
    )

    print(f"Saved final global model: {checkpoint_path}")

    global_model = build_model(n_features, model_config)
    global_model.set_parameters_numpy(final_parameters)

    if a.evaluate:
        evaluation_results = {}

        for file in files:
            plant_split = scale_plant_split(
                raw_plant_splits[file],
                shared_scaler,
            )

            if a.evaluation_protocol == "skab":
                train_scores = reconstruction_absolute_errors(
                    global_model,
                    plant_split.train_windows,
                    device=device,
                )
                if a.skab_threshold_scope == "recording":
                    threshold = calibrate_recording_thresholds(
                        train_scores,
                        plant_split.train_recording_ids,
                        a.skab_threshold_percentile,
                        a.skab_threshold_scale,
                    )
                else:
                    threshold = calibrate_threshold(
                        train_scores,
                        a.skab_threshold_percentile,
                    ) * a.skab_threshold_scale
                metrics = evaluate_skab_model(
                    global_model,
                    plant_split.test_windows,
                    plant_split.test_labels,
                    plant_split.test_recording_ids,
                    threshold,
                    consecutive_sequences=a.skab_persistence,
                    device=device,
                )
            else:
                calibration_errors = reconstruction_errors(
                    global_model,
                    plant_split.calibration_windows,
                    device=device,
                )
                threshold = calibrate_threshold(
                    calibration_errors,
                    a.threshold_percentile,
                )
                metrics = evaluate_model(
                    global_model,
                    plant_split.test_windows,
                    plant_split.test_labels,
                    threshold,
                    device=device,
                )

            evaluation_results[file.stem] = metrics

            run_store.record(
                "federated",
                "evaluation",
                metrics,
                client_id=file.stem,
                run_id=run_id,
            )

        macro_f1 = sum(
            result["f1"]
            for result in evaluation_results.values()
        ) / len(evaluation_results)

        F1.set(macro_f1)
        print(json.dumps(evaluation_results, indent=2))
except Exception as error:
    run_store.complete_run(
        run_id,
        status="failed",
        error_message=str(error),
    )
    raise
else:
    run_store.complete_run(run_id)
    print(f"Completed federated run: {run_id}")

if a.keep_metrics_seconds > 0:
    time.sleep(a.keep_metrics_seconds)
