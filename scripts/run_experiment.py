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
    evaluate_skab_scores,
    hybrid_recording_scores,
)
from hydrofed.storage import ExperimentStore
from hydrofed.training.local import (
    calibrate_threshold,
    reconstruction_absolute_errors,
    reconstruction_errors,
    reconstruction_signed_residuals,
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
    WindowStandardScaler,
    fit_shared_standard_window_scaler,
    fit_shared_window_scaler,
    scale_plant_split,
)
from hydrofed.runtime import resolve_device
from hydrofed.reproducibility import set_random_seed

p = argparse.ArgumentParser()

p.add_argument("--partitions", default="data/partitions")
p.add_argument(
    "--clients",
    default="",
    help="Comma-separated plants to train, such as plant_1,plant_3 (default: every partition)",
)
p.add_argument("--fraction-fit", type=float, default=1.0)
p.add_argument("--metrics-port", type=int, default=8000)
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
    "--skab-training-policy",
    choices=("chronological", "normal-only"),
    default="chronological",
    help="Use every early SKAB window or retain only label-confirmed normal windows",
)
p.add_argument(
    "--skab-threshold-scope",
    choices=("plant", "recording"),
    default="plant",
    help="Calibrate one SKAB threshold per plant or per recording",
)
p.add_argument(
    "--skab-scaling",
    choices=("shared", "plant"),
    default="shared",
    help="Pool scaler statistics across plants or fit them locally at each plant",
)
p.add_argument(
    "--skab-score",
    choices=("reconstruction", "hybrid"),
    default="reconstruction",
    help="Use the SKAB absolute reconstruction score or the per-recording hybrid",
)
p.add_argument("--skab-smoothing", type=float, default=0.2)
p.add_argument("--skab-mad-multiplier", type=float, default=24.0)
p.add_argument("--skab-calibration-tail", type=float, default=0.25)
p.add_argument(
    "--skab-exclude-anomaly-free",
    action="store_true",
    help="Train on the anomaly-free recording without testing on it, as SKAB does",
)

a=p.parse_args(); files=sorted(Path(a.partitions).glob("plant_*.npz"))

if a.clients:
    requested = [name.strip() for name in a.clients.split(",") if name.strip()]
    unknown = sorted(set(requested) - {file.stem for file in files})
    if unknown:
        p.error(f"--clients names unknown plants: {', '.join(unknown)}")
    files = [file for file in files if file.stem in requested]
if not 0 < a.fraction_fit <= 1:
    p.error("--fraction-fit must be in (0, 1]")

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

excluded_test_recordings = ("anomaly-free",) if a.skab_exclude_anomaly_free else ()


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
        window_labels=labels,
        training_policy=a.skab_training_policy,
        exclude_test_recordings=excluded_test_recordings,
    )


raw_plant_splits = {}
for file in files:
    raw_plant_splits[file] = build_plant_split(file)

if a.evaluation_protocol == "skab" and a.skab_scaling == "plant":
    # Local statistics never leave a plant, and each plant keeps its own units.
    plant_scalers = {
        file: WindowStandardScaler.fit(plant_split.train_windows)
        for file, plant_split in raw_plant_splits.items()
    }
    scaling_name = "plant_skab_train_standard"
elif a.evaluation_protocol == "skab":
    shared_scaler = fit_shared_standard_window_scaler(
        plant_split.train_windows
        for plant_split in raw_plant_splits.values()
    )
    plant_scalers = dict.fromkeys(raw_plant_splits, shared_scaler)
    scaling_name = "shared_skab_train_standard"
else:
    shared_scaler = fit_shared_window_scaler(
        plant_split.train_windows
        for plant_split in raw_plant_splits.values()
    )
    plant_scalers = dict.fromkeys(raw_plant_splits, shared_scaler)
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
        "clients": [file.stem for file in files],
        "fraction_fit": a.fraction_fit,
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
        "skab_training_policy": a.skab_training_policy,
        "skab_score": a.skab_score,
        "skab_smoothing": a.skab_smoothing,
        "skab_mad_multiplier": a.skab_mad_multiplier,
        "skab_calibration_tail": a.skab_calibration_tail,
        "skab_exclude_anomaly_free": a.skab_exclude_anomaly_free,
    },
)

print(f"Started federated run: {run_id}")

try:
    serve(a.metrics_port)

    def client_fn(cid):
        # Record rounds under the plant's name, so a subset of plants
        # (--clients) still maps back to the right partition.
        return HydroFedClient(
            files[int(cid)].stem,
            files[int(cid)],
            epochs=a.epochs,
            batch_size=a.batch_size,
            learning_rate=a.learning_rate,
            device=device,
            seed=a.seed + int(cid),
            scaler=plant_scalers[files[int(cid)]],
            evaluation_protocol=a.evaluation_protocol,
            skab_train_size=a.skab_train_size,
            skab_training_policy=a.skab_training_policy,
            skab_exclude_test_recordings=excluded_test_recordings,
            model_config=model_config,
        ).to_client()

    strategy = TrackingFedAvg(
        db_path=a.db,
        run_id=run_id,
        fraction_fit=a.fraction_fit,
        fraction_evaluate=1.0,
        min_fit_clients=max(1, round(a.fraction_fit * len(files))),
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
                    "plant_scalers": {
                        file.stem: {
                            "means": scaler.means,
                            "standard_deviations": scaler.standard_deviations,
                        }
                        for file, scaler in plant_scalers.items()
                    },
                    "skab_train_size": a.skab_train_size,
                    "skab_training_policy": a.skab_training_policy,
                }
                if scaling_name == "plant_skab_train_standard"
                else {
                    "means": shared_scaler.means,
                    "standard_deviations": shared_scaler.standard_deviations,
                    "skab_train_size": a.skab_train_size,
                    "skab_training_policy": a.skab_training_policy,
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
                plant_scalers[file],
            )

            if a.evaluation_protocol == "skab" and a.skab_score == "hybrid":
                test_scores, threshold = hybrid_recording_scores(
                    plant_split.train_windows.mean(axis=1),
                    reconstruction_signed_residuals(
                        global_model,
                        plant_split.train_windows,
                        device=device,
                    ),
                    plant_split.train_recording_ids,
                    plant_split.test_windows.mean(axis=1),
                    reconstruction_signed_residuals(
                        global_model,
                        plant_split.test_windows,
                        device=device,
                    ),
                    plant_split.test_recording_ids,
                    smoothing=a.skab_smoothing,
                    mad_multiplier=a.skab_mad_multiplier,
                    calibration_tail=a.skab_calibration_tail,
                )
                metrics = evaluate_skab_scores(
                    test_scores,
                    plant_split.test_labels,
                    plant_split.test_recording_ids,
                    threshold,
                    consecutive_sequences=a.skab_persistence,
                )
            elif a.evaluation_protocol == "skab":
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
