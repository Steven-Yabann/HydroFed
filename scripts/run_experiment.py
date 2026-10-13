#!/usr/bin/env python3
import argparse
import time
from pathlib import Path
import flwr as fl
import json

from hydrofed.evaluation import evaluate_model
from hydrofed.storage import ExperimentStore
from hydrofed.training.local import calibrate_threshold, reconstruction_errors
from flwr.common import ndarrays_to_parameters
from hydrofed.data.pipeline import load_client_partition
from hydrofed.federation.client import HydroFedClient
from hydrofed.federation.strategy import TrackingFedAvg
from hydrofed.models.lstm_autoencoder import build_model
from hydrofed.monitoring import serve, F1
from hydrofed.federation.checkpoint import save_global_checkpoint

p = argparse.ArgumentParser()

p.add_argument("--partitions", default="data/partitions")
p.add_argument("--rounds", type=int, default=2)
p.add_argument("--epochs", type=int, default=1)
p.add_argument("--db", default="artifacts/experiments.db")
p.add_argument("--keep-metrics-seconds", type=int, default=0)
p.add_argument("--output", default="artifacts/federated/global_model.pt")
p.add_argument("--evaluate", action="store_true")
p.add_argument("--threshold-percentile", type=float, default=95.0)

a=p.parse_args(); files=sorted(Path(a.partitions).glob("plant_*.npz"))
if not files: raise SystemExit("No plant partitions found; run make partition-synthetic")
n_features=load_client_partition(files[0])[0].shape[2]
serve(8000)
def client_fn(cid): return HydroFedClient(cid, files[int(cid)], epochs=a.epochs).to_client()
strategy=TrackingFedAvg(db_path=a.db, fraction_fit=1.0, fraction_evaluate=1.0, min_available_clients=len(files), initial_parameters=ndarrays_to_parameters(build_model(n_features).get_parameters_numpy()))
fl.simulation.start_simulation(client_fn=client_fn,num_clients=len(files),config=fl.server.ServerConfig(num_rounds=a.rounds),strategy=strategy,client_resources={"num_cpus":1})
final_parameters = strategy.get_final_parameters_numpy()

checkpoint_path = save_global_checkpoint(
    final_parameters,
    n_features,
    a.output,
)

print(f"Saved final global model: {checkpoint_path}")


global_model = build_model(n_features)
global_model.set_parameters_numpy(final_parameters)

if a.evaluate:
    store = ExperimentStore(a.db)
    evaluation_results = {}

    for file in files:
        windows, labels = load_client_partition(file)

        normal_windows = windows[labels == 0]
        split = max(1, int(0.8 * len(normal_windows)))

        calibration_windows = normal_windows[split:]
        if len(calibration_windows) == 0:
            calibration_windows = normal_windows[:split]

        calibration_errors = reconstruction_errors(
            global_model,
            calibration_windows,
        )

        threshold = calibrate_threshold(
            calibration_errors,
            a.threshold_percentile,
        )

        metrics = evaluate_model(
            global_model,
            windows,
            labels,
            threshold,
        )

        evaluation_results[file.stem] = metrics

        store.record(
            "federated",
            "evaluation",
            metrics,
            client_id=file.stem,
        )

    macro_f1 = sum(
    result["f1"] for result in evaluation_results.values()
    ) / len(evaluation_results)

    F1.set(macro_f1)
    print(json.dumps(evaluation_results, indent=2))

if a.keep_metrics_seconds > 0:
    time.sleep(a.keep_metrics_seconds)