#!/usr/bin/env python3
"""Run HydroFed's SKAB comparison experiments.

regimes
    Train the LSTM autoencoder federated, centralised and isolated with the
    same inputs, schedule and alert policy, then report detection metrics on
    development and final recordings and the communication each regime needs.

models
    Compare the LSTM autoencoder with transformer autoencoders and classic
    machine learning (Random Forest, XGBoost, Isolation Forest). The best
    candidate in each category is chosen on development recordings, and only
    the chosen models are reported on final recordings.

Settings were chosen on development recordings. Final recordings are for
reporting only: do not tune anything on them.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from hydrofed.experiments.communication import (
    break_even_days,
    federated_bytes,
    raw_skab_bytes,
    streaming_bytes_per_day,
)
from hydrofed.experiments.recordings import ROLES, load_skab_recordings, plant_training_windows
from hydrofed.experiments.regimes import REGIMES, SCORES, autoencoder_scores, parameter_bytes, train_regime
from hydrofed.experiments.scoring import MAD_THRESHOLDS, alert_metrics, choose_threshold, summarise_seeds
from hydrofed.models.lstm_autoencoder import build_model
from hydrofed.models.transformer_autoencoder import build_transformer_model

ARCHITECTURES = {"lstm": build_model, "transformer": build_transformer_model}
CLASSIC_SUPERVISED = {"random_forest", "xgboost"}


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--partitions", default="data/partitions_skab")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--latent-size", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--scaling", choices=("plant", "shared"), default="plant")
    parser.add_argument("--smoothing", type=float, default=0.2)
    parser.add_argument("--calibration-tail", type=float, default=0.25)
    parser.add_argument("--persistence", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--jobs", type=int, default=1, help="parallel training processes")


def lstm_config(args) -> dict:
    return dict(
        hidden_size=args.hidden_size, latent_size=args.latent_size,
        num_layers=args.num_layers, dropout=args.dropout,
    )


def autoencoder_job(job: dict) -> dict:
    """Train and score one autoencoder for one seed. Runs in a worker process.

    With a fixed ``threshold`` every score uses it. Without one, each score's
    threshold is chosen on development recordings by the predeclared rule.
    """
    args = job["args"]
    torch.set_num_threads(job["threads"])
    recordings = load_skab_recordings(args.partitions, args.scaling)
    plant_windows = plant_training_windows(recordings)
    n_features = next(iter(plant_windows.values())).shape[2]
    build = ARCHITECTURES[job["architecture"]]
    result = train_regime(
        job["regime"], plant_windows, lambda: build(n_features, job["config"]),
        rounds=job["rounds"], epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, seed=job["seed"], device=args.device,
    )
    metrics, thresholds = {}, {}
    for score in SCORES:
        scores = autoencoder_scores(
            recordings, result.models, score, args.smoothing, args.calibration_tail, args.device,
        )
        thresholds[score] = job.get("threshold")
        if thresholds[score] is None:
            thresholds[score] = choose_threshold(
                scores.select("dev"), np.array(MAD_THRESHOLDS),
                args.max_false_positive_rate, args.persistence,
            )
        metrics[score] = {
            role: alert_metrics(scores.select(role), thresholds[score], args.persistence)
            for role in ROLES
        }
    print(f"{job['label']} seed {job['seed']}: {result.seconds:.0f}s", flush=True)
    model = next(iter(result.models.values()))
    return {
        "label": job["label"], "seed": job["seed"], "metrics": metrics, "thresholds": thresholds,
        "parameters": model.count_trainable_parameters(),
        "model_bytes": parameter_bytes(model.get_parameters_numpy()),
        "uplink_bytes": result.uplink_bytes, "downlink_bytes": result.downlink_bytes,
        "seconds": result.seconds,
    }


def run_jobs(jobs: list[dict], n_jobs: int) -> list[dict]:
    threads = max(1, (os.cpu_count() or 2) // max(1, n_jobs))
    jobs = [job | {"threads": threads} for job in jobs]
    if n_jobs <= 1:
        return [autoencoder_job(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        return list(pool.map(autoencoder_job, jobs))


def summarise(runs: list[dict], label: str, scores=SCORES) -> dict:
    """Return mean ± sd per score and role for one label's seeds."""
    selected = [run for run in runs if run["label"] == label]
    return {
        score: {role: summarise_seeds([run["metrics"][score][role] for run in selected]) for role in ROLES}
        for score in scores
    }


def communication_summary(args, model_bytes: int, clients: int, round_options: list[int]) -> dict:
    raw = raw_skab_bytes(args.skab_path)
    runs = {}
    for rounds in round_options:
        traffic = federated_bytes(model_bytes, clients, rounds)
        runs[f"{rounds}_rounds"] = traffic | {
            "break_even_days_1hz": break_even_days(traffic["bytes_per_client"], raw["sensors"]),
            "multiple_of_csv": traffic["total_bytes"] / raw["csv_bytes"],
            "multiple_of_binary": traffic["total_bytes"] / raw["binary_bytes"],
            "multiple_of_baseline_binary": traffic["total_bytes"] / raw["baseline_binary_bytes"],
        }
    return {
        "model_bytes": model_bytes,
        "clients": clients,
        "raw_skab": raw,
        "streaming_bytes_per_plant_day_1hz": streaming_bytes_per_day(raw["sensors"]),
        "federated": runs,
    }


def megabytes(value: float) -> str:
    return f"{value / 1e6:.2f} MB"


METRIC_HEADER = "Recall | Precision | F1 | False alarm rate | Within-recording AUC | Events | Delay (s)"


def metric_cells(m: dict) -> str:
    def cell(name):
        return f"{m[name]['mean']:.3f} ± {m[name]['sd']:.3f}" if m[name]["mean"] is not None else "n/a"

    events = f"{m['fault_events_detected']['mean']:.1f}/{m['fault_events']['mean']:.0f}"
    return (f"{cell('recall')} | {cell('precision')} | {cell('f1')} | {cell('false_positive_rate')} | "
            f"{cell('macro_recording_auc_roc')} | {events} | {cell('median_detection_delay')}")


def print_regime_tables(summary: dict, communication: dict) -> None:
    for score in SCORES:
        for role in ROLES:
            print(f"\n{score} score, {role} recordings (mean ± sd over seeds)")
            print(f"| Regime | {METRIC_HEADER} |")
            print("|---" * 8 + "|")
            for label, scores in summary.items():
                print(f"| {label} | {metric_cells(scores[score][role])} |")

    raw = communication["raw_skab"]
    print("\nCommunication")
    print(f"Model: {communication['model_bytes']:,} bytes; SKAB CSVs: {megabytes(raw['csv_bytes'])} "
          f"({raw['csv_files']} files); all rows binary: {megabytes(raw['binary_bytes'])}; "
          f"baseline rows binary: {megabytes(raw['baseline_binary_bytes'])}")
    print("| Federated rounds | Total | Per round | Per plant | × CSVs | × binary rows | Break-even (days at 1 Hz) |")
    print("|---|---|---|---|---|---|---|")
    for name, run in communication["federated"].items():
        print(f"| {name.split('_')[0]} | {megabytes(run['total_bytes'])} | {megabytes(run['bytes_per_round'])} | "
              f"{megabytes(run['bytes_per_client'])} | {run['multiple_of_csv']:.0f}× | "
              f"{run['multiple_of_binary']:.0f}× | {run['break_even_days_1hz']:.1f} |")


def run_regimes(args) -> dict:
    labels = {regime: (regime, args.rounds) for regime in args.regimes}
    for rounds in args.federated_rounds:
        if rounds != args.rounds:
            labels[f"federated_{rounds}r"] = ("federated", rounds)

    jobs = [
        {
            "args": args, "label": label, "regime": regime, "rounds": rounds, "seed": seed,
            "architecture": "lstm", "config": lstm_config(args), "threshold": args.mad_multiplier,
        }
        for label, (regime, rounds) in labels.items()
        for seed in args.seeds
    ]
    runs = run_jobs(jobs, args.jobs)

    recordings = load_skab_recordings(args.partitions, args.scaling)
    model_bytes = runs[0]["model_bytes"]
    communication = communication_summary(
        args, model_bytes, len(recordings), sorted({rounds for _, rounds in labels.values()}),
    )
    for run in runs:
        if labels[run["label"]][0] == "federated":
            expected = federated_bytes(model_bytes, len(recordings), labels[run["label"]][1])
            if run["uplink_bytes"] + run["downlink_bytes"] != expected["total_bytes"]:
                raise RuntimeError(f"Measured bytes for {run['label']} do not match the FedAvg formula")

    summary = {label: summarise(runs, label) for label in labels}
    print_regime_tables(summary, communication)
    return {"runs": runs, "summary": summary, "communication": communication}


def deep_candidates(args) -> dict[str, dict[str, tuple[str, dict]]]:
    """Return autoencoder candidates per category, with fixed settings."""
    transformer = dict(num_heads=4, num_layers=2, latent_size=args.latent_size, dropout=args.dropout)
    return {
        "recurrent": {"lstm_autoencoder": ("lstm", lstm_config(args))},
        "transformer": {
            "transformer_autoencoder_d32": ("transformer", transformer | {"d_model": 32, "feedforward_size": 64}),
            "transformer_autoencoder_d64": ("transformer", transformer | {"d_model": 64, "feedforward_size": 128}),
        },
    }


def run_classic(args) -> dict:
    """Run the classic models in a separate process that never loads PyTorch."""
    with tempfile.TemporaryDirectory() as folder:
        output = Path(folder) / "classic.json"
        subprocess.run([
            sys.executable, "-m", "hydrofed.experiments.classic",
            "--partitions", args.partitions, "--scaling", args.scaling,
            "--seeds", *map(str, args.seeds),
            "--smoothing", str(args.smoothing), "--calibration-tail", str(args.calibration_tail),
            "--max-false-positive-rate", str(args.max_false_positive_rate),
            "--output", str(output),
        ], check=True)
        return json.loads(output.read_text())


def run_models(args) -> dict:
    candidates = deep_candidates(args)
    clients = len(list(Path(args.partitions).glob("plant_*.npz")))
    jobs = [
        {
            "args": args, "label": name, "regime": args.regime, "rounds": args.rounds, "seed": seed,
            "architecture": architecture, "config": config,
        }
        for models in candidates.values()
        for name, (architecture, config) in models.items()
        for seed in args.seeds
    ]
    deep_runs = run_jobs(jobs, args.jobs)
    print("classic models...", flush=True)
    classic_runs = run_classic(args)

    # Each candidate's selection score: the model's own residuals for
    # autoencoders, and the model's output for classic detectors.
    candidate_rows = {}
    for category, models in candidates.items():
        for name in models:
            runs = [run for run in deep_runs if run["label"] == name]
            candidate_rows[name] = {
                "category": category,
                "needs_fault_labels": False,
                "parameters": runs[0]["parameters"],
                "model_bytes": runs[0]["model_bytes"],
                "federated_bytes_per_round": 2 * runs[0]["model_bytes"] * clients if args.regime == "federated" else 0,
                "seconds": float(np.mean([run["seconds"] for run in runs])),
                "summary": summarise(deep_runs, name),
                "selection_score": "residual",
            }
    for name, runs in classic_runs.items():
        roles = {role: summarise_seeds([run[role] for run in runs]) for role in ROLES}
        if name in CLASSIC_SUPERVISED:
            roles["final_unseen_fault_type"] = summarise_seeds([run["final_unseen_fault_type"] for run in runs])
        candidate_rows[name] = {
            "category": "classic",
            "needs_fault_labels": name in CLASSIC_SUPERVISED,
            "summary": {"model": roles},
            "selection_score": "model",
        }

    def dev_auc(name):
        row = candidate_rows[name]
        return row["summary"][row["selection_score"]]["dev"]["macro_recording_auc_roc"]["mean"]

    selected = {}
    for category in ("recurrent", "transformer", "classic"):
        names = [name for name, row in candidate_rows.items() if row["category"] == category]
        selected[category] = max(names, key=dev_auc)

    print("\nCandidates on development recordings (selection uses within-recording AUC)")
    print("| Category | Model | Needs fault labels | Dev AUC | Dev recall | Dev false alarm rate | Selected |")
    print("|---|---|---|---|---|---|---|")
    for name, row in candidate_rows.items():
        dev = row["summary"][row["selection_score"]]["dev"]
        print(f"| {row['category']} | {name} | {'yes' if row['needs_fault_labels'] else 'no'} | "
              f"{dev['macro_recording_auc_roc']['mean']:.3f} | {dev['recall']['mean']:.3f} | "
              f"{dev['false_positive_rate']['mean']:.3f} | {'yes' if selected[row['category']] == name else ''} |")

    print("\nSelected models on final recordings (mean ± sd over seeds)")
    print(f"| Category | Model | Score | {METRIC_HEADER} |")
    print("|---" * 10 + "|")
    for category, name in selected.items():
        row = candidate_rows[name]
        for score, results in row["summary"].items():
            print(f"| {category} | {name} | {score} | {metric_cells(results['final'])} |")
            if "final_unseen_fault_type" in results:
                print(f"| {category} | {name} | unseen fault type | "
                      f"{metric_cells(results['final_unseen_fault_type'])} |")

    print("\nModel size and training cost")
    for category, name in selected.items():
        row = candidate_rows[name]
        if "parameters" in row:
            print(f"{name}: {row['parameters']:,} parameters, {row['model_bytes']:,} bytes, "
                  f"{megabytes(row['federated_bytes_per_round'])} per federated round, "
                  f"{row['seconds']:.0f}s training")
        else:
            print(f"{name}: trained centrally on pooled feature windows; cannot be averaged with FedAvg")

    return {
        "selected": selected,
        "candidates": candidate_rows,
        "deep_runs": deep_runs,
        "classic_runs": classic_runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    regimes = commands.add_parser("regimes", help="federated vs centralised vs isolated training")
    add_common_arguments(regimes)
    regimes.add_argument("--regimes", nargs="+", choices=REGIMES, default=list(REGIMES))
    regimes.add_argument("--federated-rounds", type=int, nargs="*", default=[5, 10],
                         help="extra federated runs with fewer rounds, for the communication trade-off")
    regimes.add_argument("--mad-multiplier", type=float, default=24.0)
    regimes.add_argument("--skab-path", default="data/SKAB")
    regimes.add_argument("--output", default="artifacts/comparison/regimes.json")

    models = commands.add_parser("models", help="LSTM vs transformer vs classic machine learning")
    add_common_arguments(models)
    models.add_argument("--regime", choices=REGIMES, default="federated",
                        help="how the autoencoders are trained; classic models are always central")
    models.add_argument("--max-false-positive-rate", type=float, default=0.15)
    models.add_argument("--output", default="artifacts/comparison/models.json")

    args = parser.parse_args()
    result = run_regimes(args) if args.command == "regimes" else run_models(args)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"config": vars(args)} | result, indent=2))
    print(f"\nSaved {output}")


if __name__ == "__main__":
    main()
