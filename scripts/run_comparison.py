#!/usr/bin/env python3
"""Run HydroFed's SKAB comparison experiments.

regimes
    Train the LSTM autoencoder federated, centralised and isolated with the
    same inputs, schedule and alert policy, then report detection metrics on
    development and final recordings and the communication each regime needs.

Settings were chosen on development recordings. Final recordings are for
reporting only: do not tune anything on them.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

from hydrofed.experiments.communication import (
    break_even_days,
    federated_bytes,
    raw_skab_bytes,
    streaming_bytes_per_day,
)
from hydrofed.experiments.recordings import ROLES, load_skab_recordings, plant_training_windows
from hydrofed.experiments.regimes import REGIMES, parameter_bytes, train_regime
from hydrofed.experiments.scoring import SCORES, alert_metrics, autoencoder_scores, summarise_seeds
from hydrofed.models.lstm_autoencoder import build_model


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


def regime_job(job: dict) -> dict:
    """Train and score one regime for one seed. Runs in a worker process."""
    args = job["args"]
    torch.set_num_threads(job["threads"])
    recordings = load_skab_recordings(args.partitions, args.scaling)
    plant_windows = plant_training_windows(recordings)
    n_features = next(iter(plant_windows.values())).shape[2]
    result = train_regime(
        job["regime"], plant_windows, lambda: build_model(n_features, lstm_config(args)),
        rounds=job["rounds"], epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, seed=job["seed"], device=args.device,
    )
    metrics = {}
    for score in SCORES:
        scores = autoencoder_scores(
            recordings, result.models, score, args.smoothing, args.calibration_tail, args.device,
        )
        metrics[score] = {
            role: alert_metrics(scores.select(role), args.mad_multiplier, args.persistence)
            for role in ROLES
        }
    print(f"{job['label']} seed {job['seed']}: {result.seconds:.0f}s", flush=True)
    return {
        "label": job["label"], "seed": job["seed"], "metrics": metrics,
        "uplink_bytes": result.uplink_bytes, "downlink_bytes": result.downlink_bytes,
        "seconds": result.seconds,
    }


def run_jobs(jobs: list[dict], n_jobs: int) -> list[dict]:
    threads = max(1, (os.cpu_count() or 2) // max(1, n_jobs))
    jobs = [job | {"threads": threads} for job in jobs]
    if n_jobs <= 1:
        return [regime_job(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        return list(pool.map(regime_job, jobs))


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


def print_regime_tables(summary: dict, communication: dict) -> None:
    for score in SCORES:
        for role in ROLES:
            print(f"\n{score} score, {role} recordings (mean ± sd over seeds)")
            print("| Regime | Recall | Precision | F1 | False alarm rate | Within-recording AUC | Events | Delay (s) |")
            print("|---|---|---|---|---|---|---|---|")
            for label, scores in summary.items():
                m = scores[score][role]
                cell = lambda name: f"{m[name]['mean']:.3f} ± {m[name]['sd']:.3f}" if m[name]["mean"] is not None else "n/a"
                events = f"{m['fault_events_detected']['mean']:.1f}/{m['fault_events']['mean']:.0f}"
                print(f"| {label} | {cell('recall')} | {cell('precision')} | {cell('f1')} | "
                      f"{cell('false_positive_rate')} | {cell('macro_recording_auc_roc')} | {events} | "
                      f"{cell('median_detection_delay')} |")

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
        {"args": args, "label": label, "regime": regime, "rounds": rounds, "seed": seed}
        for label, (regime, rounds) in labels.items()
        for seed in args.seeds
    ]
    runs = run_jobs(jobs, args.jobs)

    recordings = load_skab_recordings(args.partitions, args.scaling)
    n_features = next(iter(recordings.values()))[0].windows.shape[2]
    model_bytes = parameter_bytes(build_model(n_features, lstm_config(args)).get_parameters_numpy())
    communication = communication_summary(
        args, model_bytes, len(recordings), sorted({rounds for _, rounds in labels.values()}),
    )
    for run in runs:
        if labels[run["label"]][0] == "federated":
            expected = federated_bytes(model_bytes, len(recordings), labels[run["label"]][1])
            if run["uplink_bytes"] + run["downlink_bytes"] != expected["total_bytes"]:
                raise RuntimeError(f"Measured bytes for {run['label']} do not match the FedAvg formula")

    summary = {
        label: {
            score: {
                role: summarise_seeds([run["metrics"][score][role] for run in runs if run["label"] == label])
                for role in ROLES
            }
            for score in SCORES
        }
        for label in labels
    }
    print_regime_tables(summary, communication)
    return {"runs": runs, "summary": summary, "communication": communication}


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

    args = parser.parse_args()
    result = run_regimes(args)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    config = {key: value for key, value in vars(args).items()}
    output.write_text(json.dumps({"config": config} | result, indent=2))
    print(f"\nSaved {output}")


if __name__ == "__main__":
    main()
