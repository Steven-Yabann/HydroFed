#!/usr/bin/env python3
"""Run HydroFed's SKAB comparison experiments.

regimes
    Train the LSTM autoencoder federated, centralised and isolated with the
    same inputs, schedule and alert policy, then report detection metrics on
    development and final recordings and the communication each regime needs.

models
    Compare the LSTM autoencoder with transformer autoencoders and classic
    machine learning (Random Forest, XGBoost, Isolation Forest, and XGBoost
    trained federatedly with Flower's bagging aggregation). The best candidate
    in each category is chosen on development recordings, and only the chosen
    models are reported on final recordings.

scenarios
    Rerun the regime comparison where federated learning should matter:
    plants whose sensors differ, and plants with little training data. An
    unchanged baseline runs alongside for reference. Alert settings stay frozen.

feedback
    Measure what alert reviews could teach the detector, on development
    recordings only: re-baselining after a resolved fault, a per-plant
    threshold learned from false alarms, and reviews as XGBoost labels.
    Earlier recordings supply the reviews; later ones are evaluated.

publish
    Save existing comparison results to an experiments database so the
    dashboard can show them. ``regimes`` and ``models`` can also save directly
    with ``--db``.

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

from hydrofed.experiments.feedback import (
    alert_episodes,
    learned_threshold,
    rebaselined_scores,
    resolution_window,
    review_labels,
    split_by_time,
)
from hydrofed.experiments.communication import (
    break_even_days,
    federated_bytes,
    raw_skab_bytes,
    streaming_bytes_per_day,
)
from hydrofed.experiments.publish import (
    MODEL_EXPERIMENT,
    REGIME_EXPERIMENT,
    model_rows,
    publish,
    regime_rows,
)
from hydrofed.experiments.recordings import ROLES, load_skab_recordings, plant_training_windows
from hydrofed.experiments.regimes import ALL_REGIMES, REGIMES, SCORES, autoencoder_scores, parameter_bytes, train_regime
from hydrofed.experiments.scenarios import HETEROGENEOUS_PROFILES, SCARCE_PLANTS, SCARCE_SECONDS
from hydrofed.experiments.scoring import (
    MAD_THRESHOLDS,
    RecordingScores,
    alert_metrics,
    choose_threshold,
    summarise_seeds,
)
from hydrofed.models.lstm_autoencoder import build_model
from hydrofed.models.transformer_autoencoder import build_transformer_model
from hydrofed.storage import ExperimentStore
from hydrofed.training.local import reconstruction_signed_residuals

ARCHITECTURES = {"lstm": build_model, "transformer": build_transformer_model}
CLASSIC_SUPERVISED = {"random_forest", "xgboost", "federated_xgboost"}


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
    scenario = job.get("scenario", {})
    recordings = load_skab_recordings(
        args.partitions, args.scaling,
        profiles=HETEROGENEOUS_PROFILES if scenario.get("heterogeneous") else None,
        training_seconds=scenario.get("training_seconds"),
    )
    plant_windows = plant_training_windows(recordings)
    sequence_seconds = {
        recording.recording_id: recording.step_seconds for items in recordings.values() for recording in items
    }
    plant_of = {
        recording.recording_id: plant for plant, items in recordings.items() for recording in items
    }
    n_features = next(iter(plant_windows.values())).shape[2]
    build = ARCHITECTURES[job["architecture"]]
    result = train_regime(
        job["regime"], plant_windows, lambda: build(n_features, job["config"]),
        rounds=job["rounds"], epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, seed=job["seed"], device=args.device,
    )
    metrics, thresholds, group_metrics = {}, {}, {}
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
            role: alert_metrics(scores.select(role), thresholds[score], args.persistence, sequence_seconds)
            for role in ROLES
        }
        if job.get("groups"):
            plants = np.array([plant_of[recording_id] for recording_id in scores.recording_ids])
            group_metrics[score] = {
                group: {
                    role: alert_metrics(
                        scores.where(np.isin(plants, members) & (scores.roles == role)),
                        thresholds[score], args.persistence, sequence_seconds,
                    )
                    for role in ROLES
                }
                for group, members in job["groups"].items()
            }
    print(f"{job['label']} seed {job['seed']}: {result.seconds:.0f}s", flush=True)
    model = next(iter(result.models.values()))
    return {
        "label": job["label"], "seed": job["seed"], "metrics": metrics, "thresholds": thresholds,
        "parameters": model.count_trainable_parameters(),
        "model_bytes": parameter_bytes(model.get_parameters_numpy()),
        "uplink_bytes": result.uplink_bytes, "downlink_bytes": result.downlink_bytes,
        "seconds": result.seconds,
        "group_metrics": group_metrics,
        "training_windows": {plant: len(windows) for plant, windows in plant_windows.items()},
    }


def run_jobs(jobs: list[dict], n_jobs: int, worker=None) -> list[dict]:
    worker = worker or autoencoder_job
    threads = max(1, (os.cpu_count() or 2) // max(1, n_jobs))
    jobs = [job | {"threads": threads} for job in jobs]
    if n_jobs <= 1:
        return [worker(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        return list(pool.map(worker, jobs))


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
        if labels[run["label"]][0] in ("federated", "fedprox", "federated_finetune"):
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
            federated = args.regime == "federated"
            candidate_rows[name] = {
                "category": category,
                "training": "FedAvg" if federated else args.regime,
                "needs_fault_labels": False,
                "parameters": runs[0]["parameters"],
                "model_bytes": runs[0]["model_bytes"],
                "federated_bytes_per_round": 2 * runs[0]["model_bytes"] * clients if federated else 0,
                "communication_bytes": 2 * runs[0]["model_bytes"] * clients * args.rounds if federated else 0,
                "rounds": args.rounds,
                "seconds": float(np.mean([run["seconds"] for run in runs])),
                "summary": summarise(deep_runs, name),
                "selection_score": "residual",
            }
    for name, runs in classic_runs.items():
        roles = {role: summarise_seeds([run[role] for run in runs]) for role in ROLES}
        if name in CLASSIC_SUPERVISED:
            roles["final_unseen_fault_type"] = summarise_seeds([run["final_unseen_fault_type"] for run in runs])
        federated = name == "federated_xgboost"
        candidate_rows[name] = {
            "category": "federated_classic" if federated else "classic",
            "training": "FedXgbBagging" if federated else "centralised",
            "needs_fault_labels": name in CLASSIC_SUPERVISED,
            "summary": {"model": roles},
            "selection_score": "model",
        }
        if federated:
            costs = [run["training"] for run in runs]
            candidate_rows[name] |= {
                "clients": costs[0]["clients"],
                "rounds": costs[0]["rounds"],
                "trees": costs[0]["trees"],
                "model_bytes": float(np.mean([cost["model_bytes"] for cost in costs])),
                "communication_bytes": float(np.mean([
                    cost["uplink_bytes"] + cost["downlink_bytes"] for cost in costs
                ])),
                "uplink_bytes": float(np.mean([cost["uplink_bytes"] for cost in costs])),
                "downlink_bytes": float(np.mean([cost["downlink_bytes"] for cost in costs])),
            }

    def dev_auc(name):
        row = candidate_rows[name]
        return row["summary"][row["selection_score"]]["dev"]["macro_recording_auc_roc"]["mean"]

    selected = {}
    for category in ("recurrent", "transformer", "classic", "federated_classic"):
        names = [name for name, row in candidate_rows.items() if row["category"] == category]
        if names:
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
        elif row["training"] == "FedXgbBagging":
            print(f"{name}: {row['trees']} trees from {row['clients']} plants over {row['rounds']} rounds; "
                  f"{megabytes(row['uplink_bytes'])} up, {megabytes(row['downlink_bytes'])} down")
        else:
            print(f"{name}: trained centrally on pooled feature windows")

    return {
        "selected": selected,
        "candidates": candidate_rows,
        "deep_runs": deep_runs,
        "classic_runs": classic_runs,
    }


def scenario_specs(args) -> dict[str, dict]:
    """Return each scenario's data changes, keyed by scenario name."""
    specs = {}
    for name in args.scenarios:
        if name == "baseline":
            specs[name] = {}
        elif name == "heterogeneous":
            specs[name] = {"heterogeneous": True}
        else:
            for seconds in args.scarce_seconds:
                specs[f"scarce_{seconds}s"] = {
                    "training_seconds": dict.fromkeys(args.scarce_plants, seconds),
                }
    return specs


def print_scenario_tables(summary: dict, groups: dict[str, list[str]]) -> None:
    def cells(metrics):
        def value(name):
            mean = metrics[name]["mean"]
            return f"{mean:.3f}" if mean is not None else "n/a"
        return f"{value('recall')} / {value('false_positive_rate')} / {value('macro_recording_auc_roc')}"

    plant_groups = [group for group in groups if group.startswith("plant_")]
    for scenario, regimes in summary.items():
        for score in SCORES:
            print(f"\n{scenario}, {score} score, final recordings: recall / false alarm rate / AUC")
            print("| Regime | All plants | Scarce plants | Other plants |")
            print("|---|---|---|---|")
            for regime, scores in regimes.items():
                group = scores["groups"][score]
                print(f"| {regime} | {cells(scores[score]['final'])} | "
                      f"{cells(group['scarce plants']['final'])} | {cells(group['other plants']['final'])} |")
            print(f"\n{scenario}, {score} score, final recordings, within-recording AUC per plant")
            print("| Regime | " + " | ".join(plant_groups) + " |")
            print("|---" * (len(plant_groups) + 1) + "|")
            for regime, scores in regimes.items():
                values = [scores["groups"][score][plant]["final"]["macro_recording_auc_roc"]["mean"] for plant in plant_groups]
                print(f"| {regime} | " + " | ".join(f"{v:.3f}" if v is not None else "n/a" for v in values) + " |")


def run_scenarios(args) -> dict:
    specs = scenario_specs(args)
    plants = sorted(path.stem for path in Path(args.partitions).glob("plant_*.npz"))
    groups = {
        "scarce plants": list(args.scarce_plants),
        "other plants": [plant for plant in plants if plant not in args.scarce_plants],
        **{plant: [plant] for plant in plants},
    }
    jobs = [
        {
            "args": args, "label": f"{scenario}/{regime}", "regime": regime, "rounds": args.rounds,
            "seed": seed, "architecture": "lstm", "config": lstm_config(args),
            "threshold": args.mad_multiplier, "scenario": spec, "groups": groups,
        }
        for scenario, spec in specs.items()
        for regime in args.regimes
        for seed in args.seeds
    ]
    runs = run_jobs(jobs, args.jobs)

    summary = {}
    for scenario in specs:
        summary[scenario] = {}
        for regime in args.regimes:
            selected = [run for run in runs if run["label"] == f"{scenario}/{regime}"]
            summary[scenario][regime] = summarise(runs, f"{scenario}/{regime}") | {
                "groups": {
                    score: {
                        group: {
                            role: summarise_seeds([run["group_metrics"][score][group][role] for run in selected])
                            for role in ROLES
                        }
                        for group in groups
                    }
                    for score in SCORES
                },
                "training_windows": selected[0]["training_windows"],
            }
    print_scenario_tables(summary, groups)
    return {
        "scenarios": specs,
        "profiles": {plant: vars(profile) for plant, profile in HETEROGENEOUS_PROFILES.items()},
        "groups": groups,
        "summary": summary,
        "runs": runs,
    }


FEEDBACK_METHODS = {
    "detector": "Current detector",
    "relearn": "Re-learn normal after a resolved fault",
    "relearn_shelved": "Re-learn, alarms shelved while re-learning",
    "relearn_early": "Re-learn, resolved too early",
    "recentre": "Re-centre normal after a resolved fault",
    "recentre_shelved": "Re-centre, alarms shelved while re-learning",
    "recentre_early": "Re-centre, resolved too early",
    "threshold": "Per-plant threshold from false alarms",
}


def after_fault_share(scores: RecordingScores, thresholds) -> float | None:
    """Share of false-alarm seconds that come after each recording's fault ends."""
    after = total = 0
    for recording_id in dict.fromkeys(scores.recording_ids.tolist()):
        mask = scores.recording_ids == recording_id
        threshold = thresholds[recording_id] if isinstance(thresholds, dict) else thresholds
        alarms = (scores.scores[mask] > threshold) & (scores.labels[mask] == 0)
        fault = np.flatnonzero(scores.labels[mask])
        total += int(alarms.sum())
        if len(fault):
            after += int(alarms[fault[-1] + 1:].sum())
    return after / total if total else None


def feedback_job(job: dict) -> dict:
    """Train one seed's federated LSTM and apply every review method to it."""
    args = job["args"]
    torch.set_num_threads(job["threads"])
    recordings = load_skab_recordings(args.partitions, args.scaling)
    plant_windows = plant_training_windows(recordings)
    n_features = next(iter(plant_windows.values())).shape[2]
    result = train_regime(
        "federated", plant_windows, lambda: build_model(n_features, lstm_config(args)),
        rounds=args.rounds, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, seed=job["seed"], device=args.device,
    )
    model = next(iter(result.models.values()))
    past, future = split_by_time(recordings)
    threshold = args.mad_multiplier

    def components(windows):
        return [windows.mean(axis=1), reconstruction_signed_residuals(model, windows, device=args.device)]

    scored = {}
    for recording in past + future:
        baseline, test = components(recording.train_windows), components(recording.test_windows)
        labels = recording.test_labels
        original = rebaselined_scores(baseline, test, None, args.smoothing, args.calibration_tail)
        careful = resolution_window(labels, original, threshold)
        early = resolution_window(labels, original, threshold, early=True)
        scored[recording.recording_id] = {"plant": recording.plant, "labels": labels, "detector": original}
        for mode in ("relearn", "recentre"):
            for suffix, resolution, shelve in (("", careful, False), ("_shelved", careful, True), ("_early", early, False)):
                scored[recording.recording_id][mode + suffix] = rebaselined_scores(
                    baseline, test, resolution, args.smoothing, args.calibration_tail, shelve=shelve, mode=mode,
                )

    # Each plant's threshold learned from reviews of its earlier recordings.
    peaks: dict[str, dict[str, list[float]]] = {}
    for recording in past:
        item = scored[recording.recording_id]
        for episode in alert_episodes(item["detector"], item["labels"], threshold):
            kind = "fault" if episode.real_fault else "false_alarm"
            peaks.setdefault(recording.plant, {"fault": [], "false_alarm": []})[kind].append(episode.peak)
    plant_thresholds = {
        plant: learned_threshold(found["fault"], found["false_alarm"], threshold) for plant, found in peaks.items()
    }

    def future_scores(method):
        return RecordingScores(
            scores=np.concatenate([scored[r.recording_id][method] for r in future]),
            labels=np.concatenate([scored[r.recording_id]["labels"] for r in future]),
            recording_ids=np.concatenate([np.repeat(r.recording_id, r.test_mask.sum()) for r in future]),
            roles=np.concatenate([np.repeat(r.role, r.test_mask.sum()) for r in future]),
        )

    per_recording = {r.recording_id: plant_thresholds.get(r.plant, threshold) for r in future}
    settings = {name: (name, threshold) for name in FEEDBACK_METHODS} | {"threshold": ("detector", per_recording)}
    metrics = {}
    for name, (method, limit) in settings.items():
        scores = future_scores(method)
        metrics[name] = alert_metrics(scores, limit, args.persistence) | {
            "after_fault_false_alarm_share": after_fault_share(scores, limit),
        }
    print(f"feedback seed {job['seed']}: {result.seconds:.0f}s, thresholds {plant_thresholds}", flush=True)
    return {
        "seed": job["seed"],
        "metrics": metrics,
        "plant_thresholds": plant_thresholds,
        "review_labels": {
            r.recording_id: review_labels(scored[r.recording_id]["detector"], scored[r.recording_id]["labels"],
                                          threshold).tolist()
            for r in past
        },
        "recordings": {"past": [r.recording_id for r in past], "future": [r.recording_id for r in future]},
    }


def run_feedback(args) -> dict:
    jobs = [{"args": args, "seed": seed} for seed in args.seeds]
    runs = run_jobs(jobs, args.jobs, feedback_job)

    # XGBoost cannot share a process with PyTorch on macOS.
    with tempfile.TemporaryDirectory() as folder:
        labels_path, output = Path(folder) / "labels.json", Path(folder) / "xgboost.json"
        labels_path.write_text(json.dumps({str(run["seed"]): run["review_labels"] for run in runs}))
        subprocess.run([
            sys.executable, "-m", "hydrofed.experiments.feedback", "--partitions", args.partitions,
            "--review-labels", str(labels_path), "--output", str(output),
        ], check=True)
        trees = json.loads(output.read_text())
    for run in runs:
        for name, metrics in trees[str(run["seed"])].items():
            run["metrics"][f"xgboost_{name}"] = metrics

    names = {
        **FEEDBACK_METHODS,
        "xgboost_review_labels": "Federated XGBoost, labels from reviews",
        "xgboost_true_labels": "Federated XGBoost, true labels (reference)",
    }
    summary = {name: summarise_seeds([run["metrics"][name] for run in runs]) for name in names}
    print("\nLater development recordings (mean over seeds)")
    print(f"| Method | {METRIC_HEADER} | False alarms after the fault |")
    print("|---" * 10 + "|")
    for name, label in names.items():
        share = summary[name].get("after_fault_false_alarm_share", {}).get("mean")
        print(f"| {label} | {metric_cells(summary[name])} | {f'{share:.0%}' if share is not None else 'n/a'} |")
    print("\nLearned plant thresholds:", [run["plant_thresholds"] for run in runs])
    for run in runs:
        run.pop("review_labels")
    return {"summary": summary, "runs": runs, "methods": names}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    regimes = commands.add_parser("regimes", help="federated vs centralised vs isolated training")
    add_common_arguments(regimes)
    regimes.add_argument("--regimes", nargs="+", choices=ALL_REGIMES, default=list(REGIMES))
    regimes.add_argument("--federated-rounds", type=int, nargs="*", default=[5, 10],
                         help="extra federated runs with fewer rounds, for the communication trade-off")
    regimes.add_argument("--mad-multiplier", type=float, default=24.0)
    regimes.add_argument("--skab-path", default="data/SKAB")
    regimes.add_argument("--output", default="artifacts/comparison/regimes.json")

    models = commands.add_parser("models", help="LSTM vs transformer vs classic machine learning")
    add_common_arguments(models)
    models.add_argument("--regime", choices=REGIMES, default="federated",
                        help="how the autoencoders are trained; classic models are central, except federated XGBoost")
    models.add_argument("--max-false-positive-rate", type=float, default=0.15)
    models.add_argument("--output", default="artifacts/comparison/models.json")

    for command in (regimes, models):
        command.add_argument("--db", help="also save the summary to this experiments database")

    scenarios = commands.add_parser("scenarios", help="heterogeneous and data-scarce plants")
    add_common_arguments(scenarios)
    scenarios.add_argument("--scenarios", nargs="+", choices=("baseline", "heterogeneous", "scarce"),
                           default=["baseline", "heterogeneous", "scarce"])
    scenarios.add_argument("--regimes", nargs="+", choices=ALL_REGIMES, default=list(REGIMES))
    scenarios.add_argument("--scarce-plants", nargs="+", default=list(SCARCE_PLANTS))
    scenarios.add_argument("--scarce-seconds", type=int, nargs="+", default=list(SCARCE_SECONDS))
    scenarios.add_argument("--mad-multiplier", type=float, default=24.0)
    scenarios.add_argument("--output", default="artifacts/comparison/scenarios.json")

    feedback = commands.add_parser("feedback", help="what alert reviews could teach the detector")
    add_common_arguments(feedback)
    feedback.add_argument("--mad-multiplier", type=float, default=24.0)
    feedback.add_argument("--output", default="artifacts/comparison/feedback.json")

    publisher = commands.add_parser("publish", help="save existing JSON results for the dashboard")
    publisher.add_argument("--regimes", help="result of the regimes command")
    publisher.add_argument("--models", help="result of the models command")
    publisher.add_argument("--db", default="artifacts/experiments.db")

    args = parser.parse_args()
    if args.command == "publish":
        sources = {REGIME_EXPERIMENT: (args.regimes, regime_rows), MODEL_EXPERIMENT: (args.models, model_rows)}
        if not args.regimes and not args.models:
            parser.error("publish needs --regimes, --models or both")
        for experiment, (path, rows_for) in sources.items():
            if path:
                save_result(args.db, experiment, json.loads(Path(path).read_text()), rows_for)
        return

    commands_run = {
        "regimes": run_regimes, "models": run_models, "scenarios": run_scenarios, "feedback": run_feedback,
    }
    result = commands_run[args.command](args)
    result = {"config": vars(args)} | result

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(f"\nSaved {output}")
    if getattr(args, "db", None):
        if args.command == "regimes":
            save_result(args.db, REGIME_EXPERIMENT, result, regime_rows)
        else:
            save_result(args.db, MODEL_EXPERIMENT, result, model_rows)


def save_result(db_path: str, experiment: str, result: dict, rows_for) -> None:
    rows = rows_for(result)
    run_id = publish(ExperimentStore(db_path), experiment, result["config"], rows)
    print(f"Saved {len(rows)} {experiment} rows to {db_path} (run {run_id})")


if __name__ == "__main__":
    main()
