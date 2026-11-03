#!/usr/bin/env python3
"""Evaluate a saved federated model under configurable SKAB alert policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hydrofed.data.pipeline import (
    load_client_partition,
    load_client_partition_metadata,
)
from hydrofed.data.scaling import WindowStandardScaler, scale_plant_split
from hydrofed.data.splits import split_skab_plant_windows
from hydrofed.diagnostics import recording_diagnostics
from hydrofed.evaluation import calibrate_recording_thresholds, evaluate_skab_scores
from hydrofed.federation.checkpoint import load_global_checkpoint
from hydrofed.models.lstm_autoencoder import build_model
from hydrofed.runtime import resolve_device
from hydrofed.training.local import (
    calibrate_threshold,
    reconstruction_absolute_errors,
)


parser = argparse.ArgumentParser()
parser.add_argument("weights")
parser.add_argument("--partitions", default="data/partitions_skab")
parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
parser.add_argument("--train-size", type=int)
parser.add_argument(
    "--training-policy",
    choices=("auto", "chronological", "normal-only"),
    default="auto",
    help="Reuse the checkpoint policy or explicitly choose its SKAB training selection",
)
parser.add_argument("--threshold-percentile", type=float, default=99.0)
parser.add_argument("--threshold-scale", type=float, default=1.5)
parser.add_argument("--persistence", type=int, default=10)
parser.add_argument(
    "--threshold-scope",
    choices=("plant", "recording"),
    default="plant",
    help="Calibrate one threshold per plant or per SKAB recording",
)
parser.add_argument("--output", help="Optional JSON file for the policy result")
parser.add_argument("--plant", help="Evaluate only this plant, e.g. plant_1")
parser.add_argument(
    "--recording-report",
    help="Save a JSON audit of training labels and test metrics for each recording",
)
args = parser.parse_args()

if args.threshold_scale <= 0:
    parser.error("--threshold-scale must be positive")
if args.persistence <= 0:
    parser.error("--persistence must be positive")

checkpoint = load_global_checkpoint(args.weights)
preprocessing = checkpoint.get("preprocessing", {})
train_size = (
    args.train_size
    if args.train_size is not None
    else int(preprocessing.get("skab_train_size", 400))
)
training_policy = (
    preprocessing.get("skab_training_policy", "chronological")
    if args.training_policy == "auto"
    else args.training_policy
)

if preprocessing.get("scaler") != "shared_skab_train_standard":
    raise SystemExit(
        "Checkpoint was not trained with the SKAB shared standard-scaling protocol"
    )

scaler = WindowStandardScaler(
    means=np.asarray(preprocessing["means"], dtype=np.float32),
    standard_deviations=np.asarray(
        preprocessing["standard_deviations"],
        dtype=np.float32,
    ),
)
device = resolve_device(args.device)
model = build_model(
    checkpoint["n_features"],
    checkpoint.get("model_config"),
)
model.load_state_dict(checkpoint["state_dict"])

results = {}
diagnostics = {}
files = sorted(Path(args.partitions).glob("plant_*.npz"))
if args.plant:
    files = [file for file in files if file.stem == args.plant]
if not files:
    raise SystemExit("No matching plant partitions found")

for file in files:
    windows, window_labels = load_client_partition(file)
    metadata = load_client_partition_metadata(file)
    required = {"recording_ids", "endpoint_labels", "window_start_indices"}
    missing = required - metadata.keys()
    if missing:
        raise SystemExit(f"{file} is missing SKAB metadata: {sorted(missing)}")

    plant_split = split_skab_plant_windows(
        windows,
        metadata["endpoint_labels"],
        metadata["recording_ids"],
        metadata["window_start_indices"],
        train_size=train_size,
        window_labels=window_labels,
        training_policy=training_policy,
    )
    plant_split = scale_plant_split(plant_split, scaler)
    train_scores = reconstruction_absolute_errors(
        model,
        plant_split.train_windows,
        device=device,
    )
    if args.threshold_scope == "recording":
        threshold = calibrate_recording_thresholds(
            train_scores,
            plant_split.train_recording_ids,
            args.threshold_percentile,
            args.threshold_scale,
        )
    else:
        threshold = calibrate_threshold(
            train_scores,
            args.threshold_percentile,
        ) * args.threshold_scale
    test_scores = reconstruction_absolute_errors(
        model,
        plant_split.test_windows,
        device=device,
    )
    results[file.stem] = evaluate_skab_scores(
        test_scores,
        plant_split.test_labels,
        plant_split.test_recording_ids,
        threshold,
        consecutive_sequences=args.persistence,
    )
    if args.recording_report:
        # Match the split's per-recording concatenation order. Counting window
        # labels also catches faults earlier in a sequence than its endpoint.
        ends = metadata["window_start_indices"] + windows.shape[1] - 1
        train_indices = np.concatenate([
            np.flatnonzero(
                (metadata["recording_ids"] == recording_id)
                & (ends < train_size)
                & (
                    (window_labels == 0)
                    if training_policy == "normal-only"
                    else np.ones(len(window_labels), dtype=bool)
                )
            )
            for recording_id in dict.fromkeys(metadata["recording_ids"].tolist())
        ])
        diagnostics[file.stem] = recording_diagnostics(
            train_scores,
            plant_split.train_recording_ids,
            window_labels[train_indices],
            metadata["endpoint_labels"][train_indices],
            test_scores,
            plant_split.test_recording_ids,
            plant_split.test_labels,
            threshold,
            consecutive_sequences=args.persistence,
        )

macro_f1 = float(np.mean([metrics["f1"] for metrics in results.values()]))
auc_values = [
    metrics["auc_roc"]
    for metrics in results.values()
    if metrics["auc_roc"] is not None
]
summary = {
    "policy": {
        "threshold_percentile": args.threshold_percentile,
        "threshold_scale": args.threshold_scale,
        "persistence": args.persistence,
        "threshold_scope": args.threshold_scope,
    },
    "macro_f1": macro_f1,
    "macro_auc_roc": float(np.mean(auc_values)) if auc_values else None,
    "plants": results,
}

print(json.dumps(summary, indent=2))

if args.output:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")

if args.recording_report:
    report = {
        "checkpoint": str(Path(args.weights)),
        "partitions": str(Path(args.partitions)),
        "train_size": train_size,
        "training_policy": training_policy,
        "policy": summary["policy"],
        "notes": [
            "Training labels are audited only; they do not change training or thresholds.",
            "Counts are windows/endpoints, not distinct fault events.",
            "Training window flags use the partition's window-label mode.",
            "Macro recording AUC and average precision exclude single-class recordings.",
            "Compare average precision with the recording's anomaly rate.",
        ],
        "plants": diagnostics,
    }
    output = Path(args.recording_report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved recording diagnostic report: {output}")
