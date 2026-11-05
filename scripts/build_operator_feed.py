"""Replay SKAB test recordings through a saved model for the operator dashboard.

The dashboard has no live plant to read from, so this script scores every
SKAB recording, baseline and test region, with a saved LSTM autoencoder and
the frozen hybrid alert policy, then writes one JSON file per plant. The frontend plays these
back second by second as if they were arriving from the plant.

    .venv/bin/python scripts/build_operator_feed.py

The defaults match the policy frozen on development recordings
(docs/comparison.md): plant scaling, EWMA 0.2, median + 24 MAD on the last
25% of each recording's baseline, alert on the first second over threshold.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from hydrofed.data.pipeline import load_client_partition, load_client_partition_metadata
from hydrofed.experiments.recordings import load_skab_recordings
from hydrofed.experiments.regimes import autoencoder_scores
from hydrofed.federation.checkpoint import load_global_checkpoint
from hydrofed.models.lstm_autoencoder import build_model

# SKAB column order in the partitions, with units for the dashboard.
SENSORS = (
    ("accelerometer_1", "Accelerometer 1 RMS", "g"),
    ("accelerometer_2", "Accelerometer 2 RMS", "g"),
    ("current", "Motor current", "A"),
    ("pressure", "Pressure", "bar"),
    ("temperature", "Motor temperature", "°C"),
    ("thermocouple", "Fluid temperature", "°C"),
    ("voltage", "Voltage", "V"),
    ("flow_rate", "Volume flow rate", "L/min"),
)


def load_model(checkpoint_path: Path):
    checkpoint = load_global_checkpoint(checkpoint_path)
    model = build_model(int(checkpoint["n_features"]), checkpoint.get("model_config"))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def raw_readings(partition: Path, recording_id: str, selected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the unscaled reading at the end of each selected window, with its second.

    ``selected`` is a mask over the recording's windows, such as a
    recording's ``train_mask`` or ``test_mask``.
    """
    windows, _ = load_client_partition(partition)
    metadata = load_client_partition_metadata(partition)
    mask = metadata["recording_ids"] == recording_id
    starts = metadata["window_start_indices"][mask]
    return windows[mask][selected][:, -1, :], starts[selected] + windows.shape[1] - 1


def rounded(values: np.ndarray, digits: int) -> list[float]:
    return [round(float(value), digits) for value in values]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--partitions", default="data/partitions_skab")
    parser.add_argument("--checkpoint", default="artifacts/skab_runs/hybrid_latent4_seed42.pt")
    parser.add_argument("--output", default="artifacts/operator")
    parser.add_argument("--threshold", type=float, default=24.0, help="alert threshold in baseline MAD units")
    parser.add_argument("--smoothing", type=float, default=0.2)
    parser.add_argument("--calibration-tail", type=float, default=0.25)
    parser.add_argument("--train-size", type=int, default=400)
    args = parser.parse_args()

    partitions = Path(args.partitions)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    # As on the SKAB leaderboard, the anomaly-free recording is not tested: it
    # drifts from its own first minutes. Its baseline still joins the replay.
    recordings = load_skab_recordings(partitions, scaling="plant", train_size=args.train_size)
    # Score the baseline too, by treating its windows as test windows. The
    # replay then starts each recording in normal operation, before any fault.
    baselines = {
        plant: [replace(recording, test_mask=recording.train_mask) for recording in items]
        for plant, items in recordings.items()
    }
    model = load_model(Path(args.checkpoint))
    options = {"score": "hybrid", "smoothing": args.smoothing, "calibration_tail": args.calibration_tail}
    scores = autoencoder_scores(recordings, dict.fromkeys(recordings, model), **options)
    baseline_scores = autoencoder_scores(baselines, dict.fromkeys(baselines, model), **options)

    index = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "dataset": "SKAB",
            "checkpoint": args.checkpoint,
            "score": "hybrid",
            "threshold_mad": args.threshold,
            "smoothing": args.smoothing,
            "calibration_tail": args.calibration_tail,
            "baseline_seconds": args.train_size,
        },
        "sensors": [{"key": key, "label": label, "unit": unit} for key, label, unit in SENSORS],
        "plants": [],
    }

    for plant, items in recordings.items():
        plant_recordings = []
        for recording in items:
            parts = []
            for part_scores, selected in ((baseline_scores, recording.train_mask), (scores, recording.test_mask)):
                if not selected.any():
                    continue
                mask = part_scores.recording_ids == recording.recording_id
                readings, seconds = raw_readings(partitions / f"{plant}.npz", recording.recording_id, selected)
                if mask.sum() != len(seconds):
                    raise RuntimeError(f"score and reading counts differ for {recording.recording_id}")
                parts.append((part_scores.scores[mask], part_scores.labels[mask], readings, seconds))
            recording_scores, labels, readings, seconds = (np.concatenate(values) for values in zip(*parts))
            plant_recordings.append({
                "id": recording.recording_id.removeprefix("SKAB-master/data/"),
                "fault_type": recording.fault_type,
                "role": recording.role,
                "baseline_seconds": int(recording.train_mask.sum()),
                "seconds": seconds.astype(int).tolist(),
                "score": rounded(recording_scores, 2),
                "label": labels.astype(int).tolist(),
                "alert": (recording_scores > args.threshold).astype(int).tolist(),
                "sensors": {key: rounded(readings[:, column], 4) for column, (key, _, _) in enumerate(SENSORS)},
            })

        (output / f"{plant}.json").write_text(json.dumps({"plant": plant, "recordings": plant_recordings}))
        alerts = sum(sum(item["alert"]) for item in plant_recordings)
        index["plants"].append({
            "id": plant,
            "recordings": len(plant_recordings),
            "seconds": sum(len(item["seconds"]) for item in plant_recordings),
            "alert_seconds": alerts,
        })
        print(f"{plant}: {len(plant_recordings)} recordings, {alerts} alert seconds")

    (output / "index.json").write_text(json.dumps(index, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
