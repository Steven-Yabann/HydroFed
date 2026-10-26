"""SKAB-compatible data pipeline with an entirely offline synthetic path."""
from __future__ import annotations

import json
import io
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import requests

LABEL_NAMES = ("anomaly", "changepoint")

SKAB_ARCHIVE_URL = "https://github.com/waico/SKAB/archive/refs/heads/master.zip"

def download_skab(destination: str | Path, url: str = SKAB_ARCHIVE_URL) -> Path:
    """Download SKAB when ``destination`` does not already contain CSV files."""
    destination = Path(destination)
    if destination.exists() and any(destination.rglob("*.csv")):
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=60); response.raise_for_status()
    root = destination.resolve()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if root not in target.parents and target != root:
                raise ValueError("Unsafe path in SKAB archive")
        archive.extractall(destination)
    return destination

def load_skab_csvs(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    files = sorted(path.rglob("*.csv"))

    if not files:
        raise FileNotFoundError(f"No CSV files found under {path}")

    frames = []

    for file in files:
        frame = pd.read_csv(file, sep=None, engine="python")
        for label_name in LABEL_NAMES:
            if label_name not in frame:
                frame[label_name] = 0
            else:
                frame[label_name] = frame[label_name].fillna(0).astype(np.int64)

        frame["_recording_id"] = str(file.relative_to(path))
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)

def sensor_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"datetime", "timestamp", "time", *LABEL_NAMES}
    return [c for c in frame.select_dtypes(include=np.number).columns if c.lower() not in excluded]

def prepare_dataframe(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    clean = frame.copy()
    columns = sensor_columns(clean)

    if "_recording_id" in clean:
        clean[columns] = clean.groupby(
            "_recording_id",
            sort=False,
        )[columns].transform(lambda recording: recording.ffill().bfill())
    else:
        clean[columns] = clean[columns].ffill().bfill()

    if not columns:
        raise ValueError("No numeric sensor columns found")
    mins, spans = clean[columns].min(), clean[columns].max() - clean[columns].min()
    clean[columns] = (clean[columns] - mins) / spans.replace(0, 1)
    return clean, columns

def create_windows(features: np.ndarray,
                   labels: np.ndarray,
                   window_length: int,
                   stride: int = 1,
                   label_mode: str = "any") -> tuple[np.ndarray, np.ndarray]:
    if window_length <= 0 or stride <= 0 or len(features) < window_length:
        raise ValueError("window_length/stride must be positive and data must contain a full window")
    if label_mode not in {"any", "majority"}:
        raise ValueError("label_mode must be 'any' or 'majority'")
    starts = range(0, len(features) - window_length + 1, stride)
    windows, targets = [], []
    for start in starts:
        windows.append(features[start:start + window_length])
        chunk = np.asarray(labels[start:start + window_length]).astype(bool)
        targets.append(bool(chunk.any()) if label_mode == "any" else bool(chunk.mean() > .5))
    return np.asarray(windows, dtype=np.float32), np.asarray(targets, dtype=np.int64)

def generate_synthetic_telemetry(n_rows: int = 1500, n_features: int = 8,
                                 anomaly_fraction: float = .08, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n_rows)
    data = {f"sensor_{i+1}": np.sin(t / (16 + i * 3)) + .08*rng.normal(size=n_rows)
            for i in range(n_features)}
    labels = np.zeros(n_rows, dtype=int)
    count = max(1, int(n_rows * anomaly_fraction))
    indices = rng.choice(n_rows, count, replace=False)
    labels[indices] = 1
    for values in data.values():
        values[indices] += rng.normal(3.0, .5, count)
    return pd.DataFrame({"datetime": pd.date_range("2024-01-01", periods=n_rows, freq="min"),
                         **data, "anomaly": labels})

def partition_non_iid(windows: np.ndarray, labels: np.ndarray, n_clients: int = 5,
                      seed: int = 42) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Partition sorted anomaly scores into heterogeneous, non-overlapping clients."""
    if n_clients <= 0 or len(windows) != len(labels):
        raise ValueError("n_clients must be positive and windows/labels must align")
    rng = np.random.default_rng(seed)
    normal, anomalous = np.where(labels == 0)[0], np.where(labels != 0)[0]
    rng.shuffle(normal); rng.shuffle(anomalous)
    normal_chunks = np.array_split(normal, n_clients)
    # Rotating anomaly allocation creates intentionally unequal local prevalence.
    weights = np.arange(1, n_clients + 1, dtype=float)
    counts = np.floor(len(anomalous) * weights / weights.sum()).astype(int)
    counts[-1] += len(anomalous) - int(counts.sum())
    anomaly_chunks, cursor = [], 0
    for count in counts:
        anomaly_chunks.append(anomalous[cursor:cursor + count]); cursor += count
    result = {}
    for i in range(n_clients):
        take = np.concatenate((normal_chunks[i], anomaly_chunks[i]))
        rng.shuffle(take)
        result[f"plant_{i+1}"] = (windows[take], labels[take])
    return result

def save_partitions(partitions: dict[str, tuple[np.ndarray, np.ndarray]], output: str | Path) -> Path:
    directory = Path(output); directory.mkdir(parents=True, exist_ok=True)
    summary = {}
    for client_id, (windows, labels) in partitions.items():
        np.savez_compressed(directory / f"{client_id}.npz", windows=windows, labels=labels)
        summary[client_id] = {"windows": int(len(windows)), "anomalies": int(labels.sum()),
                              "anomaly_rate": float(labels.mean()) if len(labels) else 0.0}
    path = directory / "partition_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    return path

def partition_temporal(
    windows: np.ndarray,
    labels: np.ndarray,
    n_clients: int = 5,
    minimum_anomalies: int = 0,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Assign each plant an ordered time segment with a minimum anomaly count."""
    if n_clients <= 0:
        raise ValueError("n_clients must be positive")

    if len(windows) != len(labels):
        raise ValueError("windows and labels must have the same length")

    if minimum_anomalies < 0:
        raise ValueError("minimum_anomalies cannot be negative")

    total_anomalies = int(labels.sum())
    required_anomalies = n_clients * minimum_anomalies

    if total_anomalies < required_anomalies:
        raise ValueError(
            f"Cannot give {n_clients} plants at least {minimum_anomalies} "
            f"anomalies each: only {total_anomalies} anomalous windows exist"
        )

    # Keep time order, but move a boundary right when its plant would otherwise
    # receive too few anomalies. Reserve enough anomalies for later plants.
    boundaries = [0]
    start = 0

    for plant_index in range(1, n_clients):
        remaining_plants = n_clients - plant_index
        ideal_end = (
            len(windows) * plant_index + n_clients - 1
        ) // n_clients

        valid_ends = []
        for end in range(start + 1, len(windows)):
            current_anomalies = int(labels[start:end].sum())
            remaining_anomalies = int(labels[end:].sum())

            if (
                current_anomalies >= minimum_anomalies
                and remaining_anomalies >= remaining_plants * minimum_anomalies
            ):
                valid_ends.append(end)

        if not valid_ends:
            raise ValueError(
                "Cannot create contiguous temporal partitions that satisfy "
                "the requested minimum anomaly count"
            )

        end = min(valid_ends, key=lambda candidate: abs(candidate - ideal_end))
        boundaries.append(end)
        start = end

    boundaries.append(len(windows))

    return {
        f"plant_{index + 1}": (
            windows[boundaries[index]:boundaries[index + 1]],
            labels[boundaries[index]:boundaries[index + 1]],
        )
        for index in range(n_clients)
    }

def partition_temporal_by_recording(
    windows: np.ndarray,
    labels: np.ndarray,
    recording_ids: np.ndarray,
    n_clients: int = 5,
    minimum_anomalies: int = 0,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Assign contiguous groups of whole recordings to simulated plants."""
    if n_clients <= 0:
        raise ValueError("n_clients must be positive")

    if len(windows) != len(labels) or len(windows) != len(recording_ids):
        raise ValueError(
            "windows, labels, and recording_ids must have the same length"
        )

    if minimum_anomalies < 0:
        raise ValueError("minimum_anomalies cannot be negative")

    recording_starts = np.flatnonzero(
        np.r_[True, recording_ids[1:] != recording_ids[:-1]]
    )
    recording_ends = np.r_[recording_starts[1:], len(recording_ids)]

    if len(recording_starts) < n_clients:
        raise ValueError("Need at least one recording per client")

    if len(pd.unique(recording_ids)) != len(recording_starts):
        raise ValueError("recording_ids must be grouped by source recording")

    total_anomalies = int(labels.sum())
    required_anomalies = n_clients * minimum_anomalies

    if total_anomalies < required_anomalies:
        raise ValueError(
            f"Cannot give {n_clients} plants at least {minimum_anomalies} "
            f"anomalies each: only {total_anomalies} anomalous windows exist"
        )

    boundaries = [0]
    start_recording = 0

    for plant_index in range(1, n_clients):
        remaining_plants = n_clients - plant_index
        ideal_end = len(windows) * plant_index / n_clients
        valid_ends = []

        for end_recording in range(
            start_recording + 1,
            len(recording_ends) - remaining_plants + 1,
        ):
            end = recording_ends[end_recording - 1]
            current_anomalies = int(labels[boundaries[-1]:end].sum())
            remaining_anomalies = int(labels[end:].sum())

            if (
                current_anomalies >= minimum_anomalies
                and remaining_anomalies >= remaining_plants * minimum_anomalies
            ):
                valid_ends.append((end, end_recording))

        if not valid_ends:
            raise ValueError(
                "Cannot partition whole recordings while satisfying the "
                "requested minimum anomaly count"
            )

        end, start_recording = min(
            valid_ends,
            key=lambda candidate: abs(candidate[0] - ideal_end),
        )
        boundaries.append(end)

    boundaries.append(len(windows))

    return {
        f"plant_{index + 1}": (
            windows[boundaries[index]:boundaries[index + 1]],
            labels[boundaries[index]:boundaries[index + 1]],
        )
        for index in range(n_clients)
    }

def create_windows_with_recording_ids(
    features: np.ndarray,
    labels: np.ndarray,
    recording_ids: np.ndarray,
    window_length: int,
    stride: int = 1,
    label_mode: str = "any",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create windows within recordings and retain each window's source ID."""
    if len(features) != len(labels) or len(features) != len(recording_ids):
        raise ValueError(
            "features, labels, and recording_ids must have the same length"
        )

    window_sets = []
    label_sets = []
    window_recording_sets = []

    for recording_id in pd.unique(recording_ids):
        mask = recording_ids == recording_id
        recording_features = features[mask]
        recording_labels = labels[mask]

        if len(recording_features) < window_length:
            continue

        windows, window_labels = create_windows(
            recording_features,
            recording_labels,
            window_length,
            stride,
            label_mode,
        )

        window_sets.append(windows)
        label_sets.append(window_labels)
        window_recording_sets.append(
            np.full(len(windows), recording_id, dtype=object)
        )

    if not window_sets:
        raise ValueError("No recording contains enough rows for one full window")

    return (
        np.concatenate(window_sets),
        np.concatenate(label_sets),
        np.concatenate(window_recording_sets),
    )


def create_windows_by_recording(
    features: np.ndarray,
    labels: np.ndarray,
    recording_ids: np.ndarray,
    window_length: int,
    stride: int = 1,
    label_mode: str = "any",
) -> tuple[np.ndarray, np.ndarray]:
    """Create windows separately within each source recording."""
    windows, window_labels, _ = create_windows_with_recording_ids(
        features,
        labels,
        recording_ids,
        window_length,
        stride,
        label_mode,
    )
    return windows, window_labels

def load_client_partition(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        return data["windows"].astype(np.float32), data["labels"].astype(np.int64)
