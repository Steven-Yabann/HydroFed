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
    """Download and safely unpack the public SKAB repository archive."""
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
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
    files = sorted(Path(path).rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {path}")
    frames = [pd.read_csv(file, sep=None, engine="python") for file in files]
    return pd.concat(frames, ignore_index=True)

def sensor_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"datetime", "timestamp", "time", *LABEL_NAMES}
    return [c for c in frame.select_dtypes(include=np.number).columns if c.lower() not in excluded]

def prepare_dataframe(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    clean = frame.copy().ffill().bfill()
    columns = sensor_columns(clean)
    if not columns:
        raise ValueError("No numeric sensor columns found")
    mins, spans = clean[columns].min(), clean[columns].max() - clean[columns].min()
    clean[columns] = (clean[columns] - mins) / spans.replace(0, 1)
    return clean, columns

def create_windows(features: np.ndarray, labels: np.ndarray, window_length: int,
                   stride: int = 1, label_mode: str = "any") -> tuple[np.ndarray, np.ndarray]:
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

def load_client_partition(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        return data["windows"].astype(np.float32), data["labels"].astype(np.int64)
