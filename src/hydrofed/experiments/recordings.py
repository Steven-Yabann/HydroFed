"""Load SKAB plant partitions as scaled recordings with a dev/final split.

Comparison experiments choose settings on development recordings and report
them on final recordings. The roles alternate within each plant, so both sets
cover every plant and every fault type in similar proportions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hydrofed.data.pipeline import load_client_partition, load_client_partition_metadata
from hydrofed.data.scaling import WindowStandardScaler

ROLES = ("dev", "final")


@dataclass(frozen=True)
class SkabRecording:
    """One SKAB recording, scaled and split SKAB-style into baseline and test."""

    plant: str
    recording_id: str
    role: str
    windows: np.ndarray
    endpoint_labels: np.ndarray
    window_labels: np.ndarray
    train_mask: np.ndarray
    test_mask: np.ndarray

    @property
    def fault_type(self) -> str:
        """Return the SKAB folder name, such as ``valve1`` or ``other``."""
        return Path(self.recording_id).parent.name

    @property
    def train_windows(self) -> np.ndarray:
        return self.windows[self.train_mask]

    @property
    def test_windows(self) -> np.ndarray:
        return self.windows[self.test_mask]

    @property
    def test_labels(self) -> np.ndarray:
        return self.endpoint_labels[self.test_mask]


def recording_roles(recording_ids: np.ndarray) -> dict[str, str]:
    """Alternate development and final roles in recording order."""
    return {
        str(recording_id): ROLES[index % 2]
        for index, recording_id in enumerate(dict.fromkeys(np.asarray(recording_ids).tolist()))
    }


def load_skab_recordings(
    partitions: str | Path,
    scaling: str = "plant",
    train_size: int = 400,
    exclude_test_recordings: tuple[str, ...] = ("anomaly-free",),
) -> dict[str, list[SkabRecording]]:
    """Load every plant partition as a list of recordings.

    Baseline windows end inside a recording's first ``train_size`` points and
    contain no anomalous labels. Test windows start at or after that boundary.
    Scalers are fitted on baseline windows only, per plant or across plants.
    Excluded recordings keep their baseline for training but are not tested.
    """
    if scaling not in {"plant", "shared"}:
        raise ValueError("scaling must be 'plant' or 'shared'")

    files = sorted(Path(partitions).glob("plant_*.npz"))
    if not files:
        raise FileNotFoundError(f"No plant partitions found under {partitions}")

    raw: dict[str, list[dict]] = {}
    for file in files:
        windows, window_labels = load_client_partition(file)
        metadata = load_client_partition_metadata(file)
        recording_ids = metadata["recording_ids"]
        roles = recording_roles(recording_ids)
        recordings = []

        for recording_id, role in roles.items():
            mask = recording_ids == recording_id
            starts = metadata["window_start_indices"][mask]
            ends = starts + windows.shape[1] - 1
            excluded = any(name in recording_id for name in exclude_test_recordings)
            recordings.append({
                "plant": file.stem,
                "recording_id": recording_id,
                "role": role,
                "windows": windows[mask],
                "endpoint_labels": metadata["endpoint_labels"][mask].astype(np.int64),
                "window_labels": window_labels[mask],
                "train_mask": (ends < train_size) & (window_labels[mask] == 0),
                "test_mask": (starts >= train_size) & (not excluded),
            })
        raw[file.stem] = recordings

    def baseline(recordings):
        return np.concatenate([item["windows"][item["train_mask"]] for item in recordings])

    if scaling == "shared":
        shared = WindowStandardScaler.fit(baseline([r for items in raw.values() for r in items]))
        scalers = dict.fromkeys(raw, shared)
    else:
        scalers = {plant: WindowStandardScaler.fit(baseline(items)) for plant, items in raw.items()}

    return {
        plant: [
            SkabRecording(**item | {"windows": scalers[plant].transform(item["windows"])})
            for item in items
        ]
        for plant, items in raw.items()
    }


def plant_training_windows(recordings: dict[str, list[SkabRecording]]) -> dict[str, np.ndarray]:
    """Return each plant's normal baseline windows for model training."""
    return {
        plant: np.concatenate([recording.train_windows for recording in items])
        for plant, items in recordings.items()
    }
