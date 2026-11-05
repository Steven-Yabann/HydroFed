"""Load SKAB plant partitions as scaled recordings with a dev/final split.

Comparison experiments choose settings on development recordings and report
them on final recordings. The roles alternate within each plant, so both sets
cover every plant and every fault type in similar proportions.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hydrofed.data.pipeline import load_client_partition, load_client_partition_metadata
from hydrofed.data.scaling import WindowStandardScaler
from hydrofed.experiments.scenarios import (
    PlantProfile,
    apply_profile,
    recording_series,
    window_series,
)

ROLES = ("dev", "final")


@dataclass(frozen=True)
class SkabRecording:
    """One SKAB recording, scaled and split SKAB-style into baseline and test.

    ``train_mask`` marks the baseline every detector compares against.
    ``fit_mask`` marks the baseline windows a model may train on, which is
    smaller when a scenario limits a plant's training data.
    """

    plant: str
    recording_id: str
    role: str
    windows: np.ndarray
    endpoint_labels: np.ndarray
    window_labels: np.ndarray
    train_mask: np.ndarray
    test_mask: np.ndarray
    step_seconds: int = 1
    fit_mask: np.ndarray | None = None

    @property
    def fault_type(self) -> str:
        """Return the SKAB folder name, such as ``valve1`` or ``other``."""
        return Path(self.recording_id).parent.name

    @property
    def train_windows(self) -> np.ndarray:
        return self.windows[self.train_mask]

    @property
    def fit_windows(self) -> np.ndarray:
        return self.windows[self.train_mask if self.fit_mask is None else self.fit_mask]

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
    profiles: Mapping[str, PlantProfile] | None = None,
    training_seconds: Mapping[str, int] | None = None,
    profile_seed: int = 0,
) -> dict[str, list[SkabRecording]]:
    """Load every plant partition as a list of recordings.

    Baseline windows end inside a recording's first ``train_size`` seconds and
    contain no anomalous labels. Test windows start at or after that boundary.
    Scalers are fitted on baseline windows only, per plant or across plants.
    Excluded recordings keep their baseline for training but are not tested.

    ``profiles`` changes how a plant's sensors record the data, with noise
    drawn from ``profile_seed`` so every training seed sees the same data.
    ``training_seconds`` limits a plant's model training windows to those
    ending within the first seconds of each recording.
    """
    profiles = profiles or {}
    training_seconds = training_seconds or {}
    if scaling not in {"plant", "shared"}:
        raise ValueError("scaling must be 'plant' or 'shared'")

    files = sorted(Path(partitions).glob("plant_*.npz"))
    if not files:
        raise FileNotFoundError(f"No plant partitions found under {partitions}")

    raw: dict[str, list[dict]] = {}
    for plant_index, file in enumerate(files):
        windows, window_labels = load_client_partition(file)
        metadata = load_client_partition_metadata(file)
        recording_ids = metadata["recording_ids"]
        roles = recording_roles(recording_ids)
        length = windows.shape[1]
        profile = profiles.get(file.stem)
        if profile is not None:
            starts = metadata["window_start_indices"]
            baseline_rows = windows[(starts + length - 1 < train_size) & (window_labels == 0)]
            reference = baseline_rows.reshape(-1, windows.shape[2])
            reference_mean, reference_sd = reference.mean(axis=0), reference.std(axis=0)
        recordings = []

        for recording_index, (recording_id, role) in enumerate(roles.items()):
            mask = recording_ids == recording_id
            recording_windows = windows[mask]
            endpoint_labels = metadata["endpoint_labels"][mask].astype(np.int64)
            recording_window_labels = window_labels[mask]
            starts = metadata["window_start_indices"][mask]
            step = 1
            if profile is not None:
                rng = np.random.default_rng([profile_seed, plant_index, recording_index])
                values, labels = apply_profile(
                    *recording_series(recording_windows, endpoint_labels, recording_window_labels),
                    profile, reference_mean, reference_sd, rng,
                )
                recording_windows, endpoint_labels, recording_window_labels, starts = window_series(
                    values, labels, length,
                )
                step = profile.step_seconds
            # Times are in seconds, so slower loggers keep the same boundaries.
            start_seconds = starts * step
            end_seconds = (starts + length - 1) * step
            excluded = any(name in recording_id for name in exclude_test_recordings)
            train_mask = (end_seconds < train_size) & (recording_window_labels == 0)
            recordings.append({
                "plant": file.stem,
                "recording_id": recording_id,
                "role": role,
                "windows": recording_windows,
                "endpoint_labels": endpoint_labels,
                "window_labels": recording_window_labels,
                "train_mask": train_mask,
                "test_mask": (start_seconds >= train_size) & (not excluded),
                "step_seconds": step,
                "fit_mask": (
                    train_mask & (end_seconds < training_seconds[file.stem])
                    if file.stem in training_seconds else None
                ),
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
        plant: np.concatenate([recording.fit_windows for recording in items])
        for plant, items in recordings.items()
    }
