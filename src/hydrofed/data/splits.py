"""Utilities for separating each plant's train, calibration, and test data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlantDataSplit:
    """Disjoint windows used by one simulated plant."""

    train_windows: np.ndarray
    calibration_windows: np.ndarray
    test_windows: np.ndarray
    test_labels: np.ndarray
    test_recording_ids: np.ndarray | None = None


def split_plant_windows(
    windows: np.ndarray,
    labels: np.ndarray,
    train_fraction: float = 0.6,
    calibration_fraction: float = 0.2,
) -> PlantDataSplit:
    """Split an ordered plant partition into training, calibration, and test data.

    The autoencoder trains and calibrates only on normal windows. The final
    test portion retains both normal and anomalous labels for evaluation.
    """
    if len(windows) != len(labels):
        raise ValueError("windows and labels must have the same length")

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")

    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction must be between 0 and 1")

    if train_fraction + calibration_fraction >= 1:
        raise ValueError("train and calibration fractions must leave test data")

    n_windows = len(windows)
    train_end = int(n_windows * train_fraction)
    calibration_end = train_end + int(n_windows * calibration_fraction)

    if train_end == 0 or calibration_end == train_end or calibration_end >= n_windows:
        raise ValueError("not enough windows for train, calibration, and test splits")

    train_labels = labels[:train_end]
    calibration_labels = labels[train_end:calibration_end]

    train_windows = windows[:train_end][train_labels == 0]
    calibration_windows = windows[train_end:calibration_end][calibration_labels == 0]
    test_windows = windows[calibration_end:]
    test_labels = labels[calibration_end:]

    if len(train_windows) == 0:
        raise ValueError("training split contains no normal windows")

    if len(calibration_windows) == 0:
        raise ValueError("calibration split contains no normal windows")

    return PlantDataSplit(
        train_windows=train_windows,
        calibration_windows=calibration_windows,
        test_windows=test_windows,
        test_labels=test_labels,
    )


def split_skab_plant_windows(
    windows: np.ndarray,
    endpoint_labels: np.ndarray,
    recording_ids: np.ndarray,
    window_start_indices: np.ndarray,
    train_size: int = 400,
) -> PlantDataSplit:
    """Create SKAB-style chronological train and point-aligned test windows.

    Each recording contributes windows ending inside its first ``train_size``
    observations for training. Test windows must start at or after that
    boundary, so no test sequence contains training timestamps. Test labels are
    the original SKAB anomaly labels at each sequence endpoint.
    """
    arrays = (windows, endpoint_labels, recording_ids, window_start_indices)
    if any(len(array) != len(windows) for array in arrays[1:]):
        raise ValueError("SKAB window metadata must align with windows")
    if train_size <= 0:
        raise ValueError("train_size must be positive")

    train_sets = []
    test_sets = []
    label_sets = []
    test_recording_sets = []

    for recording_id in dict.fromkeys(recording_ids.tolist()):
        mask = recording_ids == recording_id
        starts = window_start_indices[mask]
        ends = starts + windows.shape[1] - 1
        recording_windows = windows[mask]
        recording_labels = endpoint_labels[mask]

        train_mask = ends < train_size
        test_mask = starts >= train_size

        if train_mask.any():
            # SKAB's chronological training section is used without looking at
            # its labels. In the benchmark recordings this is the pre-fault
            # operating period.
            train_sets.append(recording_windows[train_mask])

        if test_mask.any():
            test_sets.append(recording_windows[test_mask])
            label_sets.append(recording_labels[test_mask].astype(np.int64))
            test_recording_sets.append(
                np.repeat(str(recording_id), test_mask.sum())
            )

    if not train_sets:
        raise ValueError("No SKAB training windows are available")
    if not test_sets:
        raise ValueError("No SKAB test windows are available")

    train_windows = np.concatenate(train_sets)
    # The SKAB protocol calibrates its threshold on training residuals. Keep a
    # copy here so the common split shape remains usable by training code.
    return PlantDataSplit(
        train_windows=train_windows,
        calibration_windows=train_windows,
        test_windows=np.concatenate(test_sets),
        test_labels=np.concatenate(label_sets),
        test_recording_ids=np.concatenate(test_recording_sets),
    )
