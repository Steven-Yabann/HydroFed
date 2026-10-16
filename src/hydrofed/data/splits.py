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