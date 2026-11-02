"""Scaling utilities for windowed telemetry data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from hydrofed.data.splits import PlantDataSplit


def _validate_windows(windows: np.ndarray) -> np.ndarray:
    """Return floating-point windows after checking their expected shape."""
    array = np.asarray(windows, dtype=np.float32)

    if array.ndim != 3:
        raise ValueError("windows must have shape (samples, time_steps, features)")
    if len(array) == 0:
        raise ValueError("windows cannot be empty")

    return array


@dataclass(frozen=True)
class WindowMinMaxScaler:
    """Feature-wise min-max scaler fitted to training telemetry windows.

    The same feature statistics must be used by every federated client so that
    shared model weights represent the same sensor units everywhere.
    """

    minimums: np.ndarray
    spans: np.ndarray

    @classmethod
    def fit(cls, windows: np.ndarray) -> "WindowMinMaxScaler":
        """Fit feature statistics using only the supplied training windows."""
        array = _validate_windows(windows)
        minimums = array.min(axis=(0, 1))
        maximums = array.max(axis=(0, 1))
        spans = maximums - minimums

        # A constant sensor has no scale range. Keeping its transformed value
        # at zero is finite and avoids division by zero.
        spans = np.where(spans == 0, 1.0, spans).astype(np.float32)
        return cls(minimums=minimums.astype(np.float32), spans=spans)

    def transform(self, windows: np.ndarray) -> np.ndarray:
        """Apply the fitted feature-wise scale without changing window shape."""
        array = _validate_windows(windows)

        if array.shape[2] != len(self.minimums):
            raise ValueError(
                "windows have a different number of features than the fitted scaler"
            )

        return ((array - self.minimums) / self.spans).astype(np.float32)


def fit_shared_window_scaler(
    training_windows: Iterable[np.ndarray],
) -> WindowMinMaxScaler:
    """Fit one scaler across normal training windows from all simulated plants."""
    arrays = [_validate_windows(windows) for windows in training_windows]

    if not arrays:
        raise ValueError("at least one plant training window array is required")

    feature_counts = {array.shape[2] for array in arrays}
    if len(feature_counts) != 1:
        raise ValueError("all plant windows must have the same number of features")

    return WindowMinMaxScaler.fit(np.concatenate(arrays, axis=0))


def scale_plant_split(
    plant_split: PlantDataSplit,
    scaler: WindowMinMaxScaler,
) -> PlantDataSplit:
    """Apply one fitted scaler to every portion of one plant's data."""
    return PlantDataSplit(
        train_windows=scaler.transform(plant_split.train_windows),
        calibration_windows=scaler.transform(plant_split.calibration_windows),
        test_windows=scaler.transform(plant_split.test_windows),
        test_labels=plant_split.test_labels,
    )
