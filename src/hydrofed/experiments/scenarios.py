"""Make simulated plants differ, or give some of them little training data.

All five simulated plants are slices of one SKAB test rig. Two scenarios test
federated learning where it is supposed to matter:

heterogeneous
    Each plant has a fixed profile: calibration gain and offset, extra sensor
    noise, coarser sensor resolution, and a slower logging interval. A plant's
    own scaler cancels gains and offsets, but not noise, resolution or timing.

scarce
    Chosen plants train on only the first seconds of each recording. Every
    recording keeps its full baseline as the alert reference and the test
    windows do not change, so only the model's training data differs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlantProfile:
    """How one plant's sensors differ from the SKAB test rig.

    ``noise``, ``resolution`` and ``offset`` are in units of each sensor's
    baseline standard deviation at that plant.
    """

    step_seconds: int = 1
    noise: float = 0.0
    resolution: float = 0.0
    gain: float = 1.0
    offset: float = 0.0
    description: str = "SKAB test rig as recorded"

    def __post_init__(self) -> None:
        if self.step_seconds < 1:
            raise ValueError("step_seconds must be at least 1")
        if self.noise < 0 or self.resolution < 0 or self.gain <= 0:
            raise ValueError("noise and resolution must be non-negative and gain positive")


HETEROGENEOUS_PROFILES = {
    "plant_1": PlantProfile(),
    "plant_2": PlantProfile(noise=0.5, description="noisy sensors"),
    "plant_3": PlantProfile(step_seconds=5, description="logs every 5 seconds"),
    "plant_4": PlantProfile(
        step_seconds=2, noise=0.25, gain=1.2, offset=1.0,
        description="logs every 2 seconds, recalibrated sensors with some noise",
    ),
    "plant_5": PlantProfile(
        resolution=0.5, gain=0.8, offset=-1.0,
        description="coarse sensors, recalibrated",
    ),
}

SCARCE_PLANTS = ("plant_4", "plant_5")
SCARCE_SECONDS = (100, 50, 25)


def recording_series(
    windows: np.ndarray,
    endpoint_labels: np.ndarray,
    window_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild a recording's sensor rows and point labels from stride-1 windows.

    Each window's endpoint label is the label of its last row. Rows before the
    first endpoint take the first window's label, which is normal for every
    SKAB recording because faults start well after the baseline.
    """
    length = windows.shape[1]
    values = np.concatenate([windows[:, 0], windows[-1, 1:]])
    labels = np.concatenate([
        np.full(length - 1, int(window_labels[0]), dtype=np.int64),
        np.asarray(endpoint_labels, dtype=np.int64),
    ])
    return values, labels


def apply_profile(
    values: np.ndarray,
    labels: np.ndarray,
    profile: PlantProfile,
    reference_mean: np.ndarray,
    reference_sd: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Pass true sensor values through a plant's calibration, noise, ADC and logger."""
    sd = np.where(reference_sd > 0, reference_sd, 1.0)
    measured = reference_mean + profile.gain * (values - reference_mean) + profile.offset * sd
    if profile.noise:
        measured = measured + rng.normal(scale=profile.noise * sd, size=measured.shape)
    if profile.resolution:
        step = profile.resolution * sd
        measured = np.round(measured / step) * step
    return measured[::profile.step_seconds].astype(np.float32), labels[::profile.step_seconds]


def window_series(
    values: np.ndarray,
    labels: np.ndarray,
    length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return stride-1 windows, endpoint labels, any-anomaly labels and start rows."""
    starts = np.arange(len(values) - length + 1)
    index = starts[:, None] + np.arange(length)
    return values[index], labels[starts + length - 1], labels[index].max(axis=1), starts
