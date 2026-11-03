"""Calibrate SKAB detector scores and turn them into alert metrics.

This module does not import PyTorch, so classic models can use it in a
separate process. Autoencoder scoring lives in ``hydrofed.experiments.regimes``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from hydrofed.evaluation import calibrated_recording_scores, evaluate_skab_scores
from hydrofed.experiments.recordings import SkabRecording

# Candidate thresholds searched on development recordings: baseline MAD units
# for calibrated scores, and probabilities for supervised classifiers.
MAD_THRESHOLDS = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0)
PROBABILITY_THRESHOLDS = tuple(round(value, 2) for value in np.arange(0.05, 1.0, 0.05))
REPORTED_METRICS = (
    "precision", "recall", "f1", "false_positive_rate", "macro_recording_auc_roc",
    "fault_events", "fault_events_detected", "median_detection_delay",
)


@dataclass(frozen=True)
class RecordingScores:
    """Test-window scores in baseline MAD units, with labels and roles."""

    scores: np.ndarray
    labels: np.ndarray
    recording_ids: np.ndarray
    roles: np.ndarray

    def select(self, role: str) -> "RecordingScores":
        return self.where(self.roles == role)

    def where(self, mask: np.ndarray) -> "RecordingScores":
        return RecordingScores(self.scores[mask], self.labels[mask], self.recording_ids[mask], self.roles[mask])


def calibrate_components(
    recordings: list[SkabRecording],
    components: Callable[[SkabRecording, np.ndarray], list[np.ndarray]],
    smoothing: float = 0.2,
    calibration_tail: float = 0.25,
) -> RecordingScores:
    """Score tested recordings from per-window components.

    ``components(recording, windows)`` returns the arrays to compare with the
    recording's baseline, for example sensor levels and model residuals.
    """
    parts = {"train": [], "test": [], "train_ids": [], "test_ids": [], "labels": [], "roles": []}
    for recording in recordings:
        if not recording.test_mask.any() or not recording.train_mask.any():
            continue
        parts["train"].append(components(recording, recording.train_windows))
        parts["test"].append(components(recording, recording.test_windows))
        parts["train_ids"].append(np.repeat(recording.recording_id, recording.train_mask.sum()))
        parts["test_ids"].append(np.repeat(recording.recording_id, recording.test_mask.sum()))
        parts["labels"].append(recording.test_labels)
        parts["roles"].append(np.repeat(recording.role, recording.test_mask.sum()))

    n_components = len(parts["train"][0])
    stack = lambda key: [np.concatenate([item[index] for item in parts[key]]) for index in range(n_components)]
    test_ids = np.concatenate(parts["test_ids"])
    scores = calibrated_recording_scores(
        stack("train"), np.concatenate(parts["train_ids"]),
        stack("test"), test_ids, smoothing, calibration_tail,
    )
    return RecordingScores(scores, np.concatenate(parts["labels"]), test_ids, np.concatenate(parts["roles"]))


def concatenate_scores(parts: list[RecordingScores]) -> RecordingScores:
    """Join scores from several plants or folds."""
    return RecordingScores(*(np.concatenate([getattr(part, name) for part in parts])
                             for name in ("scores", "labels", "recording_ids", "roles")))


def alert_metrics(
    scores: RecordingScores,
    threshold: float,
    consecutive_sequences: int = 1,
    sequence_seconds: dict[str, float] | None = None,
) -> dict[str, float | int | None]:
    """Apply one threshold to calibrated scores and keep the reported metrics."""
    result = evaluate_skab_scores(
        scores.scores, scores.labels, scores.recording_ids, threshold, consecutive_sequences,
        sequence_seconds,
    )
    return {name: result[name] for name in REPORTED_METRICS}


def choose_threshold(
    scores: RecordingScores,
    candidates: np.ndarray,
    max_false_positive_rate: float = 0.15,
    consecutive_sequences: int = 1,
) -> float:
    """Pick the threshold with the highest recall within a false alarm budget.

    This is the predeclared rule for every model. Use development recordings
    only; if no candidate meets the budget, the strictest one is returned.
    """
    best, best_recall = float(max(candidates)), -1.0
    for threshold in sorted(candidates):
        metrics = alert_metrics(scores, float(threshold), consecutive_sequences)
        if metrics["false_positive_rate"] <= max_false_positive_rate and metrics["recall"] > best_recall:
            best, best_recall = float(threshold), metrics["recall"]
    return best


def summarise_seeds(runs: list[dict[str, float | int | None]]) -> dict[str, dict[str, float | None]]:
    """Return the mean and standard deviation of each metric across seeds."""
    summary = {}
    for name in runs[0]:
        values = [run[name] for run in runs if run[name] is not None]
        summary[name] = {
            "mean": float(np.mean(values)) if values else None,
            "sd": float(np.std(values)) if values else None,
        }
    return summary
