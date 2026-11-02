"""Anomaly metrics and model evaluation."""
from __future__ import annotations
from collections.abc import Mapping
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from hydrofed.training.local import (
    reconstruction_absolute_errors,
    reconstruction_errors,
)

def compute_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    predicted: np.ndarray | None = None,
) -> dict[str, float | None]:
    labels, scores = np.asarray(labels), np.asarray(scores)
    if predicted is None:
        predicted = (scores > threshold).astype(int)
    else:
        predicted = np.asarray(predicted, dtype=np.int64)
        if len(predicted) != len(labels):
            raise ValueError("predicted labels must align with true labels")
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predicted, average="binary", zero_division=0)

    if len(np.unique(labels)) < 2:
        auc_roc = None
    else:
        auc_roc = float(roc_auc_score(labels, scores))

    normal_scores = scores[labels == 0]
    anomalous_scores = scores[labels != 0]

    mean_normal_score = (
        float(normal_scores.mean())
        if len(normal_scores)
        else None
    )
    mean_anomalous_score = (
        float(anomalous_scores.mean())
        if len(anomalous_scores)
        else None
    )
    score_gap = (
        mean_anomalous_score - mean_normal_score
        if mean_normal_score is not None
        and mean_anomalous_score is not None
        else None
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc_roc": auc_roc,
        "mean_normal_score": mean_normal_score,
        "mean_anomalous_score": mean_anomalous_score,
        "score_gap": score_gap,
        "threshold": float(threshold),
    }

def evaluate_model(model, windows, labels, threshold, device: str = "cpu"):
    scores = reconstruction_errors(model, windows, device=device)
    return compute_metrics(labels, scores, threshold) | {"mean_anomaly_score": float(scores.mean())}


def persistent_predictions(
    scores: np.ndarray,
    threshold: float | Mapping[str, float],
    recording_ids: np.ndarray,
    consecutive_sequences: int = 10,
) -> np.ndarray:
    """Require consecutive over-threshold sequences before raising an alert."""
    scores = np.asarray(scores, dtype=float)
    recording_ids = np.asarray(recording_ids)

    if len(scores) != len(recording_ids):
        raise ValueError("scores and recording_ids must have the same length")
    if consecutive_sequences <= 0:
        raise ValueError("consecutive_sequences must be positive")

    predicted = np.zeros(len(scores), dtype=np.int64)

    for recording_id in dict.fromkeys(recording_ids.tolist()):
        indices = np.flatnonzero(recording_ids == recording_id)
        if isinstance(threshold, Mapping):
            try:
                recording_threshold = threshold[str(recording_id)]
            except KeyError as error:
                raise ValueError(
                    f"No threshold is available for recording {recording_id!r}"
                ) from error
        else:
            recording_threshold = threshold
        run_length = 0

        for index in indices:
            if scores[index] > recording_threshold:
                run_length += 1
                if run_length >= consecutive_sequences:
                    predicted[index] = 1
            else:
                run_length = 0

    return predicted


def calibrate_recording_thresholds(
    scores: np.ndarray,
    recording_ids: np.ndarray,
    percentile: float,
    scale: float = 1.0,
) -> dict[str, float]:
    """Calibrate one anomaly threshold from each recording's normal section."""
    scores = np.asarray(scores, dtype=float)
    recording_ids = np.asarray(recording_ids)

    if len(scores) != len(recording_ids):
        raise ValueError("scores and recording_ids must have the same length")
    if len(scores) == 0:
        raise ValueError("scores cannot be empty")
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100")
    if scale <= 0:
        raise ValueError("scale must be positive")

    return {
        str(recording_id): float(
            np.percentile(scores[recording_ids == recording_id], percentile) * scale
        )
        for recording_id in dict.fromkeys(recording_ids.tolist())
    }


def evaluate_skab_model(
    model,
    windows: np.ndarray,
    endpoint_labels: np.ndarray,
    recording_ids: np.ndarray,
    threshold: float | Mapping[str, float],
    consecutive_sequences: int = 10,
    device: str = "cpu",
) -> dict[str, float | None]:
    """Evaluate point-aligned SKAB sequence endpoints with persistence."""
    scores = reconstruction_absolute_errors(model, windows, device=device)
    return evaluate_skab_scores(
        scores, endpoint_labels, recording_ids, threshold, consecutive_sequences,
    )


def evaluate_skab_scores(
    scores: np.ndarray,
    endpoint_labels: np.ndarray,
    recording_ids: np.ndarray,
    threshold: float | Mapping[str, float],
    consecutive_sequences: int = 10,
) -> dict:
    """Apply the SKAB alert policy to previously computed scores."""
    predicted = persistent_predictions(
        scores,
        threshold,
        recording_ids,
        consecutive_sequences,
    )
    if isinstance(threshold, Mapping):
        threshold_values = np.asarray(list(threshold.values()), dtype=float)
        reported_threshold = float(threshold_values.mean())
        threshold_details = {
            "threshold_mode": "per_recording",
            "threshold_min": float(threshold_values.min()),
            "threshold_max": float(threshold_values.max()),
        }
    else:
        reported_threshold = float(threshold)
        threshold_details = {"threshold_mode": "per_plant"}

    return compute_metrics(endpoint_labels, scores, reported_threshold, predicted) | {
        "mean_anomaly_score": float(scores.mean()),
        "consecutive_sequences": int(consecutive_sequences),
        **threshold_details,
    }
