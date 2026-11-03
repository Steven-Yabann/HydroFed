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


def mahalanobis_scores(
    values: np.ndarray,
    baseline: np.ndarray,
    shrinkage: float = 0.1,
) -> np.ndarray:
    """Measure each row's distance from a recording's normal baseline."""
    values, baseline = np.asarray(values, dtype=float), np.asarray(baseline, dtype=float)
    if len(baseline) < 2:
        raise ValueError("baseline must contain at least two rows")
    if not 0 <= shrinkage <= 1:
        raise ValueError("shrinkage must be between 0 and 1")

    covariance = np.cov(baseline, rowvar=False) + 1e-6 * np.eye(baseline.shape[1])
    covariance = (
        (1 - shrinkage) * covariance
        + shrinkage * np.diag(np.diag(covariance))
    )
    deviations = values - baseline.mean(axis=0)
    distances = np.einsum(
        "ij,jk,ik->i",
        deviations,
        np.linalg.inv(covariance),
        deviations,
    )
    return np.sqrt(np.clip(distances, 0, None))


def robust_zscores(scores: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Express scores in median and MAD units of a normal baseline."""
    median = np.median(baseline)
    spread = 1.4826 * np.median(np.abs(baseline - median)) + 1e-6
    return (np.asarray(scores, dtype=float) - median) / spread


def exponential_smoothing(scores: np.ndarray, alpha: float) -> np.ndarray:
    """Causally smooth scores so sustained evidence outweighs single spikes."""
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")

    smoothed = np.empty(len(scores), dtype=float)
    state = float(scores[0]) if len(scores) else 0.0
    for index, score in enumerate(scores):
        state = alpha * float(score) + (1 - alpha) * state
        smoothed[index] = state
    return smoothed


def _baseline_zscores(
    baseline: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return robust z-scores of baseline and test rows for one component.

    Two-dimensional components are per-sensor features, reduced with a
    Mahalanobis distance from the baseline. One-dimensional components are
    already anomaly scores, such as an Isolation Forest output.
    """
    if baseline.ndim == 1:
        return robust_zscores(baseline, baseline), robust_zscores(values, baseline)
    baseline_distances = mahalanobis_scores(baseline, baseline)
    return (
        robust_zscores(baseline_distances, baseline_distances),
        robust_zscores(mahalanobis_scores(values, baseline), baseline_distances),
    )


def calibrated_recording_scores(
    train_components: list[np.ndarray],
    train_recording_ids: np.ndarray,
    test_components: list[np.ndarray],
    test_recording_ids: np.ndarray,
    smoothing: float = 0.2,
    calibration_tail: float = 0.25,
) -> np.ndarray:
    """Score sequences in MAD units of each recording's own normal baseline.

    Every component is compared with the recording's baseline and converted to
    a robust z-score. The largest component is smoothed, then centred on the
    median of the last part of the baseline and divided by its MAD. A
    threshold of ``k`` on the returned scores therefore means median + k MAD
    of recent normal behaviour, whichever model produced the components. Test
    labels are not used. Recordings without baseline windows are not scored.
    """
    if not train_components or len(train_components) != len(test_components):
        raise ValueError("train and test components must align and not be empty")
    if not 0 < calibration_tail <= 1:
        raise ValueError("calibration_tail must be in (0, 1]")

    train_recording_ids = np.asarray(train_recording_ids)
    test_recording_ids = np.asarray(test_recording_ids)
    train_components = [np.asarray(values, dtype=float) for values in train_components]
    test_components = [np.asarray(values, dtype=float) for values in test_components]
    test_scores = np.full(len(test_recording_ids), -np.inf)

    for recording_id in dict.fromkeys(train_recording_ids.tolist()):
        train_mask = train_recording_ids == recording_id
        test_mask = test_recording_ids == recording_id
        baseline_scores, scores = zip(*(
            _baseline_zscores(train[train_mask], test[test_mask])
            for train, test in zip(train_components, test_components)
        ))

        # The smoother runs through the baseline so scoring starts from a
        # settled state, as it would on a live sensor stream.
        smoothed = exponential_smoothing(
            np.concatenate((
                np.maximum.reduce(baseline_scores),
                np.maximum.reduce(scores),
            )),
            smoothing,
        )
        n_train = int(train_mask.sum())
        calibration = smoothed[
            n_train - max(1, int(np.ceil(n_train * calibration_tail))):n_train
        ]
        median = np.median(calibration)
        spread = 1.4826 * np.median(np.abs(calibration - median)) + 1e-6
        test_scores[test_mask] = (smoothed[n_train:] - median) / spread

    return test_scores


def hybrid_recording_scores(
    train_levels: np.ndarray,
    train_residuals: np.ndarray,
    train_recording_ids: np.ndarray,
    test_levels: np.ndarray,
    test_residuals: np.ndarray,
    test_recording_ids: np.ndarray,
    smoothing: float = 0.2,
    mad_multiplier: float = 24.0,
    calibration_tail: float = 0.25,
) -> tuple[np.ndarray, dict[str, float]]:
    """Score sequences against each recording's own normal baseline.

    Two complementary distances are combined, as in SKAB's T-squared+Q
    detector: how far sensor levels sit from the baseline, and how far the
    model's residuals do. The threshold comes from the last part of the
    baseline, which approximates the most recent normal behaviour.
    """
    if mad_multiplier <= 0:
        raise ValueError("mad_multiplier must be positive")

    test_scores = calibrated_recording_scores(
        [train_levels, train_residuals],
        train_recording_ids,
        [test_levels, test_residuals],
        test_recording_ids,
        smoothing,
        calibration_tail,
    )
    thresholds = {
        str(recording_id): float(mad_multiplier)
        for recording_id in dict.fromkeys(np.asarray(train_recording_ids).tolist())
    }
    return test_scores, thresholds


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


def false_positive_rate(labels: np.ndarray, predicted: np.ndarray) -> float | None:
    """Return the share of normal endpoints that raised an alert."""
    normal = np.asarray(labels) == 0
    return float(np.asarray(predicted)[normal].mean()) if normal.any() else None


def macro_recording_auc_roc(
    scores: np.ndarray,
    labels: np.ndarray,
    recording_ids: np.ndarray,
) -> float | None:
    """Average ROC AUC within recordings.

    Pooled AUC rewards score-level differences between recordings, so it can
    look excellent for a detector that ranks a recording's faults poorly.
    """
    scores, labels = np.asarray(scores), np.asarray(labels)
    recording_ids = np.asarray(recording_ids)
    values = [
        roc_auc_score(labels[recording_ids == recording_id], scores[recording_ids == recording_id])
        for recording_id in dict.fromkeys(recording_ids.tolist())
        if len(np.unique(labels[recording_ids == recording_id])) == 2
    ]
    return float(np.mean(values)) if values else None


def fault_event_detection(
    labels: np.ndarray,
    predicted: np.ndarray,
    recording_ids: np.ndarray,
) -> dict[str, float | int | None]:
    """Count detected faults and the delay to their first alert.

    SKAB recordings contain one fault each. A fault counts as detected when an
    alert fires between its first and last anomalous endpoint, and the delay
    is measured in sequences from its first anomalous endpoint. Sequences are
    one second apart in HydroFed's SKAB partitions.
    """
    labels, predicted = np.asarray(labels), np.asarray(predicted)
    recording_ids = np.asarray(recording_ids)
    events = detected = 0
    delays = []

    for recording_id in dict.fromkeys(recording_ids.tolist()):
        mask = recording_ids == recording_id
        anomalous = np.flatnonzero(labels[mask])
        if not len(anomalous):
            continue
        events += 1
        alerts = np.flatnonzero(predicted[mask][anomalous[0]:anomalous[-1] + 1])
        if len(alerts):
            detected += 1
            delays.append(int(alerts[0]))

    return {
        "fault_events": events,
        "fault_events_detected": detected,
        "median_detection_delay": float(np.median(delays)) if delays else None,
    }


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
        "false_positive_rate": false_positive_rate(endpoint_labels, predicted),
        "macro_recording_auc_roc": macro_recording_auc_roc(
            scores,
            endpoint_labels,
            recording_ids,
        ),
        **fault_event_detection(endpoint_labels, predicted, recording_ids),
        "mean_anomaly_score": float(scores.mean()),
        "consecutive_sequences": int(consecutive_sequences),
        **threshold_details,
    }
