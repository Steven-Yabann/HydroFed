"""Recording-level audits of saved SKAB models and their training data."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from sklearn.metrics import average_precision_score

from hydrofed.evaluation import compute_metrics, persistent_predictions


def _score_summary(scores: np.ndarray) -> dict | None:
    if len(scores) == 0:
        return None
    quantiles = np.percentile(scores, [0, 10, 50, 90, 95, 99, 100])
    return dict(zip(
        ("min", "q10", "median", "q90", "q95", "q99", "max"),
        map(float, quantiles),
    )) | {"mean": float(scores.mean())}


def _test_metrics(labels, scores, predicted, threshold) -> dict:
    metrics = compute_metrics(labels, scores, threshold, predicted)
    normal = labels == 0
    anomalous = labels == 1
    tp = int(np.count_nonzero(anomalous & (predicted == 1)))
    fp = int(np.count_nonzero(normal & (predicted == 1)))
    tn = int(np.count_nonzero(normal & (predicted == 0)))
    fn = int(np.count_nonzero(anomalous & (predicted == 0)))
    prevalence = float(anomalous.mean())
    return metrics | {
        "test_windows": len(scores),
        "test_anomalous_endpoints": int(anomalous.sum()),
        "anomaly_rate": prevalence,
        "predicted_anomalous_endpoints": int(predicted.sum()),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        # AP is the non-interpolated precision-recall summary. Mark single-
        # class recordings unavailable, as with ROC AUC.
        "average_precision": (
            float(average_precision_score(labels, scores))
            if normal.any() and anomalous.any() else None
        ),
        "always_alert_f1": 2 * prevalence / (1 + prevalence),
        "normal_score_distribution": _score_summary(scores[normal]),
        "anomalous_score_distribution": _score_summary(scores[anomalous]),
    }


def recording_diagnostics(
    train_scores: np.ndarray,
    train_recording_ids: np.ndarray,
    train_window_labels: np.ndarray,
    train_endpoint_labels: np.ndarray,
    test_scores: np.ndarray,
    test_recording_ids: np.ndarray,
    test_labels: np.ndarray,
    threshold: float | Mapping[str, float],
    consecutive_sequences: int = 1,
) -> dict:
    """Audit training contamination and evaluate recordings independently.

    Window labels describe the entire training window according to the
    partition's label mode; endpoint labels describe only its final point.
    Counts refer to overlapping windows/endpoints, not independent samples or
    distinct fault events. Labels are used for reporting, never calibration.
    """
    train_scores = np.asarray(train_scores, dtype=float)
    test_scores = np.asarray(test_scores, dtype=float)
    train_recording_ids = np.asarray(train_recording_ids)
    test_recording_ids = np.asarray(test_recording_ids)
    train_window_labels = np.asarray(train_window_labels)
    train_endpoint_labels = np.asarray(train_endpoint_labels)
    test_labels = np.asarray(test_labels)
    for arrays in (
        (train_scores, train_recording_ids, train_window_labels, train_endpoint_labels),
        (test_scores, test_recording_ids, test_labels),
    ):
        if any(array.ndim != 1 or len(array) != len(arrays[0]) for array in arrays):
            raise ValueError("scores, recording IDs and labels must be aligned vectors")
    if not len(train_scores) or not len(test_scores):
        raise ValueError("training and test scores cannot be empty")
    if not np.isfinite(train_scores).all() or not np.isfinite(test_scores).all():
        raise ValueError("scores must be finite")
    for labels in (train_window_labels, train_endpoint_labels, test_labels):
        if not np.isin(labels, [0, 1]).all():
            raise ValueError("labels must be binary")

    predicted = persistent_predictions(
        test_scores, threshold, test_recording_ids, consecutive_sequences,
    )
    records = {}
    all_ids = dict.fromkeys(train_recording_ids.tolist() + test_recording_ids.tolist())
    for recording_id in all_ids:
        train = train_recording_ids == recording_id
        test = test_recording_ids == recording_id
        cutoff = (
            float(threshold[str(recording_id)])
            if isinstance(threshold, Mapping) else float(threshold)
        )
        records[str(recording_id)] = {
            "training_windows": int(train.sum()),
            "training_windows_labeled_anomalous": int(train_window_labels[train].sum()),
            "training_anomalous_endpoints": int(train_endpoint_labels[train].sum()),
            "training_score_distribution": _score_summary(train_scores[train]),
            "threshold": cutoff,
            "test_metrics": (
                _test_metrics(test_labels[test], test_scores[test], predicted[test], cutoff)
                if test.any() else None
            ),
        }

    # Pooled scores can rank recording identity instead of faults. Report the
    # pooled result next to the mean computed within eligible recordings.
    pooled = _test_metrics(test_labels, test_scores, predicted, threshold=0.0)
    pooled.pop("threshold")
    eligible = [
        record["test_metrics"] for record in records.values()
        if record["test_metrics"] is not None
        and record["test_metrics"]["auc_roc"] is not None
    ]
    return {
        "training_windows": len(train_scores),
        "training_windows_labeled_anomalous": int(train_window_labels.sum()),
        "training_anomalous_endpoints": int(train_endpoint_labels.sum()),
        "pooled_test_metrics": pooled,
        "recordings_with_both_test_classes": len(eligible),
        "macro_recording_auc_roc": (
            float(np.mean([item["auc_roc"] for item in eligible])) if eligible else None
        ),
        "macro_recording_average_precision": (
            float(np.mean([item["average_precision"] for item in eligible]))
            if eligible else None
        ),
        "recordings": records,
    }
