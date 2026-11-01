"""Anomaly metrics and model evaluation."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from hydrofed.training.local import reconstruction_errors

def compute_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | None]:
    labels, scores = np.asarray(labels), np.asarray(scores)
    predicted = (scores > threshold).astype(int)
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
