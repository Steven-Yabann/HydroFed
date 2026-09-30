"""Anomaly metrics and model evaluation."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from hydrofed.training.local import reconstruction_errors

def compute_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    labels, scores = np.asarray(labels), np.asarray(scores)
    predicted = (scores > threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predicted, average="binary", zero_division=0)
    auc = roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else float("nan")
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1),
            "auc_roc": float(auc), "threshold": float(threshold)}

def evaluate_model(model, windows, labels, threshold):
    scores = reconstruction_errors(model, windows)
    return compute_metrics(labels, scores, threshold) | {"mean_anomaly_score": float(scores.mean())}
