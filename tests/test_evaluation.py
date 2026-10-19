import numpy as np
from hydrofed.evaluation import compute_metrics
from hydrofed.training.local import calibrate_threshold

def test_metrics_perfect_separation():
    result=compute_metrics(np.array([0,0,1,1]),np.array([.1,.2,.8,.9]),.5)
    assert result["precision"]==result["recall"]==result["f1"]==result["auc_roc"]==1

def test_threshold_percentile():
    assert calibrate_threshold(np.array([1,2,3,4]),50)==2.5

def test_metrics_mark_auc_unavailable_for_single_class_labels():
    labels = np.array([0, 0, 0, 0])
    scores = np.array([0.1, 0.2, 0.3, 0.4])

    result = compute_metrics(
        labels,
        scores,
        threshold=0.25,
    )

    assert result["auc_roc"] is None
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0
    assert result["threshold"] == 0.25