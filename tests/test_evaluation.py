import numpy as np
from hydrofed.evaluation import compute_metrics
from hydrofed.training.local import calibrate_threshold

def test_metrics_perfect_separation():
    result=compute_metrics(np.array([0,0,1,1]),np.array([.1,.2,.8,.9]),.5)
    assert result["precision"]==result["recall"]==result["f1"]==result["auc_roc"]==1

def test_threshold_percentile():
    assert calibrate_threshold(np.array([1,2,3,4]),50)==2.5
