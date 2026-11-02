import numpy as np
from hydrofed.evaluation import (
    calibrate_recording_thresholds,
    compute_metrics,
    persistent_predictions,
)
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
    assert result["mean_normal_score"] == 0.25
    assert result["mean_anomalous_score"] is None
    assert result["score_gap"] is None

def test_metrics_report_normal_and_anomalous_score_separation():
    result = compute_metrics(
        labels=np.array([0, 0, 1, 1]),
        scores=np.array([0.1, 0.3, 0.7, 0.9]),
        threshold=0.5,
    )

    assert result["mean_normal_score"] == 0.2
    assert result["mean_anomalous_score"] == 0.8
    assert result["score_gap"] == 0.6000000000000001


def test_persistent_predictions_require_a_full_run_within_each_recording():
    scores = np.array([0.9, 0.9, 0.2, 0.9, 0.9, 0.9])
    recording_ids = np.array(["a", "a", "a", "b", "b", "b"])

    predicted = persistent_predictions(
        scores,
        threshold=0.5,
        recording_ids=recording_ids,
        consecutive_sequences=2,
    )

    np.testing.assert_array_equal(predicted, np.array([0, 1, 0, 0, 1, 1]))


def test_compute_metrics_accepts_persistence_predictions():
    result = compute_metrics(
        labels=np.array([0, 1, 1]),
        scores=np.array([0.1, 0.7, 0.8]),
        threshold=0.5,
        predicted=np.array([0, 0, 1]),
    )

    assert result["precision"] == 1.0
    assert result["recall"] == 0.5


def test_recording_thresholds_calibrate_each_normal_section_separately():
    thresholds = calibrate_recording_thresholds(
        scores=np.array([1.0, 3.0, 10.0, 30.0]),
        recording_ids=np.array(["a", "a", "b", "b"]),
        percentile=50,
    )

    assert thresholds == {"a": 2.0, "b": 20.0}


def test_persistent_predictions_accept_recording_specific_thresholds():
    predicted = persistent_predictions(
        scores=np.array([2.5, 19.0, 21.0]),
        threshold={"a": 2.0, "b": 20.0},
        recording_ids=np.array(["a", "b", "b"]),
        consecutive_sequences=1,
    )

    np.testing.assert_array_equal(predicted, np.array([1, 0, 1]))
