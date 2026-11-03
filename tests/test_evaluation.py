import numpy as np
from hydrofed.evaluation import (
    calibrate_recording_thresholds,
    compute_metrics,
    exponential_smoothing,
    hybrid_recording_scores,
    macro_recording_auc_roc,
    mahalanobis_scores,
    persistent_predictions,
    robust_zscores,
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


def test_mahalanobis_flags_one_sensor_moving_within_correlated_baseline():
    rng = np.random.default_rng(0)
    shared = rng.normal(size=500)
    baseline = np.column_stack((shared, shared + 0.05 * rng.normal(size=500)))

    scores = mahalanobis_scores(
        np.array([[1.0, 1.0], [1.0, -1.0]]),
        baseline,
    )

    assert scores[1] > 3 * scores[0]


def test_robust_zscores_ignore_baseline_outliers():
    baseline = np.array([1.0, 2.0, 3.0, 4.0, 1000.0])

    assert robust_zscores(np.array([3.0]), baseline)[0] == 0.0


def test_exponential_smoothing_is_causal_and_starts_at_first_score():
    smoothed = exponential_smoothing(np.array([1.0, 1.0, 3.0, 9.0]), 0.5)

    assert smoothed[:3].tolist() == [1.0, 1.0, 2.0]


def _hybrid_scores(test_shift):
    rng = np.random.default_rng(1)
    train = rng.normal(size=(300, 3))
    test = rng.normal(size=(100, 3))
    test[50:, 2] += test_shift
    train_ids = np.array(["a"] * 150 + ["b"] * 150)
    test_ids = np.array(["a"] * 100)
    return hybrid_recording_scores(
        train,
        train,
        train_ids,
        test,
        test,
        test_ids,
    )


def test_hybrid_scores_alert_on_a_single_shifted_sensor_only():
    scores, thresholds = _hybrid_scores(test_shift=20.0)

    assert (scores[60:] > thresholds["a"]).all()
    assert (scores[:50] <= thresholds["a"]).all()


def test_hybrid_thresholds_do_not_depend_on_test_data():
    _, quiet = _hybrid_scores(test_shift=0.0)
    _, faulty = _hybrid_scores(test_shift=50.0)

    assert quiet == faulty


def test_macro_recording_auc_ignores_level_differences_between_recordings():
    # The fault ranks below its own recording's normal data, but above a
    # quieter fault-free recording, so only the pooled AUC looks good.
    scores = np.array([5.0, 6.0, 1.0, 2.0, 3.0])
    labels = np.array([1, 0, 0, 0, 0])
    recordings = np.array(["a", "a", "b", "b", "b"])

    assert compute_metrics(labels, scores, 5.5)["auc_roc"] == 0.75
    assert macro_recording_auc_roc(scores, labels, recordings) == 0.0
