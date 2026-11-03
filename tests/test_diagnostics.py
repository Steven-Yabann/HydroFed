import numpy as np
import pytest

from hydrofed.diagnostics import recording_diagnostics


def test_report_exposes_pooled_auc_confounding_and_training_contamination():
    report = recording_diagnostics(
        train_scores=np.array([1.0, 2.0, 12.0, 13.0]),
        train_recording_ids=np.array(["normal", "normal", "faulty", "faulty"]),
        # A window can contain an earlier anomaly despite its normal endpoint.
        train_window_labels=np.array([0, 0, 1, 1]),
        train_endpoint_labels=np.array([0, 0, 0, 1]),
        test_scores=np.array([1.0, 2.0, 10.0, 9.0]),
        test_recording_ids=np.array(["normal", "normal", "faulty", "faulty"]),
        test_labels=np.array([0, 0, 0, 1]),
        threshold={"normal": 1.5, "faulty": 9.5},
    )

    assert report["training_windows_labeled_anomalous"] == 2
    assert report["training_anomalous_endpoints"] == 1
    assert report["pooled_test_metrics"]["auc_roc"] == pytest.approx(2 / 3)
    assert report["macro_recording_auc_roc"] == 0.0
    assert report["recordings_with_both_test_classes"] == 1
    faulty = report["recordings"]["faulty"]
    assert faulty["training_windows_labeled_anomalous"] == 2
    assert faulty["training_anomalous_endpoints"] == 1
    assert faulty["training_score_distribution"]["median"] == 12.5
    metrics = faulty["test_metrics"]
    assert metrics["false_positives"] == metrics["false_negatives"] == 1
    assert metrics["true_positives"] == metrics["true_negatives"] == 0
    assert metrics["threshold"] == 9.5
    assert metrics["average_precision"] == 0.5
    assert metrics["anomaly_rate"] == 0.5
    assert metrics["always_alert_f1"] == pytest.approx(2 / 3)
    assert metrics["anomalous_score_distribution"]["q90"] == 9.0


def test_normal_only_recordings_report_false_alarms_and_unavailable_ranking():
    report = recording_diagnostics(
        train_scores=np.array([0.1, 0.2]),
        train_recording_ids=np.array(["a", "a"]),
        train_window_labels=np.array([0, 0]),
        train_endpoint_labels=np.array([0, 0]),
        test_scores=np.array([0.1, 0.9, 0.9]),
        test_recording_ids=np.array(["a", "a", "a"]),
        test_labels=np.array([0, 0, 0]),
        threshold=0.5,
        consecutive_sequences=2,
    )

    metrics = report["recordings"]["a"]["test_metrics"]
    assert metrics["false_positives"] == 1
    assert metrics["predicted_anomalous_endpoints"] == 1
    assert metrics["false_positive_rate"] == pytest.approx(1 / 3)
    assert metrics["auc_roc"] is None
    assert metrics["average_precision"] is None
    assert metrics["anomalous_score_distribution"] is None
    assert report["macro_recording_auc_roc"] is None
    assert report["macro_recording_average_precision"] is None


def test_report_handles_train_only_and_all_anomaly_test_recordings():
    report = recording_diagnostics(
        train_scores=np.array([0.1, 0.2]),
        train_recording_ids=np.array(["train_only", "faulty"]),
        train_window_labels=np.array([0, 0]),
        train_endpoint_labels=np.array([0, 0]),
        test_scores=np.array([0.9]),
        test_recording_ids=np.array(["faulty"]),
        test_labels=np.array([1]),
        threshold=0.5,
    )

    assert report["recordings"]["train_only"]["test_metrics"] is None
    metrics = report["recordings"]["faulty"]["test_metrics"]
    assert metrics["f1"] == 1.0
    assert metrics["false_positive_rate"] is None
    assert metrics["auc_roc"] is None
    assert metrics["average_precision"] is None


def test_report_rejects_misaligned_training_metadata():
    with pytest.raises(ValueError, match="aligned vectors"):
        recording_diagnostics(
            train_scores=np.array([0.1, 0.2]),
            train_recording_ids=np.array(["a"]),
            train_window_labels=np.array([0, 0]),
            train_endpoint_labels=np.array([0, 0]),
            test_scores=np.array([0.9]),
            test_recording_ids=np.array(["a"]),
            test_labels=np.array([1]),
            threshold=0.5,
        )
