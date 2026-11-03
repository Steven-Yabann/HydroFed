import numpy as np
import pytest

from hydrofed.evaluation import calibrated_recording_scores, fault_event_detection
from hydrofed.experiments.communication import (
    break_even_days,
    federated_bytes,
    raw_skab_bytes,
    streaming_bytes_per_day,
)
from hydrofed.experiments.recordings import (
    load_skab_recordings,
    plant_training_windows,
    recording_roles,
)
from hydrofed.experiments.regimes import federated_average, parameter_bytes, train_regime
from hydrofed.experiments.scoring import (
    RecordingScores,
    alert_metrics,
    autoencoder_scores,
    choose_threshold,
    summarise_seeds,
)
from hydrofed.models.lstm_autoencoder import build_model

WINDOW = 4
LENGTH = 30
TRAIN_SIZE = 12


def _save_plant(path, recordings, offset=0.0):
    windows, labels, ids, endpoints, starts = [], [], [], [], []
    rng = np.random.default_rng(len(recordings))
    for name in recordings:
        values = rng.normal(size=(LENGTH, 2)) + offset
        point_labels = np.zeros(LENGTH, dtype=np.int64)
        if "anomaly-free" not in name:
            point_labels[20:26] = 1
            values[20:26, 1] += 8
        for start in range(LENGTH - WINDOW + 1):
            windows.append(values[start:start + WINDOW])
            labels.append(int(point_labels[start:start + WINDOW].any()))
            ids.append(name)
            endpoints.append(point_labels[start + WINDOW - 1])
            starts.append(start)
    np.savez(
        path, windows=np.asarray(windows, dtype=np.float32), labels=np.asarray(labels),
        recording_ids=np.asarray(ids), endpoint_labels=np.asarray(endpoints),
        window_start_indices=np.asarray(starts),
    )


@pytest.fixture
def partitions(tmp_path):
    _save_plant(tmp_path / "plant_1.npz", ["d/other/1.csv", "d/anomaly-free/a.csv", "d/valve1/2.csv"])
    _save_plant(tmp_path / "plant_2.npz", ["d/valve1/3.csv", "d/other/4.csv"], offset=50.0)
    return tmp_path


def test_recording_roles_alternate_in_recording_order():
    ids = np.array(["b", "b", "a", "c", "a", "d"])

    assert recording_roles(ids) == {"b": "dev", "a": "final", "c": "dev", "d": "final"}


def test_skab_recordings_split_baseline_from_test(partitions):
    recordings = load_skab_recordings(partitions, train_size=TRAIN_SIZE)
    first = recordings["plant_1"][0]

    starts = np.arange(LENGTH - WINDOW + 1)
    assert first.role == "dev" and first.fault_type == "other"
    np.testing.assert_array_equal(first.train_mask, starts + WINDOW - 1 < TRAIN_SIZE)
    np.testing.assert_array_equal(first.test_mask, starts >= TRAIN_SIZE)
    assert first.test_labels.sum() == 6


def test_skab_recordings_train_on_anomaly_free_data_without_testing_it(partitions):
    free = load_skab_recordings(partitions, train_size=TRAIN_SIZE)["plant_1"][1]

    assert free.train_mask.any()
    assert not free.test_mask.any()


def test_plant_scaling_uses_each_plants_baseline(partitions):
    plant = load_skab_recordings(partitions, "plant", TRAIN_SIZE)
    shared = load_skab_recordings(partitions, "shared", TRAIN_SIZE)

    for name in plant:
        baseline = plant_training_windows(plant)[name]
        assert abs(baseline.mean()) < 0.1
    # One shared scaler leaves the offset between plants in the inputs.
    assert plant_training_windows(shared)["plant_2"].mean() > 0.5


def test_federated_average_weights_clients_by_examples():
    averaged = federated_average([[np.array([0.0])], [np.array([4.0])]], [3, 1])

    assert averaged[0][0] == 1.0


@pytest.mark.parametrize("regime", ["federated", "centralised", "isolated"])
def test_regimes_return_one_model_per_plant(regime):
    windows = {
        "plant_1": np.random.default_rng(0).normal(size=(8, WINDOW, 2)).astype(np.float32),
        "plant_2": np.random.default_rng(1).normal(size=(6, WINDOW, 2)).astype(np.float32),
    }
    config = {"hidden_size": 4, "latent_size": 2, "num_layers": 1}

    result = train_regime(regime, windows, lambda: build_model(2, config), rounds=2, epochs=1, batch_size=4)

    assert set(result.models) == set(windows)
    if regime == "isolated":
        assert result.models["plant_1"] is not result.models["plant_2"]
    else:
        assert result.models["plant_1"] is result.models["plant_2"]


def test_only_federated_training_moves_weights():
    windows = {name: np.zeros((4, WINDOW, 2), dtype=np.float32) for name in ("a", "b", "c")}
    factory = lambda: build_model(2, {"hidden_size": 4, "latent_size": 2, "num_layers": 1})
    model_bytes = parameter_bytes(factory().get_parameters_numpy())

    federated = train_regime("federated", windows, factory, rounds=2, epochs=1)
    centralised = train_regime("centralised", windows, factory, rounds=2, epochs=1)

    expected = federated_bytes(model_bytes, clients=3, rounds=2)
    assert federated.uplink_bytes == expected["uplink_bytes"]
    assert federated.downlink_bytes == expected["downlink_bytes"]
    assert sum(federated.round_bytes) == expected["total_bytes"]
    assert centralised.uplink_bytes == centralised.downlink_bytes == 0


def test_federated_bytes_count_both_directions():
    traffic = federated_bytes(model_bytes=100, clients=5, rounds=20)

    assert traffic["total_bytes"] == 2 * 100 * 5 * 20
    assert traffic["bytes_per_client"] == 2 * 100 * 20


def test_break_even_days_compare_training_with_streaming():
    one_day = streaming_bytes_per_day(sensors=8, sample_rate_hz=1.0)

    assert one_day == 8 * 4 * 86_400
    assert break_even_days(int(3 * one_day), sensors=8) == pytest.approx(3.0)


def test_raw_skab_bytes_measure_files_rows_and_baseline(tmp_path):
    folder = tmp_path / "data"
    folder.mkdir()
    for name, rows in (("1.csv", 500), ("2.csv", 300)):
        lines = ["datetime;a;b;anomaly"] + [f"2020-01-01 00:00:{i % 60:02d};1.0;2.0;0" for i in range(rows)]
        (folder / name).write_text("\n".join(lines))

    raw = raw_skab_bytes(tmp_path, train_size=400)

    assert raw["csv_files"] == 2 and raw["rows"] == 800 and raw["sensors"] == 2
    assert raw["binary_bytes"] == 800 * 2 * 4
    assert raw["baseline_binary_bytes"] == (400 + 300) * 2 * 4


def test_calibrated_scores_accept_one_dimensional_model_scores():
    rng = np.random.default_rng(3)
    train, test = rng.normal(size=200), rng.normal(size=50)
    test[25:] += 40

    scores = calibrated_recording_scores(
        [train], np.repeat("a", 200), [test], np.repeat("a", 50), smoothing=1.0,
    )

    assert (scores[25:] > 10).all()
    assert np.median(scores[:25]) < 3


def test_calibrated_scores_are_causal():
    rng = np.random.default_rng(4)
    train, test = rng.normal(size=(200, 2)), rng.normal(size=(50, 2))
    faulty = test.copy()
    faulty[30:, 0] += 20
    ids = np.repeat("a", 200), np.repeat("a", 50)

    quiet = calibrated_recording_scores([train], ids[0], [test], ids[1])
    alerted = calibrated_recording_scores([train], ids[0], [faulty], ids[1])

    np.testing.assert_array_equal(quiet[:30], alerted[:30])


def test_fault_event_detection_reports_detections_and_delay():
    labels = np.array([0, 1, 1, 1, 0, 0, 1, 1])
    predicted = np.array([1, 0, 0, 1, 0, 0, 0, 0])
    recordings = np.array(["a"] * 5 + ["b"] * 3)

    result = fault_event_detection(labels, predicted, recordings)

    assert result == {"fault_events": 2, "fault_events_detected": 1, "median_detection_delay": 2.0}


def test_choose_threshold_takes_highest_recall_within_alarm_budget():
    scores = RecordingScores(
        scores=np.array([0.0, 1.0, 2.0, 3.0, 1.5, 2.5, 3.5, 4.0]),
        labels=np.array([0, 0, 0, 0, 1, 1, 1, 1]),
        recording_ids=np.repeat("a", 8),
        roles=np.repeat("dev", 8),
    )

    # Threshold 2.0 alerts on one of four normal points (25%) and on all faults.
    assert choose_threshold(scores, np.array([0.5, 2.0, 3.0]), max_false_positive_rate=0.25) == 2.0
    assert choose_threshold(scores, np.array([0.5, 2.0, 3.0]), max_false_positive_rate=0.0) == 3.0


def test_autoencoder_scores_cover_tested_recordings_of_both_roles(partitions):
    recordings = load_skab_recordings(partitions, train_size=TRAIN_SIZE)
    model = build_model(2, {"hidden_size": 4, "latent_size": 2, "num_layers": 1})

    scores = autoencoder_scores(recordings, dict.fromkeys(recordings, model), "hybrid")

    assert set(scores.roles) == {"dev", "final"}
    assert not any("anomaly-free" in name for name in scores.recording_ids)
    metrics = alert_metrics(scores.select("dev"), threshold=24.0)
    assert metrics["fault_events"] == 3


def test_summarise_seeds_skips_unavailable_values():
    summary = summarise_seeds([{"recall": 0.5, "delay": None}, {"recall": 0.7, "delay": 4.0}])

    assert summary["recall"]["mean"] == pytest.approx(0.6)
    assert summary["delay"] == {"mean": 4.0, "sd": 0.0}
