import numpy as np
import pytest

from hydrofed.evaluation import calibrated_recording_scores, fault_event_detection
from hydrofed.experiments.communication import (
    break_even_days,
    federated_bytes,
    raw_skab_bytes,
    streaming_bytes_per_day,
)
from hydrofed.experiments.scenarios import (
    PlantProfile,
    apply_profile,
    recording_series,
    window_series,
)
from hydrofed.experiments.publish import REGIME_EXPERIMENT, model_rows, publish, regime_rows
from hydrofed.experiments.recordings import (
    load_skab_recordings,
    plant_training_windows,
    recording_roles,
)
from hydrofed.experiments.regimes import (
    autoencoder_scores,
    federated_average,
    parameter_bytes,
    train_regime,
)
from hydrofed.experiments.scoring import (
    REPORTED_METRICS,
    RecordingScores,
    alert_metrics,
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


def _metric(mean):
    return {"mean": mean, "sd": 0.0}


def _summary(recall):
    return {"recall": _metric(recall), "false_positive_rate": _metric(0.1)}


def test_regime_rows_flatten_every_regime_score_and_role():
    result = {
        "config": {"rounds": 20},
        "summary": {
            label: {score: {role: _summary(0.5) for role in ("dev", "final")} for score in ("hybrid", "residual")}
            for label in ("federated", "centralised", "federated_5r")
        },
        "communication": {
            "model_bytes": 100, "clients": 5, "streaming_bytes_per_plant_day_1hz": 2.0,
            "raw_skab": {"csv_bytes": 10, "binary_bytes": 5},
            "federated": {"5_rounds": {"total_bytes": 5000}, "20_rounds": {"total_bytes": 20000}},
        },
    }

    rows = regime_rows(result)

    detection = [row for row in rows if row["kind"] == "detection"]
    assert len(detection) == 3 * 2 * 2
    short = next(row for row in detection if row["client_id"] == "federated_5r")["metrics"]
    assert (short["regime"], short["rounds"]) == ("federated", 5)
    central = next(row for row in detection if row["client_id"] == "centralised")["metrics"]
    assert (central["regime"], central["rounds"], central["recall"]["mean"]) == ("centralised", 20, 0.5)
    assert {row["metrics"]["rounds"] for row in rows if row["kind"] == "communication"} == {5, 20}
    raw = next(row for row in rows if row["kind"] == "raw_data")["metrics"]
    assert raw["model_bytes"] == 100 and raw["csv_bytes"] == 10


def test_model_rows_mark_selected_candidates_and_training():
    def candidate(category, training, dev_auc):
        return {
            "category": category, "training": training, "needs_fault_labels": training != "FedAvg",
            "selection_score": "model",
            "summary": {"model": {"dev": {"macro_recording_auc_roc": _metric(dev_auc)}, "final": _summary(0.6)}},
        }

    result = {
        "selected": {"recurrent": "lstm_autoencoder"},
        "candidates": {
            "lstm_autoencoder": candidate("recurrent", "FedAvg", 0.87) | {"parameters": 10},
            "federated_xgboost": candidate("federated_classic", "FedXgbBagging", 0.9) | {"trees": 240},
        },
    }

    rows = model_rows(result)

    candidates = {row["client_id"]: row["metrics"] for row in rows if row["kind"] == "candidate"}
    assert candidates["lstm_autoencoder"]["selected"] is True
    assert candidates["federated_xgboost"]["selected"] is False
    assert candidates["federated_xgboost"]["trees"] == 240
    assert candidates["lstm_autoencoder"]["dev_auc"] == 0.87
    detection = [row for row in rows if row["kind"] == "detection"]
    assert {(row["client_id"], row["metrics"]["role"]) for row in detection} == {
        ("lstm_autoencoder", "dev"), ("lstm_autoencoder", "final"),
        ("federated_xgboost", "dev"), ("federated_xgboost", "final"),
    }


def test_published_comparisons_are_served_by_the_api(tmp_path):
    from fastapi.testclient import TestClient

    from hydrofed.api import create_app
    from hydrofed.auth import UserStore
    from hydrofed.storage import ExperimentStore

    database = tmp_path / "experiments.db"
    rows = [{"kind": "detection", "client_id": "federated", "metrics": {"role": "final", **_summary(0.7)}}]
    run_id = publish(ExperimentStore(database), REGIME_EXPERIMENT, {"rounds": 20}, rows)
    UserStore(database).create("operator@example.com", "Operator", "operator", "correct horse battery")
    client = TestClient(create_app(database))
    token = client.post(
        "/api/auth/login", json={"email": "operator@example.com", "password": "correct horse battery"},
    ).json()["token"]
    client.headers["Authorization"] = f"Bearer {token}"

    runs = client.get("/api/runs", params={"experiment": REGIME_EXPERIMENT}).json()["runs"]
    results = client.get("/api/results", params={"run_id": run_id, "kind": "detection"}).json()["results"]

    assert runs[0]["run_id"] == run_id and runs[0]["status"] == "completed"
    assert results[0]["metrics"]["recall"]["mean"] == 0.7


def test_recording_series_round_trips_stride_one_windows():
    rng = np.random.default_rng(5)
    values = rng.normal(size=(20, 3)).astype(np.float32)
    labels = np.zeros(20, dtype=np.int64)
    labels[12:15] = 1
    windows, endpoints, window_labels, starts = window_series(values, labels, WINDOW)

    rebuilt_values, rebuilt_labels = recording_series(windows, endpoints, window_labels)

    np.testing.assert_array_equal(rebuilt_values, values)
    np.testing.assert_array_equal(rebuilt_labels, labels)
    assert window_labels.tolist() == [int(labels[s:s + WINDOW].any()) for s in starts]


def test_apply_profile_logs_slower_and_coarser_without_moving_labels():
    values = np.arange(12, dtype=float).reshape(6, 2)
    labels = np.array([0, 0, 1, 1, 0, 0])
    mean, sd = np.zeros(2), np.ones(2)

    slow, slow_labels = apply_profile(values, labels, PlantProfile(step_seconds=2), mean, sd, np.random.default_rng(0))
    coarse, _ = apply_profile(values, labels, PlantProfile(resolution=4.0), mean, sd, np.random.default_rng(0))
    shifted, _ = apply_profile(values, labels, PlantProfile(gain=2.0, offset=1.0), mean, sd, np.random.default_rng(0))

    np.testing.assert_array_equal(slow, values[::2])
    assert slow_labels.tolist() == [0, 1, 0]
    assert set(np.unique(coarse)) <= {0.0, 4.0, 8.0, 12.0}
    np.testing.assert_allclose(shifted, 2 * values + 1)


def test_apply_profile_noise_is_repeatable_for_a_seed():
    values, labels = np.zeros((50, 2)), np.zeros(50, dtype=np.int64)
    profile = PlantProfile(noise=0.5)

    first, _ = apply_profile(values, labels, profile, np.zeros(2), np.ones(2), np.random.default_rng(1))
    second, _ = apply_profile(values, labels, profile, np.zeros(2), np.ones(2), np.random.default_rng(1))

    np.testing.assert_array_equal(first, second)
    assert 0.3 < first.std() < 0.7


def test_default_profile_reproduces_partition_windows(partitions):
    plain = load_skab_recordings(partitions, train_size=TRAIN_SIZE)
    profiled = load_skab_recordings(
        partitions, train_size=TRAIN_SIZE, profiles=dict.fromkeys(plain, PlantProfile()),
    )

    for plant in plain:
        for before, after in zip(plain[plant], profiled[plant]):
            np.testing.assert_array_equal(before.windows, after.windows)
            np.testing.assert_array_equal(before.train_mask, after.train_mask)
            np.testing.assert_array_equal(before.test_mask, after.test_mask)
            np.testing.assert_array_equal(before.endpoint_labels, after.endpoint_labels)


def test_plant_scaling_cancels_calibration_gain_and_offset(partitions):
    plain = load_skab_recordings(partitions, train_size=TRAIN_SIZE)
    recalibrated = load_skab_recordings(
        partitions, train_size=TRAIN_SIZE, profiles={"plant_2": PlantProfile(gain=3.0, offset=5.0)},
    )

    for before, after in zip(plain["plant_2"], recalibrated["plant_2"]):
        np.testing.assert_allclose(before.windows, after.windows, atol=1e-4)


def test_slower_logging_keeps_baseline_and_test_boundaries_in_seconds(partitions):
    recording = load_skab_recordings(
        partitions, train_size=TRAIN_SIZE, profiles={"plant_1": PlantProfile(step_seconds=2)},
    )["plant_1"][0]

    starts = np.arange(len(recording.windows)) * 2
    assert recording.step_seconds == 2
    assert len(recording.windows) == LENGTH // 2 - WINDOW + 1
    np.testing.assert_array_equal(recording.train_mask, starts + (WINDOW - 1) * 2 < TRAIN_SIZE)
    np.testing.assert_array_equal(recording.test_mask, starts >= TRAIN_SIZE)


def test_training_seconds_limit_model_data_but_not_the_alert_baseline(partitions):
    full = load_skab_recordings(partitions, train_size=TRAIN_SIZE)
    scarce = load_skab_recordings(partitions, train_size=TRAIN_SIZE, training_seconds={"plant_1": 6})

    for before, after in zip(full["plant_1"], scarce["plant_1"]):
        np.testing.assert_array_equal(before.train_mask, after.train_mask)
        np.testing.assert_array_equal(before.test_mask, after.test_mask)
        assert len(after.fit_windows) == 6 - WINDOW + 1
    assert len(plant_training_windows(scarce)["plant_1"]) < len(plant_training_windows(full)["plant_1"])
    assert len(plant_training_windows(scarce)["plant_2"]) == len(plant_training_windows(full)["plant_2"])


def test_fault_delays_use_each_recordings_logging_interval():
    labels = np.array([0, 1, 1, 1, 0, 1, 1])
    predicted = np.array([0, 0, 1, 1, 0, 0, 1])
    recordings = np.array(["fast"] * 4 + ["slow"] * 3)

    result = fault_event_detection(labels, predicted, recordings, {"fast": 1, "slow": 5})

    assert result["fault_events_detected"] == 2
    assert result["median_detection_delay"] == (1 + 5) / 2


def test_alert_metrics_are_empty_for_an_empty_selection():
    empty = RecordingScores(np.array([]), np.array([], dtype=int), np.array([]), np.array([]))

    assert alert_metrics(empty, threshold=1.0) == {name: None for name in REPORTED_METRICS}


def test_personalised_regimes_move_the_same_bytes_as_fedavg():
    windows = {
        name: np.random.default_rng(i).normal(size=(8, WINDOW, 2)).astype(np.float32)
        for i, name in enumerate(("plant_1", "plant_2"))
    }
    factory = lambda: build_model(2, {"hidden_size": 4, "latent_size": 2, "num_layers": 1})

    fedavg = train_regime("federated", windows, factory, rounds=2, epochs=1, batch_size=4)
    fedprox = train_regime("fedprox", windows, factory, rounds=2, epochs=1, batch_size=4)
    tuned = train_regime("federated_finetune", windows, factory, rounds=2, epochs=1, batch_size=4)

    for result in (fedprox, tuned):
        assert (result.uplink_bytes, result.downlink_bytes) == (fedavg.uplink_bytes, fedavg.downlink_bytes)
    assert fedprox.models["plant_1"] is fedprox.models["plant_2"]


def test_fine_tuning_gives_each_plant_its_own_model_from_the_global_one():
    windows = {
        name: np.random.default_rng(i).normal(size=(8, WINDOW, 2)).astype(np.float32)
        for i, name in enumerate(("plant_1", "plant_2"))
    }
    factory = lambda: build_model(2, {"hidden_size": 4, "latent_size": 2, "num_layers": 1})

    fedavg = train_regime("federated", windows, factory, rounds=2, epochs=1, batch_size=4)
    tuned = train_regime("federated_finetune", windows, factory, rounds=2, epochs=1, batch_size=4)

    global_weights = fedavg.models["plant_1"].get_parameters_numpy()
    plant_1 = tuned.models["plant_1"].get_parameters_numpy()
    plant_2 = tuned.models["plant_2"].get_parameters_numpy()
    assert tuned.models["plant_1"] is not tuned.models["plant_2"]
    assert any(not np.array_equal(a, b) for a, b in zip(plant_1, global_weights))
    assert any(not np.array_equal(a, b) for a, b in zip(plant_1, plant_2))
