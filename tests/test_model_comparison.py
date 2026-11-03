import json
import subprocess
import sys

import numpy as np
import pytest
import torch

from hydrofed.experiments.classic import (
    isolation_forest_scores,
    labelled_windows,
    recording_features,
    supervised_cross_validated_scores,
    supervised_scores,
    window_features,
)
from hydrofed.experiments.recordings import SkabRecording
from hydrofed.experiments.regimes import train_regime
from hydrofed.models.lstm_autoencoder import build_model
from hydrofed.models.transformer_autoencoder import TransformerAutoencoder, build_transformer_model
from hydrofed.training.local import reconstruction_signed_residuals

WINDOW = 5


def _recording(name, role, seed, fault=True, offset=0.0):
    rng = np.random.default_rng(seed)
    length = 80
    values = rng.normal(size=(length, 2)) + offset
    labels = np.zeros(length, dtype=np.int64)
    if fault:
        labels[50:70] = 1
        values[50:70, 0] += 6
    starts = np.arange(length - WINDOW + 1)
    windows = np.stack([values[start:start + WINDOW] for start in starts]).astype(np.float32)
    endpoints = labels[starts + WINDOW - 1]
    return SkabRecording(
        plant="plant_1", recording_id=name, role=role, windows=windows,
        endpoint_labels=endpoints,
        window_labels=np.array([labels[s:s + WINDOW].any() for s in starts], dtype=np.int64),
        train_mask=starts + WINDOW - 1 < 30, test_mask=starts >= 30,
    )


@pytest.fixture
def recordings():
    return [
        _recording(f"d/{kind}/{index}.csv", "dev" if index % 2 == 0 else "final", index, offset=index * 10.0)
        for index, kind in enumerate(["valve1", "valve1", "other", "other", "valve1", "other", "valve1", "other"])
    ]


def test_transformer_reconstructs_windows_with_the_same_shape():
    model = TransformerAutoencoder(n_features=3, d_model=8, num_heads=2, feedforward_size=16)
    batch = torch.randn(4, 10, 3)

    assert model(batch).shape == batch.shape
    assert model.reconstruction_errors(batch).shape == (4,)


def test_transformer_rejects_wrong_feature_count():
    model = TransformerAutoencoder(n_features=3, d_model=8, num_heads=2)

    with pytest.raises(ValueError, match="Expected batch"):
        model(torch.randn(2, 10, 4))


def test_transformer_weights_round_trip_without_position_encodings():
    first = build_transformer_model(3, {"d_model": 8, "num_heads": 2, "feedforward_size": 16})
    second = build_transformer_model(3, {"d_model": 8, "num_heads": 2, "feedforward_size": 16})

    second.set_parameters_numpy(first.get_parameters_numpy())

    assert "positions" not in first.state_dict()
    for left, right in zip(first.parameters(), second.parameters()):
        assert torch.equal(left, right)


def test_transformer_trains_under_fedavg():
    windows = {name: np.random.default_rng(i).normal(size=(8, 6, 2)).astype(np.float32) for i, name in enumerate("ab")}
    config = {"d_model": 8, "num_heads": 2, "feedforward_size": 16, "num_layers": 1}

    result = train_regime("federated", windows, lambda: build_transformer_model(2, config), rounds=2, epochs=1)

    residuals = reconstruction_signed_residuals(result.models["a"], windows["a"])
    assert residuals.shape == (8, 2) and np.isfinite(residuals).all()
    assert result.uplink_bytes > 0


def test_lstm_keeps_numpy_weight_exchange():
    model = build_model(2, {"hidden_size": 4, "latent_size": 2, "num_layers": 1})

    model.set_parameters_numpy(model.get_parameters_numpy())

    assert model.count_trainable_parameters() > 0


def test_window_features_summarise_each_sensor():
    windows = np.array([[[1.0, 10.0], [3.0, 10.0]]])

    features = window_features(windows)

    # mean, sd, min, max, last, change for each of two sensors
    np.testing.assert_allclose(features[0], [2, 10, 1, 0, 1, 10, 3, 10, 3, 10, 2, 0])


def test_recording_features_remove_offsets_between_recordings():
    quiet = _recording("d/valve1/0.csv", "dev", 1, offset=0.0)
    shifted = _recording("d/valve1/1.csv", "dev", 1, offset=100.0)

    np.testing.assert_allclose(
        recording_features(quiet, quiet.test_windows),
        recording_features(shifted, shifted.test_windows),
        atol=1e-3,
    )


def test_labelled_windows_treat_baseline_as_normal_and_skip_the_boundary(recordings):
    features, labels, groups = labelled_windows(recordings[:1])
    recording = recordings[0]

    assert len(labels) == recording.train_mask.sum() + recording.test_mask.sum()
    assert labels.sum() == recording.test_labels.sum()
    assert set(groups) == {recording.recording_id}


def test_random_forest_learns_a_repeated_fault(recordings):
    dev = [recording for recording in recordings if recording.role == "dev"]
    final = [recording for recording in recordings if recording.role == "final"]

    scores = supervised_scores("random_forest", dev, final, seed=0)

    assert set(scores.roles) == {"final"}
    assert scores.scores[scores.labels == 1].mean() > scores.scores[scores.labels == 0].mean() + 0.5


def test_classic_models_import_without_pytorch():
    # XGBoost crashes on macOS when PyTorch's OpenMP runtime is loaded.
    code = "import sys, hydrofed.experiments.classic; sys.exit('torch' in sys.modules)"

    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_classic_runner_evaluates_every_model_in_its_own_process(tmp_path):
    pytest.importorskip("xgboost")
    partitions = tmp_path / "partitions"
    partitions.mkdir()
    rng = np.random.default_rng(0)
    windows, labels, ids, endpoints, starts = [], [], [], [], []
    for index, kind in enumerate(["valve1", "other"] * 4):
        values = rng.normal(size=(80, 2))
        point_labels = np.zeros(80, dtype=np.int64)
        point_labels[50:70] = 1
        values[50:70, 0] += 6
        for start in range(80 - WINDOW + 1):
            windows.append(values[start:start + WINDOW])
            labels.append(int(point_labels[start:start + WINDOW].any()))
            ids.append(f"d/{kind}/{index}.csv")
            endpoints.append(point_labels[start + WINDOW - 1])
            starts.append(start)
    np.savez(
        partitions / "plant_1.npz", windows=np.asarray(windows, dtype=np.float32),
        labels=np.asarray(labels), recording_ids=np.asarray(ids),
        endpoint_labels=np.asarray(endpoints), window_start_indices=np.asarray(starts),
    )
    output = tmp_path / "classic.json"

    subprocess.run([
        sys.executable, "-m", "hydrofed.experiments.classic", "--partitions", str(partitions),
        "--seeds", "0", "--train-size", "30", "--output", str(output),
    ], check=True)

    results = json.loads(output.read_text())
    assert set(results) == {"random_forest", "xgboost", "federated_xgboost", "isolation_forest"}
    for name in ("random_forest", "xgboost", "federated_xgboost"):
        assert results[name][0]["final"]["macro_recording_auc_roc"] > 0.8
        assert "final_unseen_fault_type" in results[name][0]
    assert "final_unseen_fault_type" not in results["isolation_forest"][0]
    assert results["federated_xgboost"][0]["training"]["uplink_bytes"] > 0
    assert "training" not in results["xgboost"][0]


FEDERATED_XGBOOST_CHECK = """
import json
import numpy as np
from hydrofed.experiments.classic import train_federated_xgboost
import xgboost as xgb

rng = np.random.default_rng(0)
plants = {}
for index in range(3):
    features = rng.normal(size=(300, 4))
    plants[f"plant_{index}"] = (features, (features[:, 0] > 0.5).astype(int))
model, cost = train_federated_xgboost(plants, seed=0, rounds=4)
test = rng.normal(size=(500, 4))
accuracy = float(((model.predict(xgb.DMatrix(test)) > 0.5) == (test[:, 0] > 0.5)).mean())
print(json.dumps(cost | {"accuracy": accuracy}))
"""


def test_federated_xgboost_appends_one_tree_per_plant_per_round():
    pytest.importorskip("xgboost")

    output = subprocess.run(
        [sys.executable, "-c", FEDERATED_XGBOOST_CHECK], capture_output=True, text=True, check=True,
    ).stdout
    cost = json.loads(output.strip().splitlines()[-1])

    assert cost["clients"] == 3 and cost["rounds"] == 4
    assert cost["trees"] == 3 * 4
    # Plants download the shared model from round two onward and upload one tree every round.
    assert 0 < cost["uplink_bytes"] < cost["downlink_bytes"]
    assert cost["model_bytes"] < cost["downlink_bytes"]
    assert cost["accuracy"] > 0.9


def test_cross_validation_never_scores_a_recording_it_trained_on(recordings, monkeypatch):
    dev = [recording for recording in recordings if recording.role == "dev"]
    seen = []
    original = supervised_scores

    def tracking(name, train, test, seed):
        seen.append(({r.recording_id for r in train}, {r.recording_id for r in test}))
        return original(name, train, test, seed)

    monkeypatch.setattr("hydrofed.experiments.classic.supervised_scores", tracking)
    scores = supervised_cross_validated_scores("random_forest", dev, seed=0, folds=2)

    assert all(not train & test for train, test in seen)
    assert set(scores.recording_ids) == {r.recording_id for r in dev}


def test_isolation_forest_scores_faults_above_normal_windows(recordings):
    scores = isolation_forest_scores(recordings, seed=0)

    assert np.median(scores.scores[scores.labels == 1]) > np.median(scores.scores[scores.labels == 0])
