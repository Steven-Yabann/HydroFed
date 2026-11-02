import numpy as np
import torch

from hydrofed.federation.checkpoint import load_global_checkpoint, save_global_checkpoint
from hydrofed.models.lstm_autoencoder import build_model
from hydrofed.federation.client import HydroFedClient
from hydrofed.data.pipeline import save_partitions
from hydrofed.data.scaling import WindowMinMaxScaler
from flwr.common import ndarrays_to_parameters
from hydrofed.federation.strategy import TrackingFedAvg

def test_client_fit_and_evaluate_contract(tmp_path):
    rng=np.random.default_rng(2); windows=rng.normal(size=(12,6,2)).astype(np.float32)
    labels=np.array([0]*9+[1]*3); save_partitions({"plant_1":(windows,labels)},tmp_path)
    client = HydroFedClient(
        "plant_1",
        tmp_path / "plant_1.npz",
        {"hidden_size": 4, "latent_size": 2, "num_layers": 1, "dropout": 0},
        batch_size=4,
        device="cpu",
        seed=7
    )

    assert client.device == "cpu"
    assert client.seed == 7
    parameters=client.get_parameters({}); updated,count,metrics=client.fit(parameters,{"local_epochs":1})
    loss,evaluated,evaluation_metrics=client.evaluate(updated,{})
    assert count==7 and len(updated)==len(parameters) and metrics["client_id"]=="plant_1"
    assert evaluated==3 and loss>=0 and evaluation_metrics["local_loss"]==loss


def test_client_applies_supplied_shared_scaler_to_every_split(tmp_path):
    windows = np.arange(12, dtype=np.float32).reshape(12, 1, 1)
    labels = np.array([0] * 9 + [1] * 3, dtype=np.int64)
    save_partitions({"plant_1": (windows, labels)}, tmp_path)
    scaler = WindowMinMaxScaler.fit(np.array([[[0.0], [10.0]]], dtype=np.float32))

    client = HydroFedClient(
        "plant_1",
        tmp_path / "plant_1.npz",
        {"hidden_size": 4, "latent_size": 2, "num_layers": 1, "dropout": 0},
        batch_size=4,
        device="cpu",
        scaler=scaler,
    )

    assert client.plant_split.train_windows.max() <= 0.7
    assert client.plant_split.test_windows.max() > 1.0


def test_client_builds_skab_split_from_partition_provenance(tmp_path):
    windows = np.arange(24, dtype=np.float32).reshape(12, 2, 1)
    labels = np.zeros(12, dtype=np.int64)
    save_partitions(
        {"plant_1": (windows, labels)},
        tmp_path,
        metadata={
            "plant_1": {
                "recording_ids": np.array(["recording_a"] * 12),
                "endpoint_labels": np.array([0] * 8 + [1] * 4),
                "window_start_indices": np.arange(12, dtype=np.int64),
            }
        },
    )

    client = HydroFedClient(
        "plant_1",
        tmp_path / "plant_1.npz",
        {"hidden_size": 4, "latent_size": 2, "num_layers": 1, "dropout": 0},
        batch_size=4,
        device="cpu",
        evaluation_protocol="skab",
        skab_train_size=4,
    )

    assert len(client.plant_split.train_windows) == 3
    assert len(client.plant_split.test_windows) == 8
    assert client.plant_split.test_labels.sum() == 4
    assert client.plant_split.test_recording_ids.tolist() == ["recording_a"] * 8

def test_strategy_exposes_latest_global_parameters(tmp_path):
    initial = ndarrays_to_parameters(
        [np.array([1.0, 2.0], dtype=np.float32)]
    )

    strategy = TrackingFedAvg(
        db_path=tmp_path / "experiments.db",
        initial_parameters=initial,
    )

    recovered = strategy.get_final_parameters_numpy()

    np.testing.assert_array_equal(
        recovered[0],
        np.array([1.0, 2.0], dtype=np.float32),
    )

def test_global_checkpoint_round_trips_model_weights(tmp_path):
    config = {
        "hidden_size": 4,
        "latent_size": 2,
        "num_layers": 1,
        "dropout": 0.0,
    }

    original = build_model(n_features=2, model_config=config)

    checkpoint_path = save_global_checkpoint(
        original.get_parameters_numpy(),
        n_features=2,
        output_path=tmp_path / "global_model.pt",
        model_config=config,
        preprocessing={"scaler": "shared_normal_train_minmax"},
    )

    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    restored = build_model(
        n_features=saved["n_features"],
        model_config=saved["model_config"],
    )
    restored.load_state_dict(saved["state_dict"])

    assert saved["preprocessing"] == {"scaler": "shared_normal_train_minmax"}

    for expected, actual in zip(original.parameters(), restored.parameters()):
        assert torch.allclose(expected, actual)


def test_checkpoint_serializes_numpy_preprocessing_metadata_safely(tmp_path):
    config = {
        "hidden_size": 4,
        "latent_size": 2,
        "num_layers": 1,
        "dropout": 0.0,
    }
    model = build_model(n_features=2, model_config=config)
    path = save_global_checkpoint(
        model.get_parameters_numpy(),
        n_features=2,
        output_path=tmp_path / "global_model.pt",
        model_config=config,
        preprocessing={"means": np.array([1.0, 2.0], dtype=np.float32)},
    )

    saved = load_global_checkpoint(path)

    assert saved["preprocessing"]["means"] == [1.0, 2.0]


def test_checkpoint_loader_supports_trusted_legacy_numpy_metadata(tmp_path):
    path = tmp_path / "legacy_model.pt"
    torch.save({"preprocessing": {"means": np.array([1.0])}}, path)

    saved = load_global_checkpoint(path)

    np.testing.assert_array_equal(saved["preprocessing"]["means"], [1.0])

def test_strategy_records_round_with_its_run_id(tmp_path, monkeypatch):
    strategy = TrackingFedAvg(
        db_path=tmp_path / "experiments.db",
        run_id="federated-run-123",
    )
    recorded_calls = []

    def capture_record(*args, **kwargs):
        recorded_calls.append((args, kwargs))

    monkeypatch.setattr(strategy.store, "record", capture_record)

    strategy.aggregate_fit(
        server_round=1,
        results=[],
        failures=[],
    )

    assert len(recorded_calls) == 1

    arguments, keyword_arguments = recorded_calls[0]
    assert arguments[0] == "federated"
    assert arguments[1] == "federation_round"
    assert keyword_arguments["round_number"] == 1
    assert keyword_arguments["run_id"] == "federated-run-123"
