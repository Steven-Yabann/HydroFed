import numpy as np
import torch

from hydrofed.federation.checkpoint import save_global_checkpoint
from hydrofed.models.lstm_autoencoder import build_model
from hydrofed.federation.client import HydroFedClient
from hydrofed.data.pipeline import save_partitions
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
    )

    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    restored = build_model(
        n_features=saved["n_features"],
        model_config=saved["model_config"],
    )
    restored.load_state_dict(saved["state_dict"])

    for expected, actual in zip(original.parameters(), restored.parameters()):
        assert torch.allclose(expected, actual)