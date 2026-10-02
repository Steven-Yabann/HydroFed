import numpy as np
import pytest
import torch

from hydrofed.models.lstm_autoencoder import LSTMAutoencoder
from hydrofed.training.local import PlantDataset, train_epochs, reconstruction_errors, calibrate_threshold


def test_plant_dataset_returns_one_window_at_a_time():
    windows = np.random.randn(10, 32, 8).astype(np.float32)

    dataset = PlantDataset(windows)

    assert len(dataset) == 10
    assert dataset[0].shape == (32, 8)


def test_plant_dataset_rejects_wrong_shape():
    windows = np.random.randn(32, 8).astype(np.float32)

    with pytest.raises(ValueError, match="windows must have shape"):
        PlantDataset(windows)


def test_plant_dataset_rejects_empty_windows():
    windows = np.empty((0, 32, 8), dtype=np.float32)

    with pytest.raises(ValueError, match="at least one"):
        PlantDataset(windows)


def test_train_epochs_returns_finite_loss():
    torch.manual_seed(0)

    windows = np.random.randn(20, 16, 4).astype(np.float32)

    model = LSTMAutoencoder(
        n_features=4,
        hidden_size=8,
        latent_size=3,
        num_layers=1,
        dropout=0.0,
    )

    loss = train_epochs(
        model,
        windows,
        epochs=1,
        batch_size=5,
        learning_rate=1e-3,
    )

    assert np.isfinite(loss)
    assert loss >= 0.0


def test_train_epochs_updates_model_parameters():
    torch.manual_seed(0)

    windows = np.random.randn(30, 16, 4).astype(np.float32)

    model = LSTMAutoencoder(
        n_features=4,
        hidden_size=8,
        latent_size=3,
        num_layers=1,
        dropout=0.0,
    )

    before = [parameter.detach().clone() for parameter in model.parameters()]

    train_epochs(
        model,
        windows,
        epochs=2,
        batch_size=10,
        learning_rate=1e-3,
    )

    after = list(model.parameters())

    assert any(
        not torch.allclose(before_parameter, after_parameter)
        for before_parameter, after_parameter in zip(before, after)
    )


def test_train_epochs_rejects_non_positive_epochs():
    windows = np.random.randn(10, 16, 4).astype(np.float32)
    model = LSTMAutoencoder(n_features=4)

    with pytest.raises(ValueError, match="epochs must be positive"):
        train_epochs(model, windows, epochs=0)


def test_train_epochs_rejects_non_positive_batch_size():
    windows = np.random.randn(10, 16, 4).astype(np.float32)
    model = LSTMAutoencoder(n_features=4)

    with pytest.raises(ValueError, match="batch_size must be positive"):
        train_epochs(model, windows, batch_size=0)

def test_scoring_and_thresholding():
    windows=np.random.randn(9,8,2).astype(np.float32); model=LSTMAutoencoder(2,4,2,1,0)
    scores=reconstruction_errors(model,windows,batch_size=4)
    assert scores.shape==(9,) and np.all(scores>=0)
    assert np.isfinite(calibrate_threshold(scores))
