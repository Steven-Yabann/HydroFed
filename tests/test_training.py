import numpy as np
import pytest
import torch

from hydrofed.models.lstm_autoencoder import LSTMAutoencoder
from hydrofed.training.local import (
    PlantDataset,
    calibrate_threshold,
    reconstruction_absolute_errors,
    reconstruction_errors,
    train_epochs,
)


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


def test_train_epochs_rejects_non_positive_learning_rate():
    windows = np.random.randn(10, 16, 4).astype(np.float32)
    model = LSTMAutoencoder(n_features=4)

    with pytest.raises(ValueError, match="learning_rate must be positive"):
        train_epochs(model, windows, learning_rate=0)

def test_scoring_and_thresholding():
    windows=np.random.randn(9,8,2).astype(np.float32); model=LSTMAutoencoder(2,4,2,1,0)
    scores=reconstruction_errors(model,windows,batch_size=4)
    assert scores.shape==(9,) and np.all(scores>=0)
    assert np.isfinite(calibrate_threshold(scores))


def test_skab_absolute_scores_return_one_finite_score_per_window():
    windows = np.random.randn(5, 8, 2).astype(np.float32)
    model = LSTMAutoencoder(2, 4, 2, 1, 0)

    scores = reconstruction_absolute_errors(model, windows, batch_size=2)

    assert scores.shape == (5,)
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0)


def _proximal_run(mu, anchor=None):
    torch.manual_seed(0)
    model = LSTMAutoencoder(n_features=2, hidden_size=4, latent_size=2, num_layers=1)
    start = [array.copy() for array in model.get_parameters_numpy()] if anchor is None else anchor
    windows = np.random.default_rng(0).normal(size=(16, 6, 2)).astype(np.float32)
    torch.manual_seed(1)
    train_epochs(model, windows, epochs=3, batch_size=4, learning_rate=0.05,
                 proximal_weights=start, proximal_mu=mu)
    return start, model.get_parameters_numpy()


def test_zero_proximal_mu_trains_exactly_like_plain_training():
    _, plain = _proximal_run(0.0)
    torch.manual_seed(0)
    model = LSTMAutoencoder(n_features=2, hidden_size=4, latent_size=2, num_layers=1)
    windows = np.random.default_rng(0).normal(size=(16, 6, 2)).astype(np.float32)
    torch.manual_seed(1)
    train_epochs(model, windows, epochs=3, batch_size=4, learning_rate=0.05)

    for left, right in zip(plain, model.get_parameters_numpy()):
        np.testing.assert_array_equal(left, right)


def test_proximal_anchor_does_not_move_with_the_model():
    torch.manual_seed(0)
    model = LSTMAutoencoder(n_features=2, hidden_size=4, latent_size=2, num_layers=1)
    start = [array.copy() for array in model.get_parameters_numpy()]
    windows = np.random.default_rng(0).normal(size=(16, 6, 2)).astype(np.float32)

    # The live arrays share memory with the weights, as Flower clients receive them.
    train_epochs(model, windows, epochs=3, batch_size=4, learning_rate=0.05,
                 proximal_weights=model.get_parameters_numpy(), proximal_mu=100.0)
    moved = sum(float(((a - b) ** 2).sum()) for a, b in zip(start, model.get_parameters_numpy()))

    _, free = _proximal_run(0.0)
    free_moved = sum(float(((a - b) ** 2).sum()) for a, b in zip(start, free))
    assert moved < free_moved


def test_proximal_term_keeps_weights_near_the_global_model():
    def distance(start, end):
        return sum(float(((a - b) ** 2).sum()) for a, b in zip(start, end))

    start, loose = _proximal_run(0.0)
    _, tight = _proximal_run(10.0)

    assert distance(start, tight) < distance(start, loose)


def test_proximal_mu_must_not_be_negative():
    model = LSTMAutoencoder(n_features=2, hidden_size=4, latent_size=2, num_layers=1)
    windows = np.zeros((4, 6, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="proximal_mu"):
        train_epochs(model, windows, proximal_weights=model.get_parameters_numpy(), proximal_mu=-1.0)
