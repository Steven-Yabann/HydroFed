import pytest
import torch

from hydrofed.models.lstm_autoencoder import LSTMAutoencoder, build_model


def test_lstm_autoencoder_preserves_input_shape():
    model = LSTMAutoencoder(n_features=8)

    batch = torch.randn(4, 32, 8)

    reconstruction = model(batch)

    assert reconstruction.shape == batch.shape


def test_reconstruction_errors_returns_one_error_per_window():
    model = LSTMAutoencoder(n_features=8)

    batch = torch.randn(4, 32, 8)

    errors = model.reconstruction_errors(batch)

    assert errors.shape == (4,)
    assert torch.all(errors >= 0)


def test_model_rejects_input_with_wrong_number_of_dimensions():
    model = LSTMAutoencoder(n_features=8)

    bad_batch = torch.randn(32, 8)

    with pytest.raises(ValueError, match="Expected batch"):
        model(bad_batch)


def test_model_rejects_input_with_wrong_number_of_features():
    model = LSTMAutoencoder(n_features=8)

    bad_batch = torch.randn(4, 32, 6)

    with pytest.raises(ValueError, match="Expected 8 features"):
        model(bad_batch)


def test_model_parameters_can_round_trip_through_numpy():
    model = LSTMAutoencoder(n_features=8)

    parameters = model.get_parameters_numpy()

    new_model = LSTMAutoencoder(n_features=8)
    new_model.set_parameters_numpy(parameters)

    for original, copied in zip(model.parameters(), new_model.parameters()):
        assert torch.allclose(original, copied)


def test_setting_wrong_number_of_parameter_arrays_fails_clearly():
    model = LSTMAutoencoder(n_features=8)

    with pytest.raises(ValueError, match="Expected"):
        model.set_parameters_numpy([])


def test_count_trainable_parameters_is_positive():
    model = LSTMAutoencoder(n_features=8)

    assert model.count_trainable_parameters() > 0


def test_build_model_uses_config_values():
    model = build_model(
        n_features=8,
        model_config={
            "hidden_size": 32,
            "latent_size": 12,
            "num_layers": 1,
            "dropout": 0.0,
        },
    )

    assert model.hidden_size == 32
    assert model.latent_size == 12
    assert model.num_layers == 1