import numpy as np
import pytest

from hydrofed.experiments.connectivity import (
    connectivity_scenarios,
    full_schedule,
    outage_schedule,
    sampled_schedule,
)
from hydrofed.experiments.regimes import parameter_bytes, train_regime
from hydrofed.models.lstm_autoencoder import build_model

PLANTS = ["plant_1", "plant_2", "plant_3", "plant_4", "plant_5"]


def _windows():
    return {
        plant: np.random.default_rng(index).normal(size=(8, 4, 2)).astype(np.float32)
        for index, plant in enumerate(PLANTS[:3])
    }


def _factory():
    return build_model(2, {"hidden_size": 4, "latent_size": 2, "num_layers": 1})


def test_sampled_schedule_picks_a_fixed_number_of_distinct_plants_each_round():
    schedule = sampled_schedule(PLANTS, rounds=20, per_round=3, seed=7)

    assert len(schedule) == 20
    assert all(len(set(round_)) == 3 and set(round_) <= set(PLANTS) for round_ in schedule)
    assert schedule == sampled_schedule(PLANTS, rounds=20, per_round=3, seed=7)
    assert len({tuple(round_) for round_ in schedule}) > 1
    with pytest.raises(ValueError):
        sampled_schedule(PLANTS, rounds=2, per_round=6, seed=0)


def test_outage_schedule_keeps_plants_out_until_they_reconnect():
    schedule = outage_schedule(PLANTS, rounds=4, offline=("plant_4", "plant_5"), first_online_round=3)

    assert schedule[:3] == [PLANTS[:3]] * 3
    assert schedule[3] == PLANTS
    assert all("plant_5" not in round_ for round_ in outage_schedule(PLANTS, 4, ("plant_5",), 4))


def test_connectivity_scenarios_cover_every_fixed_case():
    scenarios = connectivity_scenarios(PLANTS, rounds=20, seed=0)

    assert list(scenarios) == [
        "full", "sampled_4_of_5", "sampled_3_of_5", "sampled_2_of_5", "late_join", "never_join",
    ]
    assert scenarios["full"] == full_schedule(PLANTS, 20)
    assert sum("plant_4" in round_ for round_ in scenarios["late_join"]) == 5


def test_full_participation_reproduces_plain_fedavg():
    windows = _windows()
    plain = train_regime("federated", windows, _factory, rounds=2, epochs=1, batch_size=4)
    scheduled = train_regime(
        "federated", windows, _factory, rounds=2, epochs=1, batch_size=4,
        participation=full_schedule(list(windows), 2),
    )

    for left, right in zip(plain.models["plant_1"].get_parameters_numpy(),
                           scheduled.models["plant_1"].get_parameters_numpy()):
        np.testing.assert_array_equal(left, right)
    assert (plain.uplink_bytes, plain.downlink_bytes) == (scheduled.uplink_bytes, scheduled.downlink_bytes)


def test_only_connected_plants_train_and_move_bytes():
    windows = _windows()
    model_bytes = parameter_bytes(_factory().get_parameters_numpy())

    result = train_regime(
        "federated", windows, _factory, rounds=3, epochs=1, batch_size=4,
        participation=[["plant_1"], ["plant_1", "plant_2"], []],
    )

    assert result.uplink_bytes == result.downlink_bytes == 3 * model_bytes
    assert result.round_bytes == [2 * model_bytes, 4 * model_bytes, 0]
    assert np.isnan(result.loss_history["global"][2])
    # Every plant, connected or not, receives the final shared model.
    assert result.models["plant_3"] is result.models["plant_1"]


def test_a_plant_that_never_connects_leaves_the_model_unchanged_by_its_data():
    windows = _windows()
    without = train_regime(
        "federated", {plant: windows[plant] for plant in ("plant_1", "plant_2")}, _factory,
        rounds=2, epochs=1, batch_size=4,
    )
    absent = train_regime(
        "federated", windows, _factory, rounds=2, epochs=1, batch_size=4,
        participation=[["plant_1", "plant_2"]] * 2,
    )

    for left, right in zip(without.models["plant_1"].get_parameters_numpy(),
                           absent.models["plant_1"].get_parameters_numpy()):
        np.testing.assert_array_equal(left, right)


def test_participation_is_refused_for_regimes_that_are_not_federated():
    with pytest.raises(ValueError, match="federated"):
        train_regime("isolated", _windows(), _factory, rounds=1, participation=[["plant_1"]])
    with pytest.raises(ValueError, match="one list"):
        train_regime("federated", _windows(), _factory, rounds=2, participation=[["plant_1"]])
