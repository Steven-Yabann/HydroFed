"""Train one model architecture under federated, centralised or isolated regimes.

All regimes share one schedule: ``rounds`` passes of ``epochs`` local epochs,
with the optimiser restarted each round as it is on a Flower client. Federated
training runs FedAvg in-process with the same example-weighted average as
Flower's strategy, which is much faster than a Ray simulation for repeated
seeds. Bytes are counted as ``TrackingFedAvg`` counts them: the arrays each
client uploads and the global arrays sent back, without gRPC overhead.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from hydrofed.experiments.recordings import SkabRecording
from hydrofed.experiments.scoring import RecordingScores, calibrate_components, concatenate_scores
from hydrofed.reproducibility import set_random_seed
from hydrofed.training.local import reconstruction_signed_residuals, train_epochs

REGIMES = ("federated", "centralised", "isolated")
SCORES = ("hybrid", "residual")


@dataclass
class RegimeResult:
    """Trained model per plant and what training cost."""

    regime: str
    models: dict
    loss_history: dict[str, list[float]]
    uplink_bytes: int = 0
    downlink_bytes: int = 0
    seconds: float = 0.0
    round_bytes: list[int] = field(default_factory=list)


def parameter_bytes(parameters: list[np.ndarray]) -> int:
    """Return the payload size of a list of parameter arrays."""
    return int(sum(array.nbytes for array in parameters))


def federated_average(
    client_parameters: list[list[np.ndarray]],
    client_sizes: list[int],
) -> list[np.ndarray]:
    """Average client weights in proportion to their training examples."""
    total = sum(client_sizes)
    if not client_parameters or total <= 0:
        raise ValueError("FedAvg needs at least one client with training data")
    return [
        sum(parameters[index] * size for parameters, size in zip(client_parameters, client_sizes)) / total
        for index in range(len(client_parameters[0]))
    ]


def run_fedavg(
    client_windows: dict[str, np.ndarray],
    model_factory: Callable,
    rounds: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: str = "cpu",
):
    """Run FedAvg over the given clients and return the global model.

    Returns the model, the weighted loss per round, the bytes moved per round,
    and the total uplink and downlink bytes.
    """
    if rounds <= 0:
        raise ValueError("rounds must be positive")

    set_random_seed(seed)
    global_model = model_factory()
    state = [array.copy() for array in global_model.get_parameters_numpy()]
    history, round_bytes = [], []
    uplink = downlink = 0

    for round_index in range(rounds):
        results, sizes, losses = [], [], []
        for client_index, windows in enumerate(client_windows.values()):
            set_random_seed(seed + client_index + round_index * 100)
            model = model_factory()
            model.set_parameters_numpy(state)
            downlink += parameter_bytes(state)
            losses.append(train_epochs(model, windows, epochs, batch_size, learning_rate, device=device))
            results.append(model.get_parameters_numpy())
            uplink += parameter_bytes(results[-1])
            sizes.append(len(windows))

        state = federated_average(results, sizes)
        history.append(float(np.average(losses, weights=sizes)))
        round_bytes.append(sum(parameter_bytes(item) for item in results) + parameter_bytes(state) * len(results))

    global_model.set_parameters_numpy(state)
    return global_model, history, round_bytes, uplink, downlink


def train_regime(
    regime: str,
    plant_windows: dict[str, np.ndarray],
    model_factory: Callable,
    rounds: int = 20,
    epochs: int = 5,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    seed: int = 42,
    device: str = "cpu",
) -> RegimeResult:
    """Train under one regime and return the model each plant would use.

    - federated: FedAvg across all plants, one shared global model
    - centralised: all plants' windows pooled at one server, one shared model
    - isolated: each plant trains alone on its own windows

    Only federated training moves model weights. Centralised training instead
    moves raw data, which ``hydrofed.experiments.communication`` accounts for.
    """
    if regime not in REGIMES:
        raise ValueError(f"regime must be one of {REGIMES}")

    start = time.perf_counter()
    options = dict(
        model_factory=model_factory, rounds=rounds, epochs=epochs, batch_size=batch_size,
        learning_rate=learning_rate, seed=seed, device=device,
    )

    if regime == "federated":
        model, history, round_bytes, uplink, downlink = run_fedavg(plant_windows, **options)
        return RegimeResult(
            regime, dict.fromkeys(plant_windows, model), {"global": history},
            uplink, downlink, time.perf_counter() - start, round_bytes,
        )

    if regime == "centralised":
        pooled = {"central": np.concatenate(list(plant_windows.values()))}
        model, history, *_ = run_fedavg(pooled, **options)
        return RegimeResult(
            regime, dict.fromkeys(plant_windows, model), {"central": history},
            seconds=time.perf_counter() - start,
        )

    models, histories = {}, {}
    for plant, windows in plant_windows.items():
        models[plant], histories[plant], *_ = run_fedavg({plant: windows}, **options)
    return RegimeResult(regime, models, histories, seconds=time.perf_counter() - start)


def autoencoder_scores(
    recordings: dict[str, list[SkabRecording]],
    models: dict,
    score: str = "hybrid",
    smoothing: float = 0.2,
    calibration_tail: float = 0.25,
    device: str = "cpu",
) -> RecordingScores:
    """Score every plant's recordings with that plant's autoencoder.

    ``residual`` uses only the model's signed residuals. ``hybrid`` also
    measures how far sensor levels sit from the baseline, which needs no model.
    """
    if score not in SCORES:
        raise ValueError(f"score must be one of {SCORES}")

    def components_for(model):
        def components(recording, windows):
            residuals = reconstruction_signed_residuals(model, windows, device=device)
            return [windows.mean(axis=1), residuals] if score == "hybrid" else [residuals]
        return components

    return concatenate_scores([
        calibrate_components(items, components_for(models[plant]), smoothing, calibration_tail)
        for plant, items in recordings.items()
    ])
