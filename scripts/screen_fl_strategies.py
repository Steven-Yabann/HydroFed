#!/usr/bin/env python3
"""Screen robust, proximal, personalised and clustered FL on SKAB.

The screen behind "Screened and not added" in docs/comparison.md. FedProx and
fine-tuning were then added to scripts/run_comparison.py as regimes; the
median, trimmed-mean and clustered strategies stay here only. Mirrors
run_fedavg's seeding so FedAvg reproduces the scenarios run. Settings are fixed
in advance, not tuned:
  median / trimmed mean (drop 1 of 5 per coordinate each side), FedProx mu=0.1,
  fine-tuning 5 local epochs after FedAvg, clustered FL: 2 clusters from
  cosine similarity of client updates after 3 warm-up rounds.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from hydrofed.experiments.recordings import ROLES, load_skab_recordings, plant_training_windows
from hydrofed.experiments.regimes import autoencoder_scores, federated_average
from hydrofed.experiments.scenarios import HETEROGENEOUS_PROFILES
from hydrofed.experiments.scoring import alert_metrics, summarise_seeds
from hydrofed.models.lstm_autoencoder import build_model
from hydrofed.reproducibility import set_random_seed
from hydrofed.training.local import PlantDataset, train_epochs

CONFIG = dict(hidden_size=64, latent_size=4, num_layers=2, dropout=0.1)
ROUNDS, EPOCHS, BATCH, LR = 20, 5, 64, 1e-3
MU, FINETUNE_EPOCHS, WARMUP, CLUSTERS = 0.1, 5, 3, 2
SCENARIOS = {
    "baseline": {},
    "heterogeneous": {"profiles": HETEROGENEOUS_PROFILES},
    "scarce_25s": {"training_seconds": {"plant_4": 25, "plant_5": 25}},
}
METHODS = ("fedavg", "median", "trimmed", "fedprox", "clustered")
GROUPS = {"scarce plants": ["plant_4", "plant_5"], "other plants": ["plant_1", "plant_2", "plant_3"]}


def train_prox(model, windows, global_params, epochs, mu):
    """train_epochs plus FedProx's proximal term mu/2 * ||w - w_global||^2."""
    loader = DataLoader(PlantDataset(windows), batch_size=BATCH, shuffle=True)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    anchors = [torch.as_tensor(p) for p in global_params]
    names = list(model.state_dict().keys())
    params = dict(model.named_parameters())
    for _ in range(epochs):
        for batch in loader:
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(batch), batch)
            prox = sum(((params[n] - a) ** 2).sum() for n, a in zip(names, anchors) if n in params)
            (loss + mu / 2 * prox).backward()
            optimizer.step()


def aggregate(method, results, sizes):
    if method in ("fedavg", "fedprox", "clustered"):
        return federated_average(results, sizes)
    stacked = [np.stack([r[i] for r in results]) for i in range(len(results[0]))]
    if method == "median":
        return [np.median(s, axis=0).astype(s.dtype) for s in stacked]
    if method == "trimmed":
        return [np.sort(s, axis=0)[1:-1].mean(axis=0).astype(s.dtype) for s in stacked]
    raise ValueError(method)


def local_round(state, windows, seed, method):
    set_random_seed(seed)
    model = build_model(windows.shape[2], CONFIG)
    model.set_parameters_numpy(state)
    if method == "fedprox":
        train_prox(model, windows, state, EPOCHS, MU)
    else:
        train_epochs(model, windows, EPOCHS, BATCH, LR)
    return model.get_parameters_numpy()


def run_rounds(state, clients, seed, method, rounds, first_round=0):
    for r in range(first_round, first_round + rounds):
        results = [local_round(state, w, seed + i + r * 100, method) for i, w in clients]
        state = aggregate(method, results, [len(w) for _, w in clients])
    return state


def train(method, plant_windows, seed, rounds=ROUNDS):
    """Return {plant: model} plus extra info for one method."""
    n_features = next(iter(plant_windows.values())).shape[2]
    set_random_seed(seed)
    initial = [p.copy() for p in build_model(n_features, CONFIG).get_parameters_numpy()]
    clients = list(enumerate(plant_windows.values()))
    plants = list(plant_windows)

    def model_from(state):
        model = build_model(n_features, CONFIG)
        model.set_parameters_numpy(state)
        return model

    if method != "clustered":
        state = run_rounds(initial, clients, seed, method, rounds)
        models = {method: dict.fromkeys(plants, model_from(state))}
        if method == "fedavg":
            tuned = {}
            for i, plant in enumerate(plants):
                set_random_seed(seed + 1000 + i)
                model = model_from(state)
                train_epochs(model, plant_windows[plant], FINETUNE_EPOCHS, BATCH, LR)
                tuned[plant] = model
            models["fedavg_finetune"] = tuned
        return models, {}

    state = run_rounds(initial, clients, seed, "fedavg", WARMUP)
    # Clients' next updates from the warm-up model, compared by cosine similarity.
    flat = lambda params: np.concatenate([p.ravel() for p in params])
    updates = np.stack([
        flat(local_round(state, w, seed + i + WARMUP * 100, "fedavg")) - flat(state) for i, w in clients
    ])
    unit = updates / np.linalg.norm(updates, axis=1, keepdims=True)
    from scipy.cluster.hierarchy import fcluster, linkage
    labels = fcluster(linkage(unit, method="average", metric="cosine"), CLUSTERS, criterion="maxclust")
    models = {}
    for cluster in sorted(set(labels)):
        members = [(i, w) for (i, w), label in zip(clients, labels) if label == cluster]
        cluster_state = run_rounds(state, members, seed, "clustered", rounds - WARMUP, WARMUP)
        for i, _ in members:
            models[plants[i]] = model_from(cluster_state)
    return {"clustered": models}, {"clusters": {plants[i]: int(label) for i, label in enumerate(labels)}}


def job(spec):
    torch.set_num_threads(spec["threads"])
    scenario, method, seed = spec["scenario"], spec["method"], spec["seed"]
    recordings = load_skab_recordings(spec["partitions"], "plant", **SCENARIOS[scenario])
    plant_windows = plant_training_windows(recordings)
    seconds = {r.recording_id: r.step_seconds for items in recordings.values() for r in items}
    plant_of = {r.recording_id: p for p, items in recordings.items() for r in items}
    start = time.time()
    models_by_method, info = train(method, plant_windows, seed, spec["rounds"])
    out = []
    for name, models in models_by_method.items():
        metrics = {}
        for score in ("hybrid", "residual"):
            scores = autoencoder_scores(recordings, models, score)
            plants = np.array([plant_of[i] for i in scores.recording_ids])
            metrics[score] = {
                group: {
                    role: alert_metrics(
                        scores.where((np.isin(plants, members) if members else True) & (scores.roles == role)),
                        24.0, 1, seconds,
                    )
                    for role in ROLES
                }
                for group, members in {"all": None, **GROUPS, **{p: [p] for p in recordings}}.items()
            }
        out.append({"scenario": scenario, "method": name, "seed": seed, "metrics": metrics, **info})
    print(f"{scenario}/{method} seed {seed}: {time.time() - start:.0f}s {info}", flush=True)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--partitions", default="data/partitions_skab")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    threads = max(1, (os.cpu_count() or 4) // args.jobs)
    specs = [
        {"scenario": s, "method": m, "seed": seed, "threads": threads, "rounds": args.rounds,
         "partitions": args.partitions}
        for s in args.scenarios for m in args.methods for seed in args.seeds
    ]
    with ProcessPoolExecutor(args.jobs) as pool:
        runs = [row for rows in pool.map(job, specs) for row in rows]
    summary = {}
    for row in runs:
        key = f"{row['scenario']}/{row['method']}"
        summary.setdefault(key, []).append(row)
    result = {
        key: {
            score: {
                group: {role: summarise_seeds([r["metrics"][score][group][role] for r in rows]) for role in ROLES}
                for group in rows[0]["metrics"][score]
            }
            for score in ("hybrid", "residual")
        } | {"clusters": [r.get("clusters") for r in rows]}
        for key, rows in summary.items()
    }
    with open(args.output, "w") as file:
        json.dump({"summary": result, "runs": runs}, file, indent=1)
    print("saved", args.output)
