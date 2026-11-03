"""Classic machine learning detectors on SKAB window features.

Each window becomes summary statistics per sensor, standardised against the
recording's own baseline. The autoencoders and the hybrid score get the same
baseline information, so the comparison is about the model, not the inputs.

Random Forest and XGBoost are supervised: they learn from labelled faults on
development recordings. Isolation Forest is unsupervised and, like the
autoencoders, learns only from normal baseline windows.

Tree ensembles cannot be averaged with FedAvg. Federated XGBoost instead uses
Flower's bagging aggregation: every round, each plant grows a tree on its own
data from the current shared model, and the server appends all new trees.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import GroupKFold

from hydrofed.experiments.recordings import SkabRecording
from hydrofed.experiments.scoring import (
    MAD_THRESHOLDS,
    PROBABILITY_THRESHOLDS,
    RecordingScores,
    alert_metrics,
    calibrate_components,
    choose_threshold,
    concatenate_scores,
)

SUPERVISED = ("random_forest", "xgboost", "federated_xgboost")
UNSUPERVISED = ("isolation_forest",)
CLASSIC_MODELS = SUPERVISED + UNSUPERVISED

# Shared by centralised and federated XGBoost so only the training differs.
XGBOOST_SETTINGS = dict(max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)
XGBOOST_TREES = 300
FEDERATED_XGBOOST_ROUNDS = 60


def window_features(windows: np.ndarray) -> np.ndarray:
    """Summarise each window per sensor: mean, sd, min, max, last value, change."""
    windows = np.asarray(windows, dtype=float)
    return np.concatenate([
        windows.mean(axis=1),
        windows.std(axis=1),
        windows.min(axis=1),
        windows.max(axis=1),
        windows[:, -1],
        windows[:, -1] - windows[:, 0],
    ], axis=1)


def recording_features(recording: SkabRecording, windows: np.ndarray) -> np.ndarray:
    """Return window features in standard units of the recording's baseline."""
    baseline = window_features(recording.train_windows)
    spread = baseline.std(axis=0)
    spread = np.where(spread < 1e-6, 1.0, spread)
    return (window_features(windows) - baseline.mean(axis=0)) / spread


def build_classifier(name: str, seed: int):
    """Create a supervised classifier with fixed, untuned settings."""
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample",
            n_jobs=-1, random_state=seed,
        )
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise ImportError('XGBoost needs: pip install -e ".[experiments]"') from error
        return XGBClassifier(
            n_estimators=XGBOOST_TREES, n_jobs=-1, random_state=seed, eval_metric="logloss",
            **XGBOOST_SETTINGS,
        )
    raise ValueError(f"Unknown supervised model {name!r}")


def train_federated_xgboost(
    plant_data: dict[str, tuple[np.ndarray, np.ndarray]],
    seed: int,
    rounds: int = FEDERATED_XGBOOST_ROUNDS,
):
    """Train XGBoost across plants with Flower's FedXgbBagging aggregation.

    Each round, every plant loads the shared model, boosts one tree on its own
    features and labels, and uploads only that tree. The server appends the
    new trees to the shared model and sends it back next round. Raw data never
    leaves a plant. Bytes are the serialised JSON models that Flower would
    exchange: one tree per plant up, the whole shared model down.
    """
    import xgboost as xgb
    from flwr.server.strategy.fedxgb_bagging import aggregate

    if rounds <= 0 or not plant_data:
        raise ValueError("federated XGBoost needs positive rounds and at least one plant")

    params = {
        "objective": "binary:logistic", "eval_metric": "logloss", "base_score": 0.5,
        "eta": XGBOOST_SETTINGS["learning_rate"], "max_depth": XGBOOST_SETTINGS["max_depth"],
        "subsample": XGBOOST_SETTINGS["subsample"],
        "colsample_bytree": XGBOOST_SETTINGS["colsample_bytree"], "nthread": 1,
    }
    matrices = {plant: xgb.DMatrix(features, label=labels) for plant, (features, labels) in plant_data.items()}
    shared, uplink, downlink = None, 0, 0

    for round_index in range(rounds):
        new_trees = []
        for plant_index, matrix in enumerate(matrices.values()):
            local_params = params | {"seed": seed + plant_index + round_index * 100}
            start = None
            if shared is not None:
                downlink += len(shared)
                start = xgb.Booster(params=local_params)
                start.load_model(bytearray(shared))
            booster = xgb.train(local_params, matrix, num_boost_round=1, xgb_model=start)
            trees = booster.num_boosted_rounds()
            new_trees.append(bytes(booster[trees - 1:trees].save_raw("json")))
            uplink += len(new_trees[-1])
        for tree in new_trees:
            shared = aggregate(shared, tree)

    model = xgb.Booster()
    model.load_model(bytearray(shared))
    return model, {
        "clients": len(matrices), "rounds": rounds, "trees": model.num_boosted_rounds(),
        "model_bytes": len(shared), "uplink_bytes": uplink, "downlink_bytes": downlink,
    }


def fit_supervised(name: str, train_recordings: list[SkabRecording], seed: int):
    """Train a supervised model and return its fault-probability function.

    Also returns what training cost: the federated model reports the bytes
    it exchanged. Federated XGBoost groups recordings by plant, and plants
    without labelled windows do not take part.
    """
    if name == "federated_xgboost":
        import xgboost as xgb

        plant_data = {}
        for plant in dict.fromkeys(recording.plant for recording in train_recordings):
            recordings = [
                recording for recording in train_recordings
                if recording.plant == plant and recording.test_mask.any()
            ]
            if recordings:
                features, labels, _ = labelled_windows(recordings)
                plant_data[plant] = (features, labels)
        model, cost = train_federated_xgboost(plant_data, seed)
        return lambda features: model.predict(xgb.DMatrix(features)), cost

    features, labels, _ = labelled_windows(train_recordings)
    model = build_classifier(name, seed).fit(features, labels)
    return lambda features: model.predict_proba(features)[:, 1], {}


def labelled_windows(recordings: list[SkabRecording]):
    """Return features, labels and recording IDs for supervised training.

    Baseline windows count as normal and test windows keep their endpoint
    labels. Windows that overlap the baseline boundary are left out.
    """
    features, labels, groups = [], [], []
    for recording in recordings:
        mask = recording.train_mask | recording.test_mask
        if not recording.test_mask.any():
            continue
        features.append(recording_features(recording, recording.windows[mask]))
        labels.append(np.where(recording.train_mask[mask], 0, recording.endpoint_labels[mask]))
        groups.append(np.repeat(recording.recording_id, mask.sum()))
    return np.concatenate(features), np.concatenate(labels), np.concatenate(groups)


def _scores_from_probabilities(recordings: list[SkabRecording], probability_for) -> RecordingScores:
    tested = [recording for recording in recordings if recording.test_mask.any()]
    return RecordingScores(
        scores=np.concatenate([probability_for(recording) for recording in tested]),
        labels=np.concatenate([recording.test_labels for recording in tested]),
        recording_ids=np.concatenate([
            np.repeat(recording.recording_id, recording.test_mask.sum()) for recording in tested
        ]),
        roles=np.concatenate([np.repeat(recording.role, recording.test_mask.sum()) for recording in tested]),
    )


def supervised_scores(
    name: str,
    train_recordings: list[SkabRecording],
    test_recordings: list[SkabRecording],
    seed: int,
) -> RecordingScores:
    """Train on labelled recordings and return fault probabilities for others."""
    return _supervised_scores_and_cost(name, train_recordings, test_recordings, seed)[0]


def _supervised_scores_and_cost(name, train_recordings, test_recordings, seed):
    predict, cost = fit_supervised(name, train_recordings, seed)
    scores = _scores_from_probabilities(
        test_recordings,
        lambda recording: predict(recording_features(recording, recording.test_windows)),
    )
    return scores, cost


def supervised_cross_validated_scores(
    name: str,
    recordings: list[SkabRecording],
    seed: int,
    folds: int = 5,
) -> RecordingScores:
    """Return out-of-fold probabilities, holding out whole recordings per fold.

    Development recordings are scored by models that never saw them, so the
    result can be compared with unsupervised scores and used to set a
    threshold without touching final recordings.
    """
    tested = [recording for recording in recordings if recording.test_mask.any()]
    groups = np.array([recording.recording_id for recording in tested])
    if len(groups) < 2:
        raise ValueError("cross-validation needs at least two tested recordings")
    parts = []
    splitter = GroupKFold(n_splits=min(folds, len(tested)))
    for train_index, test_index in splitter.split(groups, groups=groups):
        parts.append(supervised_scores(
            name, [tested[i] for i in train_index], [tested[i] for i in test_index], seed,
        ))
    return concatenate_scores(parts)


def isolation_forest_scores(
    recordings: list[SkabRecording],
    seed: int,
    smoothing: float = 0.2,
    calibration_tail: float = 0.25,
) -> RecordingScores:
    """Fit Isolation Forest on every baseline and score in baseline MAD units."""
    baseline = np.concatenate([
        recording_features(recording, recording.train_windows) for recording in recordings
    ])
    model = IsolationForest(n_estimators=300, random_state=seed, n_jobs=-1).fit(baseline)
    return calibrate_components(
        recordings,
        lambda recording, windows: [-model.score_samples(recording_features(recording, windows))],
        smoothing,
        calibration_tail,
    )


def evaluate_classic_model(
    name: str,
    recordings: list[SkabRecording],
    seed: int,
    smoothing: float = 0.2,
    calibration_tail: float = 0.25,
    max_false_positive_rate: float = 0.15,
) -> dict:
    """Evaluate one classic model on development and final recordings.

    The threshold is chosen on development recordings with the same rule as
    every other model. Supervised models are scored on development recordings
    out of fold, then trained on all development recordings for the final
    ones. They are also scored on each final fault type after being trained
    without that type, which shows how they handle a fault they have not seen.
    """
    dev = [recording for recording in recordings if recording.role == "dev"]
    final = [recording for recording in recordings if recording.role == "final"]

    if name in UNSUPERVISED:
        scores = isolation_forest_scores(recordings, seed, smoothing, calibration_tail)
        dev_scores, final_scores = scores.select("dev"), scores.select("final")
        candidates = MAD_THRESHOLDS
    elif name in SUPERVISED:
        dev_scores = supervised_cross_validated_scores(name, dev, seed)
        final_scores, cost = _supervised_scores_and_cost(name, dev, final, seed)
        candidates = PROBABILITY_THRESHOLDS
    else:
        raise ValueError(f"Unknown classic model {name!r}")

    threshold = choose_threshold(dev_scores, np.array(candidates), max_false_positive_rate)
    result = {
        "threshold": threshold,
        "dev": alert_metrics(dev_scores, threshold),
        "final": alert_metrics(final_scores, threshold),
    }
    if name in SUPERVISED and cost:
        result["training"] = cost

    if name in SUPERVISED:
        unseen = []
        for fault_type in dict.fromkeys(recording.fault_type for recording in final):
            training = [recording for recording in dev if recording.fault_type != fault_type]
            testing = [recording for recording in final if recording.fault_type == fault_type]
            if any(recording.test_labels.any() for recording in training):
                unseen.append(supervised_scores(name, training, testing, seed))
        result["final_unseen_fault_type"] = alert_metrics(concatenate_scores(unseen), threshold)

    return result


def main() -> None:
    """Run classic models in their own process and save the results as JSON.

    PyTorch and XGBoost load different OpenMP runtimes, which crash each other
    on macOS, so the comparison script runs this module without PyTorch.
    """
    import argparse
    import json

    from hydrofed.experiments.recordings import load_skab_recordings

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--partitions", default="data/partitions_skab")
    parser.add_argument("--models", nargs="+", choices=CLASSIC_MODELS, default=list(CLASSIC_MODELS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--scaling", choices=("plant", "shared"), default="plant")
    parser.add_argument("--train-size", type=int, default=400)
    parser.add_argument("--smoothing", type=float, default=0.2)
    parser.add_argument("--calibration-tail", type=float, default=0.25)
    parser.add_argument("--max-false-positive-rate", type=float, default=0.15)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    recordings = [
        recording
        for items in load_skab_recordings(args.partitions, args.scaling, args.train_size).values()
        for recording in items
    ]
    results = {
        name: [
            evaluate_classic_model(
                name, recordings, seed, args.smoothing, args.calibration_tail,
                args.max_false_positive_rate,
            ) | {"seed": seed}
            for seed in args.seeds
        ]
        for name in args.models
    }
    with open(args.output, "w") as file:
        json.dump(results, file, indent=2)


if __name__ == "__main__":
    main()
