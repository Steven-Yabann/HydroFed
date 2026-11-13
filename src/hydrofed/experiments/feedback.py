"""Measure what alert reviews could teach the detector.

The dashboard lets operators mark each alert as resolved (a real fault) or a
false alarm. This module simulates a reviewer who follows the SKAB labels and
tests three ways to use those reviews:

re-baseline on resolved
    Once a confirmed fault is resolved, the plant's normal reference is
    updated from the minute just after it: re-learned entirely, or only
    re-centred on the level the plant settled at. Most false-alarm seconds are
    alerts that ring on after a fault ends, so this targets them directly.

suppress on false alarm
    A plant's threshold rises just enough to silence its past false alarms,
    capped, and never above a confirmed fault seen in the past.

reviews as labels
    A supervised model learns from labels taken from reviewed alerts, which
    mark whole alerts rather than the exact fault seconds.

Learning uses each plant's earlier development recordings and evaluation its
later ones, so past reviews are judged on future alerts. Final recordings are
not used. This module does not import PyTorch, so XGBoost can use it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from hydrofed.evaluation import calibrated_recording_scores
from hydrofed.experiments.recordings import SkabRecording

ALERT_MERGE_GAP = 10  # seconds of quiet that still belong to one alert, as on the dashboard
# SKAB recordings end about 170 s after their fault, so the review delay and
# the new reference must both fit inside that.
REVIEW_DELAY = 30  # seconds after a fault ends before the reviewer marks it resolved
REBASELINE_SECONDS = 60  # normal operation used as the new reference after a resolution
MIN_REBASELINE_WINDOWS = 30
THRESHOLD_CAP = 48.0  # a learned threshold may at most double the frozen 24


@dataclass(frozen=True)
class Episode:
    """One alert: alerted test windows merged across short quiet gaps."""

    start: int
    end: int
    peak: float
    real_fault: bool


def alert_episodes(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    merge_gap: int = ALERT_MERGE_GAP,
) -> list[Episode]:
    """Group alerted windows into alerts and review each against the labels.

    A simulated reviewer marks an alert as a real fault when it overlaps any
    labelled fault second, and as a false alarm otherwise.
    """
    alerted = np.flatnonzero(np.asarray(scores) > threshold)
    episodes = []
    for run in np.split(alerted, np.flatnonzero(np.diff(alerted) > merge_gap) + 1) if len(alerted) else []:
        start, end = int(run[0]), int(run[-1])
        episodes.append(Episode(
            start, end, float(np.max(scores[start:end + 1])), bool(np.any(labels[start:end + 1])),
        ))
    return episodes


def split_by_time(recordings: dict[str, list[SkabRecording]]) -> tuple[list[SkabRecording], list[SkabRecording]]:
    """Split each plant's tested development recordings into earlier and later halves."""
    past, future = [], []
    for items in recordings.values():
        tested = [recording for recording in items if recording.role == "dev" and recording.test_mask.any()]
        cut = math.ceil(len(tested) / 2)
        past.extend(tested[:cut])
        future.extend(tested[cut:])
    return past, future


def resolution_window(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    review_delay: int = REVIEW_DELAY,
    early: bool = False,
) -> int | None:
    """Return the test window at which the reviewer resolves the fault, if ever.

    Only an alerted fault gets reviewed. A careful reviewer resolves it
    ``review_delay`` seconds after the labelled fault ends; an early reviewer
    resolves it halfway through the fault, while it is still happening.
    """
    fault = np.flatnonzero(labels)
    if not len(fault) or not np.any(np.asarray(scores)[fault] > threshold):
        return None
    if early:
        return int(fault[0] + (fault[-1] - fault[0]) // 2)
    return int(fault[-1] + review_delay)


REBASELINE_MODES = ("relearn", "recentre")


def rebaselined_scores(
    baseline: list[np.ndarray],
    test: list[np.ndarray],
    resolution: int | None,
    smoothing: float = 0.2,
    calibration_tail: float = 0.25,
    rebaseline_windows: int = REBASELINE_SECONDS,
    shelve: bool = False,
    mode: str = "relearn",
) -> np.ndarray:
    """Score a recording's test windows, updating normal after a resolution.

    ``relearn`` scores later windows against the ``rebaseline_windows`` that
    followed the resolution, calibrated on all of them. ``recentre`` keeps the
    original reference and calibration, and only shifts each component by how
    far its mean moved during those windows, so a plant that settles at a new
    level is compared with its old spread.

    While the new reference collects, the original baseline keeps scoring, or
    with ``shelve`` alarms pause, as operators shelve them after maintenance.
    Shelved windows score just below every other window, so they never alert
    and rank lowest.
    """
    if mode not in REBASELINE_MODES:
        raise ValueError(f"mode must be one of {REBASELINE_MODES}")
    ids = lambda count: np.zeros(count, dtype=int)
    original = calibrated_recording_scores(
        baseline, ids(len(baseline[0])), test, ids(len(test[0])), smoothing, calibration_tail,
    )
    if resolution is None:
        return original
    switch = resolution + rebaseline_windows
    reference = slice(max(resolution, 0), switch)
    if switch >= len(test[0]) or len(test[0][reference]) < MIN_REBASELINE_WINDOWS:
        return original

    if mode == "recentre":
        shifted = []
        for component, base in zip(test, baseline):
            moved = component.astype(float).copy()
            moved[switch:] -= component[reference].mean(axis=0) - base.mean(axis=0)
            shifted.append(moved)
        scores = calibrated_recording_scores(
            baseline, ids(len(baseline[0])), shifted, ids(len(test[0])), smoothing, calibration_tail,
        )
    else:
        scores = original.copy()
        scores[switch:] = calibrated_recording_scores(
            [component[reference] for component in test], ids(len(test[0][reference])),
            [component[switch:] for component in test], ids(len(test[0]) - switch),
            smoothing, calibration_tail=1.0,
        )
    if shelve:
        scores[max(resolution, 0):switch] = min(scores.min(), 0.0) - 1.0
    return scores


def learned_threshold(
    fault_peaks: list[float],
    false_alarm_peaks: list[float],
    base: float,
    cap: float = THRESHOLD_CAP,
    margin: float = 1.0,
) -> float:
    """Raise a plant's threshold just above its past false alarms, safely.

    The threshold never drops below ``base``, never rises above ``cap``, and
    stays below every confirmed fault's peak, so no past fault would be lost.
    """
    if not false_alarm_peaks:
        return float(base)
    target = max(false_alarm_peaks) + margin
    if fault_peaks:
        target = min(target, min(fault_peaks) - margin)
    return float(min(cap, max(base, target)))


def review_labels(scores: np.ndarray, labels: np.ndarray, threshold: float) -> np.ndarray:
    """Turn reviewed alerts into per-window labels for supervised training.

    Every window of an alert the reviewer resolved counts as a fault, including
    any ringing after the fault ended; everything else counts as normal,
    including faults that never raised an alert.
    """
    reviewed = np.zeros(len(labels), dtype=np.int64)
    for episode in alert_episodes(scores, labels, threshold):
        if episode.real_fault:
            reviewed[episode.start:episode.end + 1] = 1
    return reviewed


def with_review_labels(recording: SkabRecording, test_labels: np.ndarray) -> SkabRecording:
    """Return a recording whose test windows carry review-derived labels."""
    endpoint_labels = recording.endpoint_labels.copy()
    endpoint_labels[recording.test_mask] = test_labels
    return replace(recording, endpoint_labels=endpoint_labels)


def main() -> None:
    """Train federated XGBoost from reviews in a process without PyTorch.

    Reads review-derived labels for each seed, trains on the earlier
    development recordings with those labels and with the true labels, and
    scores the later ones.
    """
    import argparse
    import json

    from hydrofed.experiments.classic import fit_supervised, recording_features
    from hydrofed.experiments.recordings import load_skab_recordings
    from hydrofed.experiments.scoring import RecordingScores, alert_metrics

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--partitions", default="data/partitions_skab")
    parser.add_argument("--review-labels", required=True, help="JSON of review labels per seed and recording")
    parser.add_argument("--threshold", type=float, default=0.5, help="fault probability that raises an alert")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    recordings = load_skab_recordings(args.partitions)
    past, future = split_by_time(recordings)
    reviews = json.loads(open(args.review_labels).read())

    def evaluate(train: list[SkabRecording], seed: int) -> dict:
        predict, cost = fit_supervised("federated_xgboost", train, seed)
        scores = RecordingScores(
            scores=np.concatenate([predict(recording_features(r, r.test_windows)) for r in future]),
            labels=np.concatenate([r.test_labels for r in future]),
            recording_ids=np.concatenate([np.repeat(r.recording_id, r.test_mask.sum()) for r in future]),
            roles=np.concatenate([np.repeat(r.role, r.test_mask.sum()) for r in future]),
        )
        return alert_metrics(scores, args.threshold)

    results = {}
    for seed, labels in reviews.items():
        reviewed = [with_review_labels(r, np.asarray(labels[r.recording_id])) for r in past]
        results[seed] = {
            "review_labels": evaluate(reviewed, int(seed)),
            "true_labels": evaluate(past, int(seed)),
        }
    with open(args.output, "w") as file:
        json.dump(results, file, indent=2)


if __name__ == "__main__":
    main()
