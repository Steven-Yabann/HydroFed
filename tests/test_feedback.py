import numpy as np
import pytest

from hydrofed.experiments.feedback import (
    alert_episodes,
    learned_threshold,
    rebaselined_scores,
    resolution_window,
    review_labels,
    split_by_time,
    with_review_labels,
)
from hydrofed.experiments.recordings import SkabRecording


def _recording(plant, name, role, tested=True):
    windows = np.zeros((6, 2, 1), dtype=np.float32)
    return SkabRecording(
        plant=plant, recording_id=name, role=role, windows=windows,
        endpoint_labels=np.array([0, 0, 0, 1, 1, 0]), window_labels=np.zeros(6, dtype=int),
        train_mask=np.array([1, 1, 0, 0, 0, 0], dtype=bool),
        test_mask=np.array([0, 0, 1, 1, 1, 1], dtype=bool) & tested,
    )


def test_alert_episodes_merge_short_gaps_and_review_against_labels():
    scores = np.array([0, 30, 30, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 50, 0])
    labels = np.zeros(len(scores), dtype=int)
    labels[15] = 1

    episodes = alert_episodes(scores, labels, threshold=24, merge_gap=3)

    assert [(e.start, e.end, e.peak, e.real_fault) for e in episodes] == [
        (1, 5, 40.0, False), (15, 15, 50.0, True),
    ]


def test_split_by_time_uses_earlier_development_recordings_as_the_past():
    recordings = {
        "plant_1": [
            _recording("plant_1", "a", "dev"), _recording("plant_1", "b", "final"),
            _recording("plant_1", "c", "dev"), _recording("plant_1", "d", "dev"),
            _recording("plant_1", "free", "dev", tested=False),
        ],
    }

    past, future = split_by_time(recordings)

    assert [r.recording_id for r in past] == ["a", "c"]
    assert [r.recording_id for r in future] == ["d"]


def test_only_alerted_faults_are_resolved():
    labels = np.array([0, 0, 1, 1, 1, 1, 0, 0])

    assert resolution_window(labels, np.full(8, 5.0), threshold=24) is None
    assert resolution_window(labels, np.full(8, 30.0), threshold=24, review_delay=2) == 7
    assert resolution_window(labels, np.full(8, 30.0), threshold=24, early=True) == 3
    assert resolution_window(np.zeros(8, dtype=int), np.full(8, 30.0), threshold=24) is None


def _components(seed, test_shift=0.0, length=300):
    rng = np.random.default_rng(seed)
    baseline = [rng.normal(size=(200, 2))]
    test = [rng.normal(size=(length, 2))]
    test[0][150:] += test_shift
    return baseline, test


@pytest.mark.parametrize("mode", ["relearn", "recentre"])
def test_rebaselining_leaves_scores_before_the_new_reference_unchanged(mode):
    baseline, test = _components(1, test_shift=8.0)
    original = rebaselined_scores(baseline, test, None)

    updated = rebaselined_scores(baseline, test, 150, rebaseline_windows=60, mode=mode)

    np.testing.assert_array_equal(updated[:210], original[:210])


@pytest.mark.parametrize("mode", ["relearn", "recentre"])
def test_rebaselining_accepts_a_level_the_plant_settled_at(mode):
    # The plant settles at a new level after a fault; the old reference keeps alarming.
    baseline, test = _components(2, test_shift=8.0)
    original = rebaselined_scores(baseline, test, None)

    updated = rebaselined_scores(baseline, test, 150, rebaseline_windows=60, mode=mode)

    assert (original[220:] > 24).mean() > 0.9
    assert (updated[220:] > 24).mean() < 0.1


def test_shelving_silences_alarms_only_while_the_reference_collects():
    baseline, test = _components(3, test_shift=8.0)

    shelved = rebaselined_scores(baseline, test, 150, rebaseline_windows=60, shelve=True, mode="recentre")
    kept = rebaselined_scores(baseline, test, 150, rebaseline_windows=60, mode="recentre")

    assert (shelved[150:210] < shelved[210:].min()).all()
    np.testing.assert_array_equal(shelved[:150], kept[:150])
    np.testing.assert_array_equal(shelved[210:], kept[210:])


def test_rebaselining_is_skipped_without_room_for_a_reference():
    baseline, test = _components(4, length=180)
    original = rebaselined_scores(baseline, test, None)

    np.testing.assert_array_equal(rebaselined_scores(baseline, test, 150, rebaseline_windows=60), original)
    with pytest.raises(ValueError, match="mode"):
        rebaselined_scores(baseline, test, 10, mode="forget")


def test_learned_threshold_silences_past_false_alarms_without_losing_faults():
    assert learned_threshold([80.0], [30.0, 38.0], base=24) == 39.0
    # A confirmed fault peaking at 29 caps the rise below it.
    assert learned_threshold([29.0, 80.0], [38.0], base=24) == 28.0
    # Never below the frozen threshold, never above the cap.
    assert learned_threshold([25.0], [38.0], base=24) == 24.0
    assert learned_threshold([], [300.0], base=24, cap=48) == 48.0
    assert learned_threshold([80.0], [], base=24) == 24.0


def test_review_labels_mark_whole_resolved_alerts_and_nothing_else():
    scores = np.array([0, 30, 30, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 40, 0, 0, 30])
    labels = np.zeros(len(scores), dtype=int)
    labels[2] = 1
    labels[8] = 1  # a fault that never alerted

    reviewed = review_labels(scores, labels, threshold=24)

    assert reviewed.tolist() == [0, 1, 1, 1] + [0] * 14


def test_with_review_labels_replaces_only_test_labels():
    recording = _recording("plant_1", "a", "dev")

    reviewed = with_review_labels(recording, np.array([1, 1, 1, 0]))

    assert reviewed.endpoint_labels.tolist() == [0, 0, 1, 1, 1, 0]
    assert reviewed.test_labels.tolist() == [1, 1, 1, 0]
    assert recording.endpoint_labels.tolist() == [0, 0, 0, 1, 1, 0]
