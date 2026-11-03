import numpy as np
import pytest

from hydrofed.data.scaling import (
    WindowMinMaxScaler,
    WindowStandardScaler,
    fit_shared_standard_window_scaler,
    fit_shared_window_scaler,
    scale_plant_split,
)
from hydrofed.data.splits import PlantDataSplit


def test_window_scaler_uses_only_supplied_training_windows():
    training = np.array([[[2.0], [4.0]], [[6.0], [8.0]]], dtype=np.float32)
    held_out = np.array([[[100.0], [2.0]]], dtype=np.float32)

    scaler = WindowMinMaxScaler.fit(training)

    np.testing.assert_allclose(scaler.transform(training).min(), 0.0)
    np.testing.assert_allclose(scaler.transform(training).max(), 1.0)
    assert scaler.transform(held_out)[0, 0, 0] > 1.0


def test_shared_window_scaler_combines_normal_training_windows_from_plants():
    plant_one = np.array([[[0.0], [2.0]]], dtype=np.float32)
    plant_two = np.array([[[4.0], [8.0]]], dtype=np.float32)

    scaler = fit_shared_window_scaler([plant_one, plant_two])

    np.testing.assert_allclose(scaler.minimums, [0.0])
    np.testing.assert_allclose(scaler.spans, [8.0])
    np.testing.assert_allclose(scaler.transform(plant_two), [[[0.5], [1.0]]])


def test_window_scaler_handles_constant_sensor_without_non_finite_values():
    training = np.array([[[5.0, 1.0], [5.0, 3.0]]], dtype=np.float32)

    scaled = WindowMinMaxScaler.fit(training).transform(training)

    assert np.isfinite(scaled).all()
    np.testing.assert_allclose(scaled[:, :, 0], 0.0)


def test_window_scaler_rejects_feature_count_mismatch():
    scaler = WindowMinMaxScaler.fit(np.ones((2, 3, 1), dtype=np.float32))

    with pytest.raises(ValueError, match="different number of features"):
        scaler.transform(np.ones((2, 3, 2), dtype=np.float32))


def test_shared_standard_scaler_uses_training_mean_and_standard_deviation():
    plant_one = np.array([[[0.0], [2.0]]], dtype=np.float32)
    plant_two = np.array([[[4.0], [6.0]]], dtype=np.float32)

    scaler = fit_shared_standard_window_scaler([plant_one, plant_two])

    np.testing.assert_allclose(scaler.means, [3.0])
    np.testing.assert_allclose(scaler.standard_deviations, [np.sqrt(5.0)])
    np.testing.assert_allclose(scaler.transform(plant_two), [[[1 / np.sqrt(5)], [3 / np.sqrt(5)]]])


def test_standard_scaler_handles_constant_sensor():
    scaler = WindowStandardScaler.fit(np.ones((2, 3, 1), dtype=np.float32))

    np.testing.assert_allclose(scaler.transform(np.ones((1, 2, 1))), 0.0)


def test_scale_plant_split_uses_the_same_training_scaler_for_held_out_data():
    scaler = WindowMinMaxScaler.fit(np.array([[[0.0], [10.0]]], dtype=np.float32))
    split = PlantDataSplit(
        train_windows=np.array([[[0.0], [10.0]]], dtype=np.float32),
        calibration_windows=np.array([[[5.0], [10.0]]], dtype=np.float32),
        test_windows=np.array([[[15.0], [20.0]]], dtype=np.float32),
        test_labels=np.array([0], dtype=np.int64),
        train_recording_ids=np.array(["recording_a"]),
        test_recording_ids=np.array(["recording_b"]),
    )

    scaled = scale_plant_split(split, scaler)

    np.testing.assert_allclose(scaled.calibration_windows, [[[0.5], [1.0]]])
    np.testing.assert_allclose(scaled.test_windows, [[[1.5], [2.0]]])
    np.testing.assert_array_equal(scaled.test_labels, split.test_labels)
    np.testing.assert_array_equal(scaled.train_recording_ids, split.train_recording_ids)
    np.testing.assert_array_equal(scaled.test_recording_ids, split.test_recording_ids)
