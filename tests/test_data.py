import json
import numpy as np
from hydrofed.data.pipeline import *
from hydrofed.data.splits import split_plant_windows

def test_preprocessing_fills_and_scales():
    frame=generate_synthetic_telemetry(50,3); frame.loc[2,"sensor_1"]=np.nan
    clean, columns=prepare_dataframe(frame)
    assert len(columns)==3 and not clean[columns].isna().any().any()
    assert clean[columns].min().min() >= 0 and clean[columns].max().max() <= 1

def test_window_labels_any_and_majority():
    x=np.arange(20).reshape(10,2); y=np.array([0,0,1,0,0,0,1,1,1,0])
    windows, any_labels=create_windows(x,y,4,2,"any"); _, majority=create_windows(x,y,4,2,"majority")
    assert windows.shape==(4,4,2) and any_labels.tolist()==[1,1,1,1]
    assert majority.tolist()==[0,0,0,1]

def test_synthetic_partition_save_load(tmp_path):
    frame=generate_synthetic_telemetry(200,4,seed=1); clean, cols=prepare_dataframe(frame)
    x,y=create_windows(clean[cols].to_numpy(),clean.anomaly.to_numpy(),16,4)
    parts=partition_non_iid(x,y,5); summary=save_partitions(parts,tmp_path)
    loaded_x,loaded_y=load_client_partition(tmp_path/"plant_1.npz")
    assert len(parts)==5 and loaded_x.ndim==3 and len(loaded_x)==len(loaded_y)
    assert len(json.loads(summary.read_text()))==5

def test_download_skab_reuses_existing_csv(tmp_path, monkeypatch):
    dataset=tmp_path/"SKAB"; dataset.mkdir(); (dataset/"existing.csv").write_text("sensor,anomaly\n1,0\n")
    def unexpected_request(*args, **kwargs):
        raise AssertionError("existing SKAB data should not be downloaded again")
    monkeypatch.setattr("hydrofed.data.pipeline.requests.get", unexpected_request)
    assert download_skab(dataset)==dataset

def test_temporal_partition_keeps_window_order_per_plant():
    windows = np.arange(10 * 2, dtype=np.float32).reshape(10, 2, 1)
    labels = np.arange(10, dtype=np.int64)

    parts = partition_temporal(windows, labels, n_clients=3)

    np.testing.assert_array_equal(
        parts["plant_1"][1],
        np.array([0, 1, 2, 3]),
    )
    np.testing.assert_array_equal(
        parts["plant_2"][1],
        np.array([4, 5, 6]),
    )
    np.testing.assert_array_equal(
        parts["plant_3"][1],
        np.array([7, 8, 9]),
    )

def test_plant_split_keeps_train_calibration_and_test_disjoint():
    windows = np.arange(10 * 2, dtype=np.float32).reshape(10, 2, 1)
    labels = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 0])

    split = split_plant_windows(
        windows,
        labels,
        train_fraction=0.5,
        calibration_fraction=0.2,
    )

    np.testing.assert_array_equal(
        split.train_windows[:, 0, 0],
        np.array([0, 4, 6], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        split.calibration_windows[:, 0, 0],
        np.array([10, 12], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        split.test_labels,
        np.array([0, 1, 0]),
    )

def test_windows_do_not_cross_recording_boundaries():
    features = np.array(
        [[0], [0], [0], [0], [100], [100], [100], [100]],
        dtype=np.float32,
    )
    labels = np.zeros(8, dtype=np.int64)
    recording_ids = np.array(
        ["recording_a"] * 4 + ["recording_b"] * 4
    )

    windows, window_labels = create_windows_by_recording(
        features,
        labels,
        recording_ids,
        window_length=3,
    )

    assert windows.shape == (4, 3, 1)
    assert window_labels.tolist() == [0, 0, 0, 0]

    for window in windows:
        assert np.all(window == window[0])

def test_load_skab_csvs_tracks_the_source_recording(tmp_path):
    (tmp_path / "a.csv").write_text(
        "sensor,anomaly\n1,0\n2,0\n"
    )
    (tmp_path / "b.csv").write_text(
        "sensor,anomaly\n3,1\n4,0\n"
    )

    frame = load_skab_csvs(tmp_path)

    assert frame["_recording_id"].tolist() == [
        "a.csv",
        "a.csv",
        "b.csv",
        "b.csv",
    ]