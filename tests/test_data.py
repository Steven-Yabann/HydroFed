import json
import numpy as np
from hydrofed.data.pipeline import *

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
