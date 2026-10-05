#!/usr/bin/env python3
import argparse
import numpy as np
from hydrofed.data.pipeline import download_skab, generate_synthetic_telemetry, load_skab_csvs, prepare_dataframe, create_windows, partition_non_iid, save_partitions
p=argparse.ArgumentParser(); p.add_argument("--input"); p.add_argument("--output", default="data/partitions")
p.add_argument("--synthetic", action="store_true"); p.add_argument("--download", action="store_true"); p.add_argument("--rows", type=int, default=1500)
p.add_argument("--features", type=int, default=8); p.add_argument("--window", type=int, default=32)
p.add_argument("--stride", type=int, default=4); p.add_argument("--clients", type=int, default=5)
a=p.parse_args()
if not a.synthetic and not a.input:
    p.error("--input is required unless --synthetic is used")
if a.download:
    try:
        print(f"Ensuring SKAB is available under {a.input}...")
        download_skab(a.input)
    except Exception as error:
        p.error(
            f"could not download SKAB: {error}. "
            "Check your network, or clone https://github.com/waico/SKAB into the input directory."
        )
frame = generate_synthetic_telemetry(a.rows, a.features) if a.synthetic else load_skab_csvs(a.input)
frame, columns = prepare_dataframe(frame); label = frame["anomaly"].to_numpy() if "anomaly" in frame else np.zeros(len(frame))
x,y=create_windows(frame[columns].to_numpy(), label, a.window, a.stride)
path=save_partitions(partition_non_iid(x,y,a.clients),a.output)
print(f"Saved {len(x)} windows across {a.clients} clients; summary: {path}")
