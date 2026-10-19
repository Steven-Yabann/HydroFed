#!/usr/bin/env python3
import argparse
import numpy as np

from hydrofed.data.pipeline import (
    create_windows,
    download_skab,
    generate_synthetic_telemetry,
    load_skab_csvs,
    partition_non_iid,
    partition_temporal,
    prepare_dataframe,
    save_partitions,
    create_windows_by_recording,
)

p=argparse.ArgumentParser(); p.add_argument("--input"); p.add_argument("--output", default="data/partitions")
p.add_argument("--synthetic", action="store_true"); p.add_argument("--download", action="store_true"); p.add_argument("--rows", type=int, default=1500)
p.add_argument("--features", type=int, default=8); p.add_argument("--window", type=int, default=32)
p.add_argument("--stride", type=int, default=4); p.add_argument("--clients", type=int, default=5)
p.add_argument(
    "--partition-strategy",
    choices=("temporal", "non-iid"),
    default="temporal",
)

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

features = frame[columns].to_numpy()

if "_recording_id" in frame:
    x, y = create_windows_by_recording(
        features,
        label,
        frame["_recording_id"].to_numpy(),
        a.window,
        a.stride,
    )
else:
    x, y = create_windows(
        features,
        label,
        a.window,
        a.stride,
    )

partitioner = (
    partition_temporal
    if a.partition_strategy == "temporal"
    else partition_non_iid
)

path = save_partitions(
    partitioner(x, y, a.clients),
    a.output,
)

print(
    f"Saved {len(x)} windows across {a.clients} clients "
    f"using {a.partition_strategy} partitioning; summary: {path}"
)
