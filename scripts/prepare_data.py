#!/usr/bin/env python3
import argparse
import numpy as np

from hydrofed.data.pipeline import (
    create_windows,
    create_windows_by_recording,
    create_windows_with_recording_ids,
    download_skab,
    generate_synthetic_telemetry,
    load_skab_csvs,
    partition_non_iid,
    partition_temporal,
    partition_temporal_by_recording,
    prepare_dataframe,
    save_partitions,
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
p.add_argument(
    "--minimum-anomalies",
    type=int,
    default=0,
    help="Minimum anomalous windows required for each temporal plant partition",
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

window_recording_ids = None

if "_recording_id" in frame:
    x, y, window_recording_ids = create_windows_with_recording_ids(
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

if a.partition_strategy == "non-iid" and a.minimum_anomalies:
    p.error("--minimum-anomalies is only supported by temporal partitioning")

partition_kwargs = {"n_clients": a.clients}

if a.partition_strategy == "temporal":
    partition_kwargs["minimum_anomalies"] = a.minimum_anomalies

    if window_recording_ids is None:
        partitions = partition_temporal(x, y, **partition_kwargs)
        strategy_name = "temporal"
    else:
        partitions = partition_temporal_by_recording(
            x,
            y,
            window_recording_ids,
            **partition_kwargs,
        )
        strategy_name = "recording-aware temporal"
else:
    partitions = partition_non_iid(x, y, **partition_kwargs)
    strategy_name = "non-iid"

path = save_partitions(partitions, a.output)

print(
    f"Saved {len(x)} windows across {a.clients} clients "
    f"using {strategy_name} partitioning; summary: {path}"
)
