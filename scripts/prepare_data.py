#!/usr/bin/env python3
import argparse
import numpy as np

from hydrofed.data.pipeline import (
    create_windows,
    create_windows_by_recording,
    create_windows_with_recording_ids,
    create_windows_with_endpoint_metadata,
    download_skab,
    generate_synthetic_telemetry,
    load_skab_csvs,
    partition_non_iid,
    partition_stratified_iid,
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
    "--seed",
    type=int,
    default=42,
    help="Random seed used by non-IID and balanced IID-like partitioning",
)
p.add_argument(
    "--partition-strategy",
    choices=("temporal", "non-iid", "iid"),
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
# Cleaning missing values is safe before splitting. Scaling is intentionally
# deferred until the training/calibration/test split so held-out data cannot
# influence the feature statistics.
frame, columns = prepare_dataframe(frame, normalize=False); label = frame["anomaly"].to_numpy() if "anomaly" in frame else np.zeros(len(frame))

features = frame[columns].to_numpy()

window_recording_ids = None
window_metadata = None

if "_recording_id" in frame:
    x, y, window_recording_ids, window_metadata = create_windows_with_endpoint_metadata(
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
    p.error(
        "--minimum-anomalies is only supported by temporal or IID-like "
        "partitioning"
    )

if a.partition_strategy == "iid":
    required_anomalies = a.clients * a.minimum_anomalies

    if int(y.sum()) < required_anomalies:
        p.error(
            f"Cannot give {a.clients} plants at least "
            f"{a.minimum_anomalies} anomalies each: only {int(y.sum())} "
            "anomalous windows exist"
        )

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
elif a.partition_strategy == "iid":
    partitions = partition_stratified_iid(
        x,
        y,
        n_clients=a.clients,
        seed=a.seed,
    )
    strategy_name = "balanced IID-like"
else:
    partitions = partition_non_iid(
        x,
        y,
        n_clients=a.clients,
        seed=a.seed,
    )
    strategy_name = "non-IID"

partition_metadata = None

# Recording-aware temporal partitions preserve the original window order, so
# provenance arrays can be sliced alongside the resulting client windows.
# Randomised IID/non-IID controls intentionally omit SKAB provenance because
# their shuffled windows no longer represent chronological recordings.
if window_recording_ids is not None and a.partition_strategy == "temporal":
    partition_metadata = {}
    cursor = 0

    for client_id, (client_windows, _) in partitions.items():
        next_cursor = cursor + len(client_windows)
        partition_metadata[client_id] = {
            "recording_ids": window_recording_ids[cursor:next_cursor],
            "endpoint_labels": window_metadata["endpoint_labels"][cursor:next_cursor],
            "window_start_indices": window_metadata["window_start_indices"][cursor:next_cursor],
        }
        cursor = next_cursor

path = save_partitions(partitions, a.output, metadata=partition_metadata)

print(
    f"Saved {len(x)} windows across {a.clients} clients "
    f"using {strategy_name} partitioning; summary: {path}"
)
