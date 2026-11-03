"""Compare federated weight traffic with sending raw telemetry to a server.

Federated training costs a fixed number of bytes per training run, while
centralised training needs the raw telemetry. For a stored dataset like SKAB
the raw files can be much smaller than the weight traffic. A plant streaming
telemetry produces data continuously, so the useful comparison is how many
days of streaming one federated training run is worth.
"""

from __future__ import annotations

from pathlib import Path

from hydrofed.data.pipeline import load_skab_csvs, sensor_columns

SECONDS_PER_DAY = 86_400


def federated_bytes(model_bytes: int, clients: int, rounds: int) -> dict[str, int]:
    """Return FedAvg traffic when every client takes part in every round."""
    if model_bytes <= 0 or clients <= 0 or rounds <= 0:
        raise ValueError("model_bytes, clients and rounds must be positive")
    per_direction = model_bytes * clients * rounds
    return {
        "uplink_bytes": per_direction,
        "downlink_bytes": per_direction,
        "total_bytes": 2 * per_direction,
        "bytes_per_round": 2 * model_bytes * clients,
        "bytes_per_client": 2 * model_bytes * rounds,
    }


def raw_skab_bytes(
    skab_path: str | Path,
    train_size: int = 400,
    bytes_per_value: int = 4,
) -> dict[str, int]:
    """Measure what centralising the SKAB recordings would transfer.

    Reports the CSV files as stored, every sensor row as packed binary values,
    and only the baseline rows that unsupervised training uses.
    """
    files = sorted(Path(skab_path).rglob("*.csv"))
    frame = load_skab_csvs(skab_path)
    sensors = len(sensor_columns(frame))
    rows = len(frame)
    baseline_rows = int(
        frame.groupby("_recording_id").size().clip(upper=train_size).sum()
    )
    return {
        "csv_files": len(files),
        "csv_bytes": int(sum(file.stat().st_size for file in files)),
        "rows": rows,
        "sensors": sensors,
        "binary_bytes": rows * sensors * bytes_per_value,
        "baseline_binary_bytes": baseline_rows * sensors * bytes_per_value,
    }


def streaming_bytes_per_day(
    sensors: int,
    sample_rate_hz: float = 1.0,
    bytes_per_value: int = 4,
) -> float:
    """Return the raw bytes one plant streams to a server each day."""
    if sensors <= 0 or sample_rate_hz <= 0 or bytes_per_value <= 0:
        raise ValueError("sensors, sample_rate_hz and bytes_per_value must be positive")
    return sensors * sample_rate_hz * bytes_per_value * SECONDS_PER_DAY


def break_even_days(
    federated_bytes_per_client: int,
    sensors: int,
    sample_rate_hz: float = 1.0,
    bytes_per_value: int = 4,
) -> float:
    """Days of streaming after which one federated training run moves fewer bytes.

    If a plant retrains less often than this, federated learning moves less
    data than streaming its raw telemetry to a central server.
    """
    return federated_bytes_per_client / streaming_bytes_per_day(
        sensors, sample_rate_hz, bytes_per_value,
    )
