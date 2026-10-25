# HydroFed

HydroFed is a local Python 3.11 research system for privacy-preserving anomaly detection on hydropower-style telemetry. Each simulated plant keeps its windows local, trains an LSTM autoencoder, and either remains an isolated baseline or participates in Flower FedAvg.

```text
SKAB CSVs / synthetic telemetry
  → fill missing values → min-max scale → sliding windows
  → five non-IID plant NPZ files
  → isolated LSTM baselines OR Flower FedAvg
  → reconstruction scores and calibrated threshold
  → precision / recall / F1 / AUC-ROC
  → SQLite + Prometheus + Grafana
```

## Setup and tests

```bash
make setup
make test
```

The model accepts `(batch_size, window_length, n_features)` tensors and reconstructs the same shape. An anomaly score is the mean squared reconstruction error for one complete window.

## Fast offline synthetic demo

```bash
make partition-synthetic
make baseline
make evaluate
```

For a faster rehearsal:

```bash
.venv/bin/python scripts/prepare_data.py --synthetic --rows 300 --features 4 --window 16 --stride 4 --clients 5 --output /tmp/hydrofed-demo
.venv/bin/python scripts/run_baseline.py --partitions /tmp/hydrofed-demo --output /tmp/hydrofed-models --epochs 1 --db /tmp/hydrofed.db
```

## SKAB demo

Run `make partition`: it downloads SKAB into `data/SKAB` when no CSVs exist, then creates the client partitions. Existing CSVs are reused, so later runs do not download the dataset again. You can also clone `https://github.com/waico/SKAB` into `data/SKAB` yourself. The synthetic generator provides an offline fallback with SKAB-shaped numeric sensors plus an `anomaly` column.

## Federated experiment

Flower simulation uses the Ray runtime included by `make setup`. Run `make federated`.

The default is two rounds and five clients. Use `scripts/run_experiment.py --rounds 1 --epochs 1` for a reduced run. Client/round losses and approximate NumPy tensor uplink/downlink bytes are written to `artifacts/experiments.db`.

## Evaluation

`make evaluate` evaluates `plant_1` using its saved threshold. Thresholds are the 95th percentile of reconstruction errors on held-out normal windows. AUC-ROC uses continuous scores; precision, recall, and F1 use the threshold.

## Monitoring

Start a process which imports `hydrofed.monitoring` and calls `serve(8000)`, then run `make monitor-up`. Prometheus is at port 9090 and Grafana at port 3000. Run `make monitor-down` afterward. The dashboard covers losses, anomaly score, communication bytes, and F1; no cloud deployment is included.

## Generated files

- `data/partitions/plant_1.npz` … `plant_5.npz`: window and label arrays
- `data/partitions/partition_summary.json`: counts and anomaly prevalence
- `artifacts/baselines/*.pt`: weights, feature count, and calibrated threshold
- `artifacts/experiments.db`: evaluation and federation metadata

Source modules live under `src/hydrofed`; commands under `scripts`; tests under `tests`; presentation planning under `.context/planning`. Generated data and model files should normally remain uncommitted.
