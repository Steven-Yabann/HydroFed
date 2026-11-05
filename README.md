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

[docs/comparison.md](docs/comparison.md) explains how to rerun the SKAB comparisons and gives their results: federated against centralised and isolated training, and the LSTM against transformer and classic models.

## Dashboard

The React dashboard (`frontend/`) is for two roles:

- **Plant operators** watch the fleet's live anomaly scores, open a plant's score and sensor trends, and review alerts (acknowledge, investigate, resolve or mark a false alarm).
- **System administrators** can do all of that, and also configure and start federated training, pause or rename plant nodes, and add users.

Both roles can open **Model performance**, which compares federated, centralised and isolated training, the LSTM, transformer and classical model families, and communication overhead, using the results in `artifacts/comparison`.

There is no live plant, so `make operator-feed` scores every SKAB recording (baseline and test region) with the saved federated LSTM checkpoint and the frozen alert policy. The API replays those scores second by second as if they were streaming in.

```bash
make operator-feed                                      # writes artifacts/operator/
make admin EMAIL=you@example.com NAME="Your Name"       # first administrator; asks for a password
make api                                                # FastAPI on :8000
make dashboard                                          # Vite on :5173
```

`frontend/.env` sets `VITE_API_BASE_URL` to the API address.

### Sign-in

Only accounts an administrator has created can sign in, by password or single sign-on. Google and Microsoft sign-in use the OpenID Connect code flow with PKCE, and the provider's email must match an existing account. To enable a provider, register the redirect URI `<HYDROFED_PUBLIC_API_URL>/api/auth/oauth/google/callback` (or `.../microsoft/callback`) and set its client ID and secret on the API (see `.env.example`). Buttons for unconfigured providers appear greyed out in development and are hidden in production builds.

Set `HYDROFED_SECRET_KEY` in any shared deployment. Without it, the API keeps a random signing key beside the database.

## Monitoring

Start a process which imports `hydrofed.monitoring` and calls `serve(8000)`, then run `make monitor-up`. Prometheus is at port 9090 and Grafana at port 3000. Run `make monitor-down` afterward. The dashboard covers losses, anomaly score, communication bytes, and F1; no cloud deployment is included.

## Generated files

- `data/partitions/plant_1.npz` … `plant_5.npz`: window and label arrays
- `data/partitions/partition_summary.json`: counts and anomaly prevalence
- `artifacts/baselines/*.pt`: weights, feature count, and calibrated threshold
- `artifacts/experiments.db`: evaluation and federation metadata

Source modules live under `src/hydrofed`; commands under `scripts`; tests under `tests`; presentation planning under `.context/planning`. Generated data and model files should normally remain uncommitted.
