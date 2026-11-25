# HydroFed

HydroFed is a research prototype for privacy-preserving anomaly detection on
hydropower-style telemetry. Five simulated plants, built from the Skoltech
Anomaly Benchmark (SKAB), each keep their data local and train an LSTM
autoencoder together with federated averaging (FedAvg). Detected anomalies are
turned into alerts and shown on a role-based dashboard.

## Pipeline

```text
SKAB CSVs (35 recordings, 8 sensors, 1 Hz)
  → clean each recording (fill gaps within the recording only)
  → 10-second windows, stride 1, never crossing a recording boundary
  → split whole recordings across five plants (each with ≥ 100 anomalous windows)
  → alternate each plant's recordings into development and final roles
  → first 400 s of each recording = fault-free baseline, the rest = test
  → per-plant standard scaler fitted on that plant's normal baseline windows
  → train the LSTM autoencoder: federated (FedAvg), centralised or isolated
  → hybrid score: per-sensor distance from the recording's baseline plus the
    model's residual, smoothed, with a per-recording threshold
    (median + 24 MAD of the baseline tail)
  → recall, false alarm rate, within-recording AUC-ROC, faults detected, delay
  → JSON results and SQLite → FastAPI → React dashboard
```

Settings are chosen on development recordings. Final recordings are used only
for reporting.

## Setup and tests

Requires Python 3.11 and, for the dashboard, Node.js 22.12 or newer.

```bash
make setup                                # .venv with the pinned dependencies
.venv/bin/pip install -e ".[experiments]" # XGBoost, for the model comparison
make test                                 # pytest; needs no data files
```

GitHub Actions runs the same tests and the frontend build on every pull
request.

## Data

```bash
make partition-skab
```

This downloads SKAB at commit `b2c0d46` (the version every result used) into
`data/SKAB` when no CSVs are there, then writes the five plant partitions to
`data/partitions_skab`. Existing CSVs are reused.

## Reproducing the results

Every comparison uses seeds 42, 43 and 44, 20 rounds of 5 local epochs, and
the frozen alert policy. Results are written to `artifacts/comparison/`.

| Experiment | Command | Output |
| --- | --- | --- |
| Federated, centralised and isolated training | `.venv/bin/python scripts/run_comparison.py regimes --jobs 3` | `regimes.json` |
| LSTM against transformer and classic models | `.venv/bin/python scripts/run_comparison.py models --jobs 3` | `models.json` |
| Heterogeneous and data-scarce plants | `.venv/bin/python scripts/run_comparison.py scenarios --jobs 3` | `scenarios.json` |
| FedProx and fine-tuned federated training | `.venv/bin/python scripts/run_comparison.py scenarios --regimes fedprox federated_finetune --jobs 5 --output artifacts/comparison/personalised.json` | `personalised.json` |
| Plants that miss federated rounds | `.venv/bin/python scripts/run_comparison.py connectivity --jobs 6` | `connectivity.json` |
| Learning from alert reviews (development only) | `.venv/bin/python scripts/run_comparison.py feedback --jobs 3` | `feedback.json` |

These comparisons run federated training in-process with the same
example-weighted FedAvg as Flower's strategy, because a Ray simulation per seed
is slow. [docs/comparison.md](docs/comparison.md) explains each experiment and
records its results. The result files from the reported runs are attached to
the GitHub release.

### Flower simulation

`scripts/run_experiment.py` runs the same model as a Flower simulation with
`TrackingFedAvg`, which records each client's loss and the uplink and downlink
bytes per round. `make federated` runs it with the reported settings (seed 42,
20 rounds of 5 local epochs, latent size 4 and the hybrid alert policy).
[docs/skab-diagnostics.md](docs/skab-diagnostics.md#hybrid-recording-baseline-experiment)
explains the flags. `configs/skab_experiment.yaml` lists the reported settings
in one place.

## Dashboard

The React dashboard (`frontend/`) is for two roles:

- **Plant operators** are each assigned one plant. They see only that plant's
  live anomaly score and sensor trends, and review its alerts (acknowledge,
  investigate, resolve or mark a false alarm). The API enforces the
  assignment, so another plant's data is refused, not just hidden.
- **System administrators** see every plant and all alerts, read **Model
  performance** (federated, centralised and isolated training, and the model
  families), configure and start federated training, pause or rename plant
  nodes, and add users and assign operators to plants.

Each plant also has a **Messages** conversation between its operator and the
administrators. Administrators see every plant's conversation. A message can
refer to one of the plant's alerts, and the sidebar shows how many messages
are unread. When email is set up (see `docs/deployment.md`), the other side of
the conversation also gets an email, once per conversation until they read it.

There is no live plant, so `make operator-feed` scores every SKAB recording
with the saved federated LSTM checkpoint and the frozen alert policy. The API
replays those scores second by second as if they were streaming in.

The API reads its settings from environment variables, listed with their
local defaults in `.env.example`.

```bash
make operator-feed                                      # writes artifacts/operator/
make admin EMAIL=you@example.com NAME="Your Name"       # first administrator; asks for a password
make api                                                # FastAPI on :8000
make dashboard                                          # Vite on :5173
```

`frontend/.env` sets `VITE_API_BASE_URL` to the API address.

### Sign-in

Only accounts an administrator has created can sign in, by password or single
sign-on. Google and Microsoft sign-in use the OpenID Connect code flow with
PKCE, and the provider's email must match an existing account. To enable a
provider, register the redirect URI
`<HYDROFED_PUBLIC_API_URL>/api/auth/oauth/google/callback` (or
`.../microsoft/callback`) and set its client ID and secret on the API (see
`.env.example`).

Set `HYDROFED_SECRET_KEY` in any shared deployment. Without it, the API keeps a
random signing key beside the database.

## Deployment

The dashboard and API are deployed on Render from the `main` branch. The API
image is built from `Dockerfile.api` and serves the published results in
`deploy/published/`, which `make publish` copies from `artifacts/`. The
hosted API does not train itself: **Start on GitHub Actions** on the Federated
training page runs the `Federated training` workflow, which reports the run
back to the API (setup in [docs/deployment.md](docs/deployment.md)). To run
the same stack locally:

```bash
cp deployment.env.example deployment.env
docker compose --env-file deployment.env -f docker-compose.deploy.yml up --build
```

See [docs/deployment.md](docs/deployment.md) for the Render settings.

## Monitoring

`scripts/run_experiment.py` exports Prometheus metrics on port 8000 while it
trains (`--metrics-port`; `--keep-metrics-seconds` keeps them up afterwards).
Run `make monitor-up` to start Prometheus on port 9090 and Grafana on port
3000. Prometheus scrapes port 8000, so don't run the API on that port at the
same time. The Grafana dashboard shows training losses, communication bytes,
anomaly scores and F1. Run `make monitor-down` afterward.

## Generated files

These are git-ignored:

- `data/SKAB/`: the downloaded SKAB recordings
- `data/partitions_skab/plant_1.npz` … `plant_5.npz`: each plant's windows,
  labels and recording metadata, unscaled
- `artifacts/comparison/*.json`: comparison results
- `artifacts/operator/`: the replay feed for the dashboard
- `artifacts/experiments.db`: runs, results, users, plant nodes and alert
  reviews

## Layout

Source modules live under `src/hydrofed`, commands under `scripts`, and tests
under `tests`.

## License

[MIT](LICENSE)
