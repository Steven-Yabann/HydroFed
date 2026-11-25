# Deploying HydroFed

## Overview

HydroFed has been verified in two deployment modes:

- **Local Docker Compose** for production-like development and testing.
- **Render** for the public dashboard and API deployment.

The system stores experiment runs and evaluation results in SQLite. SQLite is
appropriate for the current single API instance, but the API must not be scaled
to multiple replicas while it uses one database file.

## Architecture

```text
Browser
  |
  v
Render Static Site
https://hydrofed-dashboard.onrender.com
  |
  | HTTPS requests with CORS protection
  v
Render Web Service
https://hydrofed.onrender.com
  |
  v
Persistent disk mounted at /data
  |
  v
/data/experiments.db
```

The frontend is a Vite/React static site. The API is a FastAPI Docker service.
The persistent disk preserves accounts, alert reviews, administrator settings,
saved experiment runs and evaluation metrics when the API restarts or redeploys.

The API image also carries the results it serves, from `deploy/published/`:
the operator replay feed, the comparison results and the plant partition
summary. They are produced under the git-ignored `artifacts/` and `data/`
folders, so `make publish` copies them into `deploy/published/`, which is
committed. Run `make operator-feed publish` and commit the result whenever the
model or the comparison results change.

## Local Docker Compose verification

From the repository root:

```bash
cp deployment.env.example deployment.env
docker compose --env-file deployment.env -f docker-compose.deploy.yml up --build
```

Set `HYDROFED_ADMIN_EMAIL`, `HYDROFED_ADMIN_NAME` and `HYDROFED_ADMIN_PASSWORD`
in `deployment.env` before the first start, then open `http://localhost:8080`
and sign in with that account.

Verify the API:

```bash
curl http://localhost:8080/api/health
```

Expected response:

```json
{"status":"ok"}
```

The local stack uses the `hydrofed-data` Docker volume to persist
`/data/experiments.db`.

## Render deployment

Render deploys HydroFed as two separate services.

### API web service

```text
Service name: HydroFed
Runtime: Docker
Branch: main
Docker build context: .
Dockerfile path: Dockerfile.api
Health check path: /api/health
Persistent disk mount path: /data
```

API environment variables:

```text
HYDROFED_DB_PATH=/data/experiments.db
HYDROFED_ALLOWED_ORIGINS=https://hydrofed-dashboard.onrender.com
HYDROFED_FRONTEND_URL=https://hydrofed-dashboard.onrender.com
HYDROFED_PUBLIC_API_URL=https://hydrofed.onrender.com
HYDROFED_SECRET_KEY=<a long random string>
HYDROFED_ADMIN_EMAIL=<first administrator's email>
HYDROFED_ADMIN_NAME=<their name>
HYDROFED_ADMIN_PASSWORD=<at least 10 characters, or leave unset for SSO only>
```

`HYDROFED_SECRET_KEY` signs session tokens; changing it signs everyone out.
The `HYDROFED_ADMIN_*` variables only create an account while the database has
no active administrator, so they can be removed once that person has signed in.
Everyone else is added from **Users & access** in the dashboard.

Single sign-on is optional. To enable it, add the client ID and secret for
each provider and register these redirect URIs with the provider:

```text
HYDROFED_GOOGLE_CLIENT_ID / HYDROFED_GOOGLE_CLIENT_SECRET
  https://hydrofed.onrender.com/api/auth/oauth/google/callback
HYDROFED_MICROSOFT_CLIENT_ID / HYDROFED_MICROSOFT_CLIENT_SECRET
  https://hydrofed.onrender.com/api/auth/oauth/microsoft/callback
```

The image cannot train by itself (`HYDROFED_ALLOW_TRAINING=0`): it has no
training data and a web service is the wrong place for long simulations.
Training runs on GitHub Actions instead, once it is set up (next section).
Without it, administrators can still change the federation settings and plant
nodes.

### Federated training on GitHub Actions

**Start on GitHub Actions** on the **Federated training** page dispatches the
`Federated training` workflow (`.github/workflows/federated-training.yml`)
with the saved settings and the participating plants. The workflow installs
the CPU build of PyTorch, downloads SKAB at the pinned commit, builds the
plant partitions, runs `scripts/run_experiment.py`, and posts the run and its
results back to `https://hydrofed.onrender.com/api/training/report`. The page
shows the workflow's steps while it runs, links to its logs, and lists the run
under **Recent federated runs** when it finishes. **Stop run** cancels the
workflow. The run's database, checkpoint and report are also kept as a
workflow artifact for 30 days.

Set it up once:

1. **Create a GitHub token** the API uses to start and cancel runs. In GitHub,
   go to **Settings → Developer settings → Fine-grained personal access
   tokens → Generate new token**. Limit it to the HydroFed repository and give
   it **Actions: Read and write** (Metadata: Read is added automatically).
   Nothing else is needed. Note its expiry date.
2. **Create a runner token**, a long random string the workflow uses to prove
   its results are genuine, for example with
   `python -c "import secrets; print(secrets.token_urlsafe(32))"`.
3. **Add the runner token to GitHub:** repository **Settings → Secrets and
   variables → Actions → New repository secret**, named
   `HYDROFED_RUNNER_TOKEN`.
4. **Add these to the HydroFed API service on Render**, then redeploy:

   ```text
   HYDROFED_GITHUB_REPO=Steven-Yabann/HydroFed
   HYDROFED_GITHUB_TOKEN=<the token from step 1>
   HYDROFED_RUNNER_TOKEN=<the same runner token as step 3>
   HYDROFED_PUBLIC_API_URL=https://hydrofed.onrender.com
   ```

   `HYDROFED_GITHUB_REF` chooses the branch whose code trains (default
   `main`).

The workflow file must be on the default branch before GitHub accepts a
dispatch. A 20-round run takes several minutes on GitHub's CPU runners, plus a
few minutes to install dependencies. The repository is public, so the runner
minutes are free. Only one run is in progress at a time.

The results arrive only if GitHub can reach the API, so this needs the public
https address. A local API trains on its own machine instead (unset
`HYDROFED_GITHUB_REPO`).

Training runs record their results; they do not change which model scores
the plant alerts. Those come from the replay feed, built offline with
`make operator-feed publish`.

The API health endpoint is:

```text
https://hydrofed.onrender.com/api/health
```

### Email notifications for messages

The API can email people when a plant conversation has something new for
them. A message from an operator goes to every administrator; a message from
an administrator goes to that plant's operator. Each person gets one email per
conversation and none after it until they read the conversation. Anyone can
turn the emails off under **Your account**.

Add these to the **HydroFed** API service and redeploy. Choose one way to send.

**Gmail over SMTP.** Render blocks SMTP ports on free instances only, and the
API service is on a paid instance because it has a disk. Turn on 2-Step
Verification for the Google account, create an app password at
<https://myaccount.google.com/apppasswords>, then set:

```text
HYDROFED_EMAIL_FROM=HydroFed <your.address@gmail.com>
HYDROFED_SMTP_HOST=smtp.gmail.com
HYDROFED_SMTP_PORT=587
HYDROFED_SMTP_USERNAME=your.address@gmail.com
HYDROFED_SMTP_PASSWORD=the 16-letter app password
```

**Brevo over HTTPS.** Use this if SMTP is blocked. Create a free Brevo
account, verify the sender address under **Senders, domains and dedicated
IPs**, create an API key under **SMTP and API**, then set:

```text
HYDROFED_EMAIL_FROM=HydroFed <your.address@gmail.com>
HYDROFED_BREVO_API_KEY=the API key
```

If both are set, Brevo is used. The links in the emails use
`HYDROFED_FRONTEND_URL`, so it must be the dashboard's public address. A
failed email is logged and never stops the message itself.

### Frontend static site

```text
Service name: hydrofed-dashboard
Branch: main
Root directory: frontend
Build command: npm ci && npm run build
Publish directory: dist
```

Frontend build-time environment variable:

```text
VITE_API_BASE_URL=https://hydrofed.onrender.com
```

The dashboard uses hash routes (`#/fleet`), so the static site needs no
rewrite rules. `frontend/package.json` asks for Node 22.12 or newer, which
Vite requires.

The public dashboard is:

```text
https://hydrofed-dashboard.onrender.com
```

### Releasing

Both services build from `main`. Changes are merged into `main` through pull
requests, and GitHub Actions runs the tests and the frontend build on each one.
With auto-deploy on, Render rebuilds both services when a pull request is
merged.

Versions used for reported results are tagged (for example `v1.0.0`). Each
tag's GitHub release carries the comparison result files from
`artifacts/comparison/`, which are git-ignored.

## Importing experiment history

A new Render disk starts with an empty SQLite database. Create a consistent
local backup before uploading:

```bash
.venv/bin/python -c 'import sqlite3; source = sqlite3.connect("artifacts/experiments.db"); backup = sqlite3.connect("/tmp/hydrofed-render-backup.db"); source.backup(backup); backup.close(); source.close()'
```

Upload the backup with the SSH address shown in the Render service settings:

```bash
scp -s /tmp/hydrofed-render-backup.db \
  <render-service>@ssh.<render-region>.render.com:/data/experiments.import.db
```

Then atomically replace the hosted database:

```bash
ssh <render-service>@ssh.<render-region>.render.com \
  "mv /data/experiments.import.db /data/experiments.db"
```

Replace the angle-bracket placeholders with the actual Render SSH details.
Using a temporary filename prevents a partially transferred file from becoming
the active database.

## Deployment verification

After a release, confirm that:

1. `https://hydrofed.onrender.com/api/health` returns `{"status":"ok"}`.
2. The administrator can sign in at the dashboard.
3. **Fleet overview** shows five plants and **Alerts** lists alerts.
4. **Model performance** shows the training regimes and model families.
5. An operator added from **Users & access** and assigned to a plant sees
   only **My plant**, **Alerts** and **Messages**, all limited to that plant.
6. A message the operator sends appears, unread, in the administrator's
   **Messages**, and the reply reaches the operator.
7. With email set up, that first operator message emails the administrators,
   with a link that opens the conversation.
8. An alert review and the messages survive a restart of the API service.

## Current limitations

- Plant telemetry is a replay of SKAB recordings scored offline by the
  federated model, not a live connection to plant sensors.
- The hosted dashboard trains on GitHub's CPU runners. Long GPU experiments
  still belong in Google Colab or Modal.
- A training run does not replace the model behind the plant alerts; the
  replay feed is rebuilt offline.
- SQLite supports one API instance only.
