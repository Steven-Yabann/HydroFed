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
Branch: feature/deployment
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

Training from the dashboard is switched off in the image
(`HYDROFED_ALLOW_TRAINING=0`): it has no training data and a web service is
the wrong place for long simulations. Administrators can still change the
federation settings and plant nodes.

The API health endpoint is:

```text
https://hydrofed.onrender.com/api/health
```

### Frontend static site

```text
Service name: hydrofed-dashboard
Branch: feature/deployment
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

Both services build from `feature/deployment`. After a change is merged to
`main`, release it by moving that branch to `main`:

```bash
git push origin main:feature/deployment
```

Render then rebuilds both services.

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
4. **Model performance** shows all four comparison sections.
5. An operator added from **Users & access** sees only Fleet, Alerts and
   Model performance.
6. An alert review survives a restart of the API service.

## Current limitations

- Plant telemetry is a replay of SKAB recordings scored offline by the
  federated model, not a live connection to plant sensors.
- Training runs are started locally or in Colab, not from the hosted dashboard.
- SQLite supports one API instance only.
- Long-running GPU experiments should run in Google Colab or Modal, then save
  results for dashboard display.
