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
The persistent disk preserves saved experiment runs and evaluation metrics when
the API restarts or redeploys.

## Local Docker Compose verification

From the repository root:

```bash
cp deployment.env.example deployment.env
docker compose --env-file deployment.env -f docker-compose.deploy.yml up --build
```

Open `http://localhost:8080` and confirm that the dashboard reports
`API connected`.

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
```

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

The public dashboard is:

```text
https://hydrofed-dashboard.onrender.com
```

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

The public deployment was verified by confirming that:

1. The dashboard reports `API connected`.
2. A saved federated run is selectable.
3. Five plant evaluation results are displayed.
4. The same run remains visible after restarting the Render API service.

## Current limitations

- The dashboard is read-only; training is started from the command line.
- Role workspaces are interface views and do not yet enforce authentication or
  role-based authorization.
- SQLite supports one API instance only.
- Long-running GPU experiments should run in Google Colab or Modal, then save
  results for dashboard display.