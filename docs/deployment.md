# Deploying HydroFed

## What is deployed

The deployment contains two containers:

- `frontend`: an Nginx server hosting the production React build.
- `api`: the FastAPI service that reads the SQLite experiment database.

Nginx proxies browser requests from `/api/...` to the FastAPI container. This
means a hosted browser page and its API share one origin, so the production
frontend does not need to contain a hard-coded API URL.

The named Docker volume `hydrofed-data` persists `/data/experiments.db` across
container replacement. SQLite is appropriate for this single API instance. Do
not scale the API to multiple replicas while it uses SQLite.

## Local production-like verification

From the repository root:

```bash
cp deployment.env.example deployment.env
docker compose --env-file deployment.env -f docker-compose.deploy.yml up --build
```

Open `http://localhost:8080` and verify that the dashboard reports `API
connected`.

In another terminal, run:

```bash
curl http://localhost:8080/api/health
```

The expected response is:

```json
{"status":"ok"}
```

## Importing existing experiment history

The named Docker volume starts with a new SQLite database. To display results
already stored in the local development database, start the deployment and
then copy the database into the API container's persistent `/data` volume:

docker compose --env-file deployment.env -f docker-compose.deploy.yml cp \
  artifacts/experiments.db api:/data/experiments.db

docker compose --env-file deployment.env -f docker-compose.deploy.yml \
  exec --user root api \
  chown hydrofed:hydrofed /data/experiments.db

docker compose --env-file deployment.env -f docker-compose.deploy.yml \
  restart api

Refresh the dashboard after the API restarts. A new hosted database will show
an empty dashboard until a database backup is imported or a future hosted
training worker records a new experiment run.

To stop the local deployment while retaining the database volume:

```bash
docker compose --env-file deployment.env -f docker-compose.deploy.yml down
```

To delete the deployment data as well, use `down --volumes`. Do this only when
the persisted experiment database is no longer needed.

## Hosting requirements

Choose a platform that can run Docker containers and provide persistent disk
storage. Configure these environment variables on the hosting platform:

| Variable | Value |
| --- | --- |
| `HYDROFED_DB_PATH` | `/data/experiments.db` |
| `HYDROFED_ALLOWED_ORIGINS` | The public dashboard URL, for example `https://hydrofed.example` |

Mount persistent storage at `/data`. The frontend should be publicly exposed;
the API container should remain internal when Nginx proxies `/api` on the same
host.

## Deployment smoke test

1. Open the public dashboard URL.
2. Confirm the API status indicator is connected.
3. Select a completed federated run.
4. If an experiment database was imported, confirm five plant evaluation rows
   appear.
5. Download the research CSV export.
6. Restart the containers and confirm the selected run is still listed.
