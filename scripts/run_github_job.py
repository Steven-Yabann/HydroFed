"""Run one dashboard training job on GitHub Actions and report it to the API.

The ``federated-training.yml`` workflow calls this after it has built the SKAB
plant partitions. It checks the settings the dashboard sent, runs
``scripts/run_experiment.py`` with them, then posts the run and its results to
``<callback>/api/training/report`` with the shared runner token. If training
fails, the failure is reported too, so the dashboard does not wait forever.

    python scripts/run_github_job.py --job-id abc123 --settings '{"config": {...}, "clients": [...]}' \\
        --callback-url https://hydrofed.onrender.com --partitions data/partitions_skab --output-dir artifacts/job

The runner token is read from ``HYDROFED_RUNNER_TOKEN``. Without a callback
URL the run is only saved under ``--output-dir``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import httpx

from hydrofed.jobs import PROJECT_ROOT, experiment_arguments
from hydrofed.operations import OperationsError, validate_federation_settings
from hydrofed.storage import ExperimentStore

PLANT_ID = re.compile(r"^plant_\d+$")


def read_settings(raw: str, partitions: Path) -> tuple[dict, list[str]]:
    """Validate the dashboard's settings and plants before anything runs."""
    settings = json.loads(raw)
    config = validate_federation_settings(settings.get("config", {}))
    # GitHub's hosted runners have no GPU.
    config["device"] = "cpu"
    clients = settings.get("clients", [])
    if not isinstance(clients, list) or len(clients) < 2:
        raise ValueError("Federated training needs at least two participating plants.")
    for client in clients:
        if not isinstance(client, str) or not PLANT_ID.match(client):
            raise ValueError(f"Not a plant ID: {client!r}")
        if not (partitions / f"{client}.npz").exists():
            raise ValueError(f"No partition for {client} under {partitions}")
    return config, clients


def report(callback_url: str, payload: dict, attempts: int = 4) -> None:
    """Post the report, retrying in case the API is restarting."""
    token = os.environ.get("HYDROFED_RUNNER_TOKEN", "")
    if not token:
        raise SystemExit("HYDROFED_RUNNER_TOKEN is not set, so the results cannot be reported.")
    url = f"{callback_url.rstrip('/')}/api/training/report"
    for attempt in range(1, attempts + 1):
        try:
            response = httpx.post(url, json=payload, headers={"Authorization": f"Bearer {token}"}, timeout=60)
        except httpx.HTTPError as error:
            message = str(error)
        else:
            if response.status_code < 300:
                print(f"Reported to {url}: {response.json()}")
                return
            message = f"{response.status_code} {response.text[:300]}"
            if response.status_code < 500:
                raise SystemExit(f"The API refused the report: {message}")
        print(f"Report attempt {attempt} failed: {message}", flush=True)
        if attempt < attempts:
            time.sleep(15 * attempt)
    raise SystemExit("The results could not be reported. They are saved in the workflow's artifacts.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--settings", required=True, help="JSON with 'config' and 'clients'")
    parser.add_argument("--callback-url", default="")
    parser.add_argument("--partitions", type=Path, default=PROJECT_ROOT / "data" / "partitions_skab")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "github_job")
    args = parser.parse_args()

    if not re.fullmatch(r"[0-9a-f]{6,32}", args.job_id):
        raise SystemExit("The job ID should be the dashboard's hexadecimal job ID.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    database = args.output_dir / "experiments.db"
    payload: dict = {"job_id": args.job_id}

    try:
        config, clients = read_settings(args.settings, args.partitions)
    except (ValueError, OperationsError, json.JSONDecodeError) as error:
        payload.update(status="failed", error=f"The settings were rejected: {error}")
    else:
        print(f"Training plants {', '.join(clients)} with {json.dumps(config)}", flush=True)
        arguments = experiment_arguments(config, clients, database, args.partitions, args.output_dir)
        code = subprocess.call(arguments, cwd=PROJECT_ROOT, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        runs = ExperimentStore(database).list_runs(experiment="federated", limit=1) if database.exists() else []
        if runs:
            payload.update(ExperimentStore(database).export_run(runs[0]["run_id"]))
        if code == 0 and runs and runs[0]["status"] == "completed":
            payload["status"] = "completed"
        else:
            reason = runs[0]["error_message"] if runs and runs[0]["error_message"] else f"exit code {code}"
            payload.update(status="failed", error=f"Training failed: {reason}")

    (args.output_dir / "report.json").write_text(json.dumps(payload, indent=2))
    print(f"Run {payload['status']}" + (f": {payload['error']}" if payload.get("error") else ""), flush=True)
    if args.callback_url:
        report(args.callback_url, payload)
    if payload["status"] != "completed":
        sys.exit(1)


if __name__ == "__main__":
    main()
