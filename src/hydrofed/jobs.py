"""Start federated training runs from the dashboard.

An administrator's saved federation settings become arguments to
``scripts/run_experiment.py``. Two launchers run it:

- ``TrainingLauncher`` starts it as a local process that records its run,
  rounds and evaluation straight into the API's database. It needs the
  training scripts and SKAB partitions, so the deployed API image cannot use
  it.
- ``GitHubTrainingLauncher`` dispatches the ``federated-training.yml``
  workflow on GitHub Actions. The workflow downloads SKAB, builds the plant
  partitions, trains, and posts the run back to ``/api/training/report``
  with a shared runner token. Jobs are kept in the database, so their status
  survives an API restart.

Only one run is in progress at a time with either launcher.
"""

from __future__ import annotations

import hmac
import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "run_experiment.py"
DEFAULT_PARTITIONS = PROJECT_ROOT / "data" / "partitions_skab"


class JobError(Exception):
    """A training run that cannot be started."""


def experiment_arguments(config: dict, clients: list[str], database: Path, partitions: Path, log_dir: Path) -> list[str]:
    """Translate dashboard federation settings into run_experiment.py arguments."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return [
        sys.executable, str(SCRIPT),
        "--partitions", str(partitions),
        "--clients", ",".join(clients),
        "--fraction-fit", str(config["fraction_fit"]),
        "--rounds", str(config["rounds"]),
        "--epochs", str(config["local_epochs"]),
        "--batch-size", str(config["batch_size"]),
        "--learning-rate", str(config["learning_rate"]),
        "--hidden-size", str(config["hidden_size"]),
        "--latent-size", str(config["latent_size"]),
        "--num-layers", str(config["num_layers"]),
        "--dropout", str(config["dropout"]),
        "--seed", str(config["seed"]),
        "--device", config["device"],
        "--db", str(database),
        "--output", str(log_dir / f"dashboard_{stamp}.pt"),
        "--metrics-port", os.environ.get("HYDROFED_TRAINING_METRICS_PORT", "9101"),
        "--evaluate",
        "--evaluation-protocol", "skab",
        "--skab-scaling", "plant",
        "--skab-score", "hybrid",
        "--skab-training-policy", "normal-only",
        "--skab-exclude-anomaly-free",
        # The frozen alert policy behind the reported results: alert on the
        # first sequence over threshold (the script's default is 10).
        "--skab-persistence", "1",
    ]


class TrainingLauncher:
    def __init__(self, database: Path, partitions: Path | None = None, log_dir: Path | None = None):
        self.database = Path(database)
        self.partitions = Path(partitions or os.environ.get("HYDROFED_PARTITIONS", DEFAULT_PARTITIONS))
        self.log_dir = Path(log_dir or self.database.parent / "training_logs")
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._job: dict | None = None

    def available(self) -> tuple[bool, str]:
        if os.environ.get("HYDROFED_ALLOW_TRAINING", "1") != "1":
            return False, "Training from the dashboard is switched off on this server."
        if not SCRIPT.exists():
            return False, "This server does not include the training scripts."
        if not any(self.partitions.glob("plant_*.npz")):
            return False, f"No plant partitions found under {self.partitions}."
        return True, ""

    backend = "local"

    def status(self) -> dict:
        available, reason = self.available()
        job = dict(self._job) if self._job else None
        if job and self._process is not None:
            code = self._process.poll()
            job["state"] = "running" if code is None else ("completed" if code == 0 else "failed")
            job["exit_code"] = code
            job["log_tail"] = self.log_tail()
        return {"available": available, "reason": reason, "backend": self.backend, "job": job}

    def receive_report(self, token: str | None, report: dict) -> dict:
        raise JobError("This server trains locally and does not accept run reports.")

    def log_tail(self, lines: int = 40) -> list[str]:
        if not self._job:
            return []
        path = Path(self._job["log_path"])
        if not path.exists():
            return []
        return path.read_text(errors="replace").splitlines()[-lines:]

    def start(self, config: dict, clients: list[str], user: dict) -> dict:
        available, reason = self.available()
        if not available:
            raise JobError(reason)
        if len(clients) < 2:
            raise JobError("Federated training needs at least two participating plants.")
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise JobError("A training run is already in progress.")
            self.log_dir.mkdir(parents=True, exist_ok=True)
            arguments = experiment_arguments(config, clients, self.database, self.partitions, self.log_dir)
            log_path = self.log_dir / f"run_{time.strftime('%Y%m%d-%H%M%S')}.log"
            log = log_path.open("w")
            self._process = subprocess.Popen(
                arguments, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            log.close()
            self._job = {
                "pid": self._process.pid,
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "started_by": user["name"],
                "clients": clients,
                "config": config,
                "log_path": str(log_path),
            }
        return self.status()

    def stop(self) -> dict:
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                raise JobError("No training run is in progress.")
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
        return self.status()


# --- GitHub Actions ---------------------------------------------------------

WORKFLOW_FILE = "federated-training.yml"
ACTIVE_STATES = ("queued", "running")
FINISHED_STATES = ("completed", "failed", "cancelled")

JOBS_SCHEMA = """
CREATE TABLE IF NOT EXISTS training_jobs (
    job_id TEXT PRIMARY KEY,
    backend TEXT NOT NULL,
    state TEXT NOT NULL,
    config_json TEXT NOT NULL,
    clients_json TEXT NOT NULL,
    started_by TEXT NOT NULL,
    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at TEXT,
    github_run_id INTEGER,
    url TEXT,
    run_id TEXT,
    error TEXT
);
"""
JOB_COLUMNS = (
    "job_id", "backend", "state", "config_json", "clients_json", "started_by", "started_at",
    "finished_at", "github_run_id", "url", "run_id", "error",
)


def run_title(job_id: str) -> str:
    """The workflow's run-name, used to find the run a dispatch started."""
    return f"Federated training {job_id}"


class GitHubTrainingLauncher:
    """Run federated training on GitHub Actions and receive its results."""

    backend = "github"

    def __init__(
        self,
        database: Path,
        repository: str,
        token: str,
        runner_token: str,
        callback_url: str,
        ref: str = "main",
        api_url: str = "https://api.github.com",
        http: httpx.Client | None = None,
        refresh_seconds: float = 10.0,
        clock=time.time,
    ):
        from hydrofed.storage import ExperimentStore

        self.database = Path(database)
        self.repository = repository
        self.ref = ref
        self.api_url = api_url.rstrip("/")
        self.runner_token = runner_token
        self.callback_url = callback_url.rstrip("/")
        self.refresh_seconds = refresh_seconds
        self.clock = clock
        self.store = ExperimentStore(self.database)
        self.http = http or httpx.Client(timeout=15)
        self.http.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })
        self._lock = threading.Lock()
        self._checked: dict[str, float] = {}
        self._steps: dict[str, list[dict]] = {}
        with sqlite3.connect(self.database) as db:
            db.execute(JOBS_SCHEMA)

    # --- jobs table ---

    def _latest(self) -> dict | None:
        with sqlite3.connect(self.database) as db:
            row = db.execute(
                f"SELECT {', '.join(JOB_COLUMNS)} FROM training_jobs ORDER BY started_at DESC, rowid DESC LIMIT 1"
            ).fetchone()
        return self._row(row)

    def _job(self, job_id: str) -> dict | None:
        with sqlite3.connect(self.database) as db:
            row = db.execute(
                f"SELECT {', '.join(JOB_COLUMNS)} FROM training_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        return self._row(row)

    @staticmethod
    def _row(row: tuple | None) -> dict | None:
        if row is None:
            return None
        job = dict(zip(JOB_COLUMNS, row))
        job["config"] = json.loads(job.pop("config_json"))
        job["clients"] = json.loads(job.pop("clients_json"))
        return job

    def _update(self, job_id: str, **changes) -> None:
        if not changes:
            return
        if changes.get("state") in FINISHED_STATES:
            changes.setdefault("finished_at", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(self.clock())))
        assignments = ", ".join(f"{column} = ?" for column in changes)
        with sqlite3.connect(self.database) as db:
            db.execute(f"UPDATE training_jobs SET {assignments} WHERE job_id = ?", (*changes.values(), job_id))

    # --- GitHub ---

    def _api(self, method: str, path: str, **kwargs) -> httpx.Response:
        try:
            response = self.http.request(method, f"{self.api_url}/repos/{self.repository}{path}", **kwargs)
        except httpx.HTTPError as error:
            raise JobError(f"GitHub could not be reached: {error}") from error
        if response.status_code >= 400:
            message = response.json().get("message", response.text) if response.content else response.reason_phrase
            raise JobError(f"GitHub refused the request ({response.status_code}): {message}")
        return response

    def _find_run(self, job: dict) -> dict | None:
        runs = self._api(
            "GET", f"/actions/workflows/{WORKFLOW_FILE}/runs",
            params={"event": "workflow_dispatch", "per_page": 30},
        ).json().get("workflow_runs", [])
        return next((run for run in runs if run.get("display_title") == run_title(job["job_id"])), None)

    def _refresh(self, job: dict) -> dict:
        """Bring an unfinished job up to date with its workflow run."""
        if job["state"] not in ACTIVE_STATES:
            return job
        now = self.clock()
        if now - self._checked.get(job["job_id"], 0) < self.refresh_seconds:
            return job
        self._checked[job["job_id"]] = now
        try:
            if job["github_run_id"] is None:
                run = self._find_run(job)
                if run is None:
                    return job
                self._update(job["job_id"], github_run_id=run["id"], url=run["html_url"])
            else:
                run = self._api("GET", f"/actions/runs/{job['github_run_id']}").json()
            jobs = self._api("GET", f"/actions/runs/{run['id']}/jobs").json().get("jobs", [])
            self._steps[job["job_id"]] = [
                {"name": step["name"], "status": step["status"], "conclusion": step.get("conclusion")}
                for step in (jobs[0].get("steps", []) if jobs else [])
                if not step["name"].startswith(("Set up job", "Post ", "Complete job"))
            ]
        except JobError:
            return job  # Show the last known state; GitHub may be briefly unreachable.

        changes = {}
        if run["status"] == "completed":
            conclusion = run.get("conclusion")
            if conclusion == "cancelled":
                changes["state"] = "cancelled"
            elif conclusion == "success":
                # The workflow reports before it finishes; a success without a
                # report means the results never arrived.
                latest = self._job(job["job_id"])
                if latest["state"] in ACTIVE_STATES:
                    changes.update(state="failed", error="The run finished but its results did not reach this server.")
            else:
                changes.update(state="failed", error=job["error"] or f"The workflow run ended: {conclusion}.")
        elif run["status"] == "in_progress" and job["state"] == "queued":
            changes["state"] = "running"
        self._update(job["job_id"], **changes)
        return self._job(job["job_id"])

    # --- launcher interface ---

    def available(self) -> tuple[bool, str]:
        if not self.callback_url.startswith("https://"):
            return False, (
                "GitHub Actions cannot send results back to this server: set HYDROFED_PUBLIC_API_URL "
                "to the API's public https address."
            )
        return True, ""

    def status(self) -> dict:
        available, reason = self.available()
        job = self._latest()
        if job:
            job = self._refresh(job)
            job["steps"] = self._steps.get(job["job_id"], [])
        return {
            "available": available, "reason": reason, "backend": self.backend,
            "repository": self.repository, "job": job,
        }

    def start(self, config: dict, clients: list[str], user: dict) -> dict:
        available, reason = self.available()
        if not available:
            raise JobError(reason)
        if len(clients) < 2:
            raise JobError("Federated training needs at least two participating plants.")
        with self._lock:
            latest = self._latest()
            if latest and self._refresh(latest)["state"] in ACTIVE_STATES:
                raise JobError("A training run is already in progress.")
            job_id = uuid.uuid4().hex[:12]
            self._api(
                "POST", f"/actions/workflows/{WORKFLOW_FILE}/dispatches",
                json={
                    "ref": self.ref,
                    "inputs": {
                        "job_id": job_id,
                        "settings": json.dumps({"config": config, "clients": clients}),
                        "callback_url": self.callback_url,
                    },
                },
            )
            with sqlite3.connect(self.database) as db:
                db.execute(
                    "INSERT INTO training_jobs(job_id, backend, state, config_json, clients_json, started_by, started_at) "
                    "VALUES(?, 'github', 'queued', ?, ?, ?, ?)",
                    (job_id, json.dumps(config), json.dumps(clients), user["name"],
                     time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(self.clock()))),
                )
        return self.status()

    def stop(self) -> dict:
        with self._lock:
            job = self._latest()
            if not job or job["state"] not in ACTIVE_STATES:
                raise JobError("No training run is in progress.")
            if job["github_run_id"] is None:
                run = self._find_run(job)
                if run is None:
                    raise JobError("GitHub has not started the run yet. Try again in a few seconds.")
                self._update(job["job_id"], github_run_id=run["id"], url=run["html_url"])
                job["github_run_id"] = run["id"]
            self._api("POST", f"/actions/runs/{job['github_run_id']}/cancel")
            self._update(job["job_id"], state="cancelled")
        return self.status()

    def receive_report(self, token: str | None, report: dict) -> dict:
        """Save the run a workflow reports, if the token and job match."""
        if not token or not hmac.compare_digest(token, self.runner_token):
            raise PermissionError("The runner token is not valid.")
        job = self._job(str(report.get("job_id", "")))
        if job is None:
            raise JobError("Unknown training job.")
        if job["state"] not in ACTIVE_STATES:
            raise JobError(f"This training job is already {job['state']}.")
        state = "completed" if report.get("status") == "completed" else "failed"
        run_id = None
        if report.get("run"):
            run_id = self.store.import_run({"run": report["run"], "results": report.get("results", [])})
        self._update(
            job["job_id"], state=state, run_id=run_id,
            error=None if state == "completed" else str(report.get("error") or "Training failed.")[:2000],
        )
        return {"job_id": job["job_id"], "state": state, "run_id": run_id}


def make_launcher(database: Path) -> TrainingLauncher | GitHubTrainingLauncher:
    """Use GitHub Actions when it is configured, otherwise train locally."""
    repository = os.environ.get("HYDROFED_GITHUB_REPO", "").strip()
    token = os.environ.get("HYDROFED_GITHUB_TOKEN", "").strip()
    runner_token = os.environ.get("HYDROFED_RUNNER_TOKEN", "").strip()
    if repository and token and runner_token:
        return GitHubTrainingLauncher(
            database, repository, token, runner_token,
            callback_url=os.environ.get("HYDROFED_PUBLIC_API_URL", "http://localhost:8000"),
            ref=os.environ.get("HYDROFED_GITHUB_REF", "main"),
            api_url=os.environ.get("HYDROFED_GITHUB_API_URL", "https://api.github.com"),
        )
    return TrainingLauncher(database)
