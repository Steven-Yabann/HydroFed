import importlib.util
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from hydrofed.api import create_app
from hydrofed.auth import UserStore
from hydrofed.jobs import GitHubTrainingLauncher, JobError, run_title
from hydrofed.storage import ExperimentStore
from test_dashboard_api import write_feed

PASSWORD = "correct horse battery"
RUNNER_TOKEN = "runner-secret"
CONFIG = {"rounds": 2, "local_epochs": 1}


class FakeGitHub:
    """Answer the few GitHub Actions endpoints the launcher calls."""

    def __init__(self):
        self.dispatches = []
        self.cancelled = []
        self.run = None

    def handler(self, request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer gh-token"
        path = request.url.path
        if request.method == "POST" and path.endswith("/dispatches"):
            body = json.loads(request.content)
            self.dispatches.append(body)
            job_id = body["inputs"]["job_id"]
            self.run = {
                "id": 77, "display_title": run_title(job_id), "status": "queued", "conclusion": None,
                "html_url": "https://github.com/owner/repo/actions/runs/77",
            }
            return httpx.Response(204)
        if path.endswith("/runs") and request.method == "GET":
            return httpx.Response(200, json={"workflow_runs": [self.run] if self.run else []})
        if path.endswith("/runs/77/jobs"):
            return httpx.Response(200, json={"jobs": [{"steps": [
                {"name": "Set up job", "status": "completed", "conclusion": "success"},
                {"name": "Install dependencies", "status": "completed", "conclusion": "success"},
                {"name": "Train and report", "status": "in_progress", "conclusion": None},
            ]}]})
        if path.endswith("/runs/77/cancel"):
            self.cancelled.append(77)
            return httpx.Response(202)
        if path.endswith("/runs/77"):
            return httpx.Response(200, json=self.run)
        return httpx.Response(404, json={"message": "Not Found"})


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("HYDROFED_SECRET_KEY", "test-secret")
    database = tmp_path / "experiments.db"
    UserStore(database).create("admin@example.com", "Ada Admin", "admin", PASSWORD)
    github = FakeGitHub()
    launcher = GitHubTrainingLauncher(
        database, "owner/repo", "gh-token", RUNNER_TOKEN, "https://api.example.com",
        http=httpx.Client(transport=httpx.MockTransport(github.handler)), refresh_seconds=0,
    )
    client = TestClient(create_app(database, feed_path=write_feed(tmp_path / "feed"), launcher=launcher))
    token = client.post("/api/auth/login", json={"email": "admin@example.com", "password": PASSWORD}).json()["token"]
    client.headers["Authorization"] = f"Bearer {token}"
    return client, launcher, github, database


def exported_run(run_id="run123"):
    return {
        "run": {
            "run_id": run_id, "experiment": "federated", "status": "completed",
            "config": {"rounds": 2, "local_epochs": 1, "clients": ["plant_1", "plant_2"]},
            "started_at": "2026-11-25 10:00:00", "finished_at": "2026-11-25 10:05:00", "error_message": None,
        },
        "results": [
            {"experiment": "federated", "client_id": "plant_1", "round": None, "kind": "evaluation",
             "metrics": {"f1": 0.7}, "created_at": "2026-11-25 10:05:00"},
            {"experiment": "federated", "client_id": None, "round": 1, "kind": "round",
             "metrics": {"loss": 0.1}, "created_at": "2026-11-25 10:02:00"},
        ],
    }


def test_start_dispatches_the_workflow_with_saved_settings(setup):
    client, launcher, github, _ = setup

    status = client.post("/api/admin/training").json()

    assert status["backend"] == "github"
    assert status["job"]["state"] == "queued"
    inputs = github.dispatches[0]["inputs"]
    assert github.dispatches[0]["ref"] == "main"
    assert inputs["callback_url"] == "https://api.example.com"
    settings = json.loads(inputs["settings"])
    assert settings["clients"] == ["plant_1", "plant_2"]
    assert settings["config"]["rounds"] == 20


def test_status_follows_the_workflow_run(setup, tmp_path):
    client, launcher, github, _ = setup
    launcher.start(CONFIG, ["plant_1", "plant_2"], {"name": "Ada"})

    github.run["status"] = "in_progress"
    job = client.get("/api/admin/training").json()["job"]

    assert job["state"] == "running"
    assert job["url"].endswith("/runs/77")
    assert [step["name"] for step in job["steps"]] == ["Install dependencies", "Train and report"]
    with pytest.raises(JobError, match="already in progress"):
        launcher.start(CONFIG, ["plant_1", "plant_2"], {"name": "Ada"})


def test_report_saves_the_run_and_finishes_the_job(setup):
    client, launcher, github, database = setup
    job_id = launcher.start(CONFIG, ["plant_1", "plant_2"], {"name": "Ada"})["job"]["job_id"]
    report = {"job_id": job_id, "status": "completed", **exported_run()}

    refused = client.post("/api/training/report", json=report, headers={"Authorization": "Bearer wrong"})
    assert refused.status_code == 401

    accepted = client.post("/api/training/report", json=report, headers={"Authorization": f"Bearer {RUNNER_TOKEN}"})
    assert accepted.json() == {"job_id": job_id, "state": "completed", "run_id": "run123"}
    runs = client.get("/api/runs?experiment=federated").json()["runs"]
    assert runs[0]["run_id"] == "run123" and runs[0]["result_count"] == 2

    again = client.post("/api/training/report", json=report, headers={"Authorization": f"Bearer {RUNNER_TOKEN}"})
    assert again.status_code == 409
    github.run.update(status="completed", conclusion="success")
    assert client.get("/api/admin/training").json()["job"]["state"] == "completed"


def test_unknown_jobs_cannot_report(setup):
    client, *_ = setup
    response = client.post(
        "/api/training/report", json={"job_id": "nope", "status": "completed", **exported_run()},
        headers={"Authorization": f"Bearer {RUNNER_TOKEN}"},
    )
    assert response.status_code == 409


def test_a_run_that_finishes_without_reporting_is_failed(setup):
    client, launcher, github, _ = setup
    launcher.start(CONFIG, ["plant_1", "plant_2"], {"name": "Ada"})
    github.run.update(status="completed", conclusion="success")

    job = client.get("/api/admin/training").json()["job"]

    assert job["state"] == "failed"
    assert "did not reach" in job["error"]


def test_stop_cancels_the_workflow_run(setup):
    client, launcher, github, _ = setup
    launcher.start(CONFIG, ["plant_1", "plant_2"], {"name": "Ada"})

    job = client.delete("/api/admin/training").json()["job"]

    assert github.cancelled == [77]
    assert job["state"] == "cancelled"


def test_github_training_needs_a_public_https_api(tmp_path):
    launcher = GitHubTrainingLauncher(
        tmp_path / "db.sqlite", "owner/repo", "gh-token", RUNNER_TOKEN, "http://localhost:8000",
        http=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(500))),
    )
    available, reason = launcher.available()
    assert not available and "HYDROFED_PUBLIC_API_URL" in reason
    with pytest.raises(JobError):
        launcher.start(CONFIG, ["plant_1", "plant_2"], {"name": "Ada"})


def test_runs_round_trip_between_databases(tmp_path):
    source = ExperimentStore(tmp_path / "source.db")
    run_id = source.create_run("federated", {"rounds": 2})
    source.record("federated", "evaluation", {"f1": 0.5}, client_id="plant_1", run_id=run_id)
    source.complete_run(run_id)

    target = ExperimentStore(tmp_path / "target.db")
    target.import_run(source.export_run(run_id))
    target.import_run(source.export_run(run_id))  # importing twice keeps one copy

    runs = target.list_runs()
    assert [(run["run_id"], run["status"], run["result_count"]) for run in runs] == [(run_id, "completed", 1)]


def _runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_github_job.py"
    spec = importlib.util.spec_from_file_location("run_github_job", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_checks_settings_before_training(tmp_path):
    runner = _runner()
    for plant in ("plant_1", "plant_2"):
        (tmp_path / f"{plant}.npz").write_bytes(b"")

    config, clients = runner.read_settings(
        json.dumps({"config": {"rounds": 3, "device": "mps"}, "clients": ["plant_1", "plant_2"]}), tmp_path,
    )
    assert config["rounds"] == 3 and config["device"] == "cpu" and clients == ["plant_1", "plant_2"]

    bad = [
        {"config": {"rounds": 9999}, "clients": ["plant_1", "plant_2"]},
        {"config": {"surprise": 1}, "clients": ["plant_1", "plant_2"]},
        {"config": {}, "clients": ["plant_1"]},
        {"config": {}, "clients": ["plant_1", "; rm -rf /"]},
        {"config": {}, "clients": ["plant_1", "plant_9"]},
    ]
    for settings in bad:
        with pytest.raises(Exception):
            runner.read_settings(json.dumps(settings), tmp_path)
