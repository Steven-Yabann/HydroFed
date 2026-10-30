from fastapi.testclient import TestClient

from hydrofed.api import create_app
from hydrofed.storage import ExperimentStore


def test_health_endpoint_returns_ok(tmp_path):
    client = TestClient(create_app(tmp_path / "experiments.db"))

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_results_endpoint_filters_saved_experiment_data(tmp_path):
    database = tmp_path / "experiments.db"
    store = ExperimentStore(database)

    store.record(
        "federated",
        "evaluation",
        {"f1": 0.48, "auc_roc": 0.61},
        client_id="plant_1",
    )
    store.record(
        "baseline",
        "evaluation",
        {"f1": 0.32},
        client_id="plant_1",
    )

    client = TestClient(create_app(database))
    response = client.get("/api/results?experiment=federated")

    assert response.status_code == 200

    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["experiment"] == "federated"
    assert results[0]["client_id"] == "plant_1"
    assert results[0]["metrics"]["f1"] == 0.48

def test_dashboard_origin_is_allowed(tmp_path):
    client = TestClient(create_app(tmp_path / "experiments.db"))

    response = client.options(
        "/api/results",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert (
        response.headers["access-control-allow-origin"]
        == "http://localhost:5173"
    )

def test_api_allows_a_configured_deployment_origin(tmp_path, monkeypatch):
    origin = "https://dashboard.hydrofed.example"
    monkeypatch.setenv("HYDROFED_ALLOWED_ORIGINS", origin)

    client = TestClient(create_app(tmp_path / "experiments.db"))

    response = client.options(
        "/api/results",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin

def test_runs_endpoint_and_run_filter_return_one_experiment(tmp_path):
    database = tmp_path / "experiments.db"
    store = ExperimentStore(database)

    first_run = store.create_run(
        "federated",
        {"rounds": 2, "seed": 42},
    )
    second_run = store.create_run(
        "federated",
        {"rounds": 5, "seed": 43},
    )

    store.record(
        "federated",
        "evaluation",
        {"f1": 0.25},
        client_id="plant_1",
        run_id=first_run,
    )
    store.record(
        "federated",
        "evaluation",
        {"f1": 0.50},
        client_id="plant_1",
        run_id=second_run,
    )
    store.complete_run(first_run)
    store.complete_run(second_run)

    client = TestClient(create_app(database))

    runs_response = client.get("/api/runs?experiment=federated")
    results_response = client.get(f"/api/results?run_id={first_run}")

    assert runs_response.status_code == 200
    assert len(runs_response.json()["runs"]) == 2

    results = results_response.json()["results"]
    assert len(results) == 1
    assert results[0]["run_id"] == first_run
    assert results[0]["metrics"]["f1"] == 0.25