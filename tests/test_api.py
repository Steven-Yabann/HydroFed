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