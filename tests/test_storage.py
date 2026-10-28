import sqlite3

from hydrofed.storage import ExperimentStore


def test_run_groups_its_saved_results(tmp_path):
    store = ExperimentStore(tmp_path / "experiments.db")

    run_id = store.create_run(
        "federated",
        {"rounds": 5, "seed": 42},
    )

    store.record(
        "federated",
        "evaluation",
        {"f1": 0.48},
        client_id="plant_1",
        run_id=run_id,
    )
    store.complete_run(run_id)

    runs = store.list_runs(experiment="federated")
    results = store.list_results(run_id=run_id)

    assert len(runs) == 1
    assert runs[0]["run_id"] == run_id
    assert runs[0]["status"] == "completed"
    assert runs[0]["config"] == {"rounds": 5, "seed": 42}
    assert runs[0]["result_count"] == 1

    assert len(results) == 1
    assert results[0]["run_id"] == run_id
    assert results[0]["metrics"]["f1"] == 0.48


def test_existing_database_gains_run_id_column(tmp_path):
    database_path = tmp_path / "legacy.db"

    with sqlite3.connect(database_path) as database:
        database.execute(
            """
            CREATE TABLE results (
                id INTEGER PRIMARY KEY,
                experiment TEXT,
                client_id TEXT,
                round INTEGER,
                kind TEXT,
                metrics_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

    ExperimentStore(database_path)

    with sqlite3.connect(database_path) as database:
        columns = {
            row[1]
            for row in database.execute("PRAGMA table_info(results)")
        }

    assert "run_id" in columns