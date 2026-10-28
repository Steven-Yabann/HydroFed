"""SQLite storage for HydroFed experiment runs and their results."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from uuid import uuid4


RUNS_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiment_runs (
    run_id TEXT PRIMARY KEY,
    experiment TEXT NOT NULL,
    status TEXT NOT NULL,
    config_json TEXT NOT NULL,
    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at TEXT,
    error_message TEXT
);
"""

RESULTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY,
    run_id TEXT,
    experiment TEXT,
    client_id TEXT,
    round INTEGER,
    kind TEXT,
    metrics_json TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


class ExperimentStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.path) as database:
            database.execute(RUNS_SCHEMA)
            database.execute(RESULTS_SCHEMA)
            self._migrate_results_schema(database)

    @staticmethod
    def _migrate_results_schema(database: sqlite3.Connection) -> None:
        columns = {
            row[1]
            for row in database.execute("PRAGMA table_info(results)")
        }

        if "run_id" not in columns:
            database.execute("ALTER TABLE results ADD COLUMN run_id TEXT")

    def create_run(
        self,
        experiment: str,
        config: dict | None = None,
    ) -> str:
        if not experiment:
            raise ValueError("experiment must not be empty")

        run_id = uuid4().hex

        with sqlite3.connect(self.path) as database:
            database.execute(
                """
                INSERT INTO experiment_runs(run_id, experiment, status, config_json)
                VALUES(?, ?, ?, ?)
                """,
                (
                    run_id,
                    experiment,
                    "running",
                    json.dumps(config or {}),
                ),
            )

        return run_id

    def complete_run(
        self,
        run_id: str,
        status: str = "completed",
        error_message: str | None = None,
    ) -> None:
        if status not in {"completed", "failed"}:
            raise ValueError("status must be 'completed' or 'failed'")

        with sqlite3.connect(self.path) as database:
            cursor = database.execute(
                """
                UPDATE experiment_runs
                SET status = ?, finished_at = CURRENT_TIMESTAMP, error_message = ?
                WHERE run_id = ?
                """,
                (status, error_message, run_id),
            )

            if cursor.rowcount != 1:
                raise ValueError(f"Unknown experiment run: {run_id}")

    def record(
        self,
        experiment: str,
        kind: str,
        metrics: dict,
        client_id: str | None = None,
        round_number: int | None = None,
        run_id: str | None = None,
    ) -> None:
        with sqlite3.connect(self.path) as database:
            database.execute(
                """
                INSERT INTO results(
                    run_id,
                    experiment,
                    client_id,
                    round,
                    kind,
                    metrics_json
                )
                VALUES(?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    experiment,
                    client_id,
                    round_number,
                    kind,
                    json.dumps(metrics),
                ),
            )

    def list_runs(
        self,
        experiment: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        clauses = []
        parameters = []

        if experiment is not None:
            clauses.append("experiment_runs.experiment = ?")
            parameters.append(experiment)

        where_clause = (
            f"WHERE {' AND '.join(clauses)}"
            if clauses
            else ""
        )

        parameters.append(limit)

        query = f"""
            SELECT
                experiment_runs.run_id,
                experiment_runs.experiment,
                experiment_runs.status,
                experiment_runs.config_json,
                experiment_runs.started_at,
                experiment_runs.finished_at,
                experiment_runs.error_message,
                COUNT(results.id)
            FROM experiment_runs
            LEFT JOIN results
                ON results.run_id = experiment_runs.run_id
            {where_clause}
            GROUP BY experiment_runs.run_id
            ORDER BY experiment_runs.started_at DESC
            LIMIT ?
        """

        with sqlite3.connect(self.path) as database:
            rows = database.execute(query, parameters).fetchall()

        return [
            {
                "run_id": row[0],
                "experiment": row[1],
                "status": row[2],
                "config": json.loads(row[3]),
                "started_at": row[4],
                "finished_at": row[5],
                "error_message": row[6],
                "result_count": row[7],
            }
            for row in rows
        ]

    def list_results(
        self,
        experiment: str | None = None,
        kind: str | None = None,
        client_id: str | None = None,
        run_id: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        clauses = []
        parameters = []

        if experiment is not None:
            clauses.append("experiment = ?")
            parameters.append(experiment)

        if kind is not None:
            clauses.append("kind = ?")
            parameters.append(kind)

        if client_id is not None:
            clauses.append("client_id = ?")
            parameters.append(client_id)

        if run_id is not None:
            clauses.append("run_id = ?")
            parameters.append(run_id)

        where_clause = (
            f"WHERE {' AND '.join(clauses)}"
            if clauses
            else ""
        )

        parameters.append(limit)

        query = f"""
            SELECT
                id,
                run_id,
                experiment,
                client_id,
                round,
                kind,
                metrics_json,
                created_at
            FROM results
            {where_clause}
            ORDER BY id DESC
            LIMIT ?
        """

        with sqlite3.connect(self.path) as database:
            rows = database.execute(query, parameters).fetchall()

        return [
            {
                "id": row[0],
                "run_id": row[1],
                "experiment": row[2],
                "client_id": row[3],
                "round": row[4],
                "kind": row[5],
                "metrics": json.loads(row[6]),
                "created_at": row[7],
            }
            for row in rows
        ]