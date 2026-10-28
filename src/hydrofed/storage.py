"""Small SQLite experiment store shared by baseline and federation."""
from __future__ import annotations
import json, sqlite3
from pathlib import Path

SCHEMA = """CREATE TABLE IF NOT EXISTS results(id INTEGER PRIMARY KEY, experiment TEXT, client_id TEXT, round INTEGER, kind TEXT, metrics_json TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP);"""

class ExperimentStore:
    def __init__(self, path: str | Path):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as db: db.execute(SCHEMA)
    def record(self, experiment: str, kind: str, metrics: dict, client_id: str | None = None, round_number: int | None = None):
        with sqlite3.connect(self.path) as db:
            db.execute("INSERT INTO results(experiment,client_id,round,kind,metrics_json) VALUES(?,?,?,?,?)",
                       (experiment, client_id, round_number, kind, json.dumps(metrics)))

    def list_results(
        self,
        experiment: str | None = None,
        kind: str | None = None,
        client_id: str | None = None,
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

        where_clause = (
            f"WHERE {' AND '.join(clauses)}"
            if clauses
            else ""
        )

        query = f"""
            SELECT id, experiment, client_id, round, kind, metrics_json, created_at
            FROM results
            {where_clause}
            ORDER BY id DESC
            LIMIT ?
        """
        parameters.append(limit)

        with sqlite3.connect(self.path) as db:
            rows = db.execute(query, parameters).fetchall()

        return [
            {
                "id": row[0],
                "experiment": row[1],
                "client_id": row[2],
                "round": row[3],
                "kind": row[4],
                "metrics": json.loads(row[5]),
                "created_at": row[6],
            }
            for row in rows
        ]
