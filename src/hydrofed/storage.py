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
