"""Plant operations data for the dashboard: replay feed, alerts and node settings.

There is no live plant, so ``scripts/build_operator_feed.py`` scores the SKAB
test recordings once and this module plays them back on the wall clock. Each
plant loops through its recordings from a fixed epoch, so every viewer and
every API process sees the same "current" second without shared state.
Alerts are contiguous runs of alerted seconds, numbered per replay cycle, so
an operator's review of one cycle's alert does not carry into the next.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

REPLAY_EPOCH = 1_793_491_200  # 2026-11-01T00:00:00Z
ALERT_MERGE_GAP = 10  # seconds of quiet that still belong to the same alert
REVIEW_STATUSES = ("open", "acknowledged", "investigating", "resolved", "false_alarm")

OPERATIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS plant_nodes (
    client_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    location TEXT NOT NULL DEFAULT '',
    capacity_mw REAL,
    participates INTEGER NOT NULL DEFAULT 1,
    notes TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_by INTEGER
);
CREATE TABLE IF NOT EXISTS alert_reviews (
    alert_id TEXT PRIMARY KEY,
    client_id TEXT NOT NULL,
    status TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    reviewed_by INTEGER,
    reviewer_name TEXT,
    reviewed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_by INTEGER
);
"""

# The settings the frozen SKAB comparison used (docs/comparison.md).
DEFAULT_FEDERATION_CONFIG = {
    "strategy": "FedAvg",
    "rounds": 20,
    "local_epochs": 5,
    "fraction_fit": 1.0,
    "batch_size": 64,
    "learning_rate": 0.001,
    "hidden_size": 64,
    "latent_size": 4,
    "num_layers": 2,
    "dropout": 0.1,
    "seed": 42,
    "device": "auto",
}
FEDERATION_LIMITS = {
    "rounds": (1, 200),
    "local_epochs": (1, 50),
    "fraction_fit": (0.2, 1.0),
    "batch_size": (8, 1024),
    "learning_rate": (1e-5, 0.1),
    "hidden_size": (8, 512),
    "latent_size": (1, 128),
    "num_layers": (1, 4),
    "dropout": (0.0, 0.5),
    "seed": (0, 2**31 - 1),
}
FEDERATION_CHOICES = {"strategy": ("FedAvg",), "device": ("auto", "cpu", "cuda", "mps")}


class OperationsError(Exception):
    """A request the operations data cannot satisfy."""


def severity(peak_score: float, threshold: float) -> str:
    if peak_score >= 4 * threshold:
        return "critical"
    if peak_score >= 2 * threshold:
        return "high"
    return "medium"


def health_state(score: float | None, threshold: float) -> str:
    if score is None:
        return "unknown"
    if score > threshold:
        return "alarm"
    if score > 0.5 * threshold:
        return "watch"
    return "normal"


@dataclass
class PlantTimeline:
    """One plant's tested recordings joined into a single replay loop."""

    plant: str
    recordings: list[dict]

    @cached_property
    def offsets(self) -> list[int]:
        offsets, total = [], 0
        for recording in self.recordings:
            offsets.append(total)
            total += len(recording["seconds"])
        return offsets

    @property
    def length(self) -> int:
        return sum(len(recording["seconds"]) for recording in self.recordings)

    def locate(self, position: int) -> tuple[int, int]:
        """Return the recording index and the index inside it for a loop position."""
        for index in range(len(self.recordings) - 1, -1, -1):
            if position >= self.offsets[index]:
                return index, position - self.offsets[index]
        return 0, 0

    @cached_property
    def episodes(self) -> list[dict]:
        """Alerted runs of seconds, merged across short quiet gaps."""
        episodes = []
        for index, recording in enumerate(self.recordings):
            alerts, scores, labels = recording["alert"], recording["score"], recording["label"]
            start = None
            last = None
            for second, alerted in enumerate(alerts):
                if alerted:
                    if start is None:
                        start = second
                    elif second - last > ALERT_MERGE_GAP:
                        episodes.append(self._episode(index, start, last, scores, labels))
                        start = second
                    last = second
            if start is not None:
                episodes.append(self._episode(index, start, last, scores, labels))
        return episodes

    def _episode(self, index: int, start: int, end: int, scores: list, labels: list) -> dict:
        recording = self.recordings[index]
        peak_at = max(range(start, end + 1), key=lambda second: scores[second])
        return {
            "recording_index": index,
            "recording": recording["id"],
            "fault_type": recording["fault_type"],
            "start": self.offsets[index] + start,
            "end": self.offsets[index] + end,
            "peak_score": scores[peak_at],
            "peak_offset": self.offsets[index] + peak_at,
            "labelled_fault": any(labels[start:end + 1]),
            "sensor_deviation": self._deviating_sensor(recording, start, peak_at),
        }

    @staticmethod
    def _deviating_sensor(recording: dict, start: int, peak_at: int) -> str:
        """Name the sensor that moved most, relative to its spread before the alert."""
        best, best_change = None, -1.0
        before = max(0, start - 60)
        for key, values in recording["sensors"].items():
            reference = values[before:start] or values[:1]
            mean = sum(reference) / len(reference)
            spread = (sum((value - mean) ** 2 for value in reference) / len(reference)) ** 0.5 or 1e-6
            change = abs(values[peak_at] - mean) / spread
            if change > best_change:
                best, best_change = key, change
        return best


class OperationsStore:
    """Replay feed, node settings, alert reviews and federation settings."""

    def __init__(self, database_path: str | Path, feed_path: str | Path, clock=time.time):
        self.path = Path(database_path)
        self.feed_path = Path(feed_path)
        self.clock = clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as database:
            database.executescript(OPERATIONS_SCHEMA)
        self._timelines: dict[str, PlantTimeline] = {}

    # --- replay feed ---

    @cached_property
    def index(self) -> dict:
        index_file = self.feed_path / "index.json"
        if not index_file.exists():
            raise OperationsError(
                "No operator feed found. Run scripts/build_operator_feed.py to create it."
            )
        return json.loads(index_file.read_text())

    @property
    def threshold(self) -> float:
        return float(self.index["source"]["threshold_mad"])

    def plant_ids(self) -> list[str]:
        return [plant["id"] for plant in self.index["plants"]]

    def timeline(self, plant: str) -> PlantTimeline:
        if plant not in self.plant_ids():
            raise OperationsError(f"Unknown plant: {plant}")
        if plant not in self._timelines:
            data = json.loads((self.feed_path / f"{plant}.json").read_text())
            self._timelines[plant] = PlantTimeline(plant, data["recordings"])
        return self._timelines[plant]

    def replay_position(self, plant: str) -> tuple[int, int, float]:
        """Return the replay cycle, loop position and wall time of that position."""
        timeline = self.timeline(plant)
        now = self.clock()
        # Stagger plants so they are not all in the same phase of their loop.
        stagger = self.plant_ids().index(plant) * 997
        elapsed = int(now - REPLAY_EPOCH) + stagger
        cycle, position = divmod(elapsed, timeline.length)
        return cycle, position, float(int(now))

    def _reading(self, timeline: PlantTimeline, position: int) -> dict:
        index, offset = timeline.locate(position)
        recording = timeline.recordings[index]
        return {
            "score": recording["score"][offset],
            "alert": bool(recording["alert"][offset]),
            "label": bool(recording["label"][offset]),
            "sensors": {key: values[offset] for key, values in recording["sensors"].items()},
            "recording": recording["id"],
            "fault_type": recording["fault_type"],
        }

    def plant_summary(self, plant: str) -> dict:
        timeline = self.timeline(plant)
        cycle, position, now = self.replay_position(plant)
        reading = self._reading(timeline, position)
        alerts = self.alerts(plant)
        open_alerts = [alert for alert in alerts if alert["status"] in ("open", "acknowledged", "investigating")]
        recent = [self._reading(timeline, max(0, position - back))["score"] for back in range(299, -1, -5)]
        node = self.node(plant)
        return {
            "id": plant,
            "name": node["display_name"],
            "location": node["location"],
            "capacity_mw": node["capacity_mw"],
            "participates": node["participates"],
            "status": health_state(reading["score"], self.threshold),
            "score": reading["score"],
            "threshold": self.threshold,
            "alert": reading["alert"],
            "sensors": reading["sensors"],
            "recording": reading["recording"],
            "updated_at": now,
            "cycle": cycle,
            "position": position,
            "loop_seconds": timeline.length,
            "open_alerts": len(open_alerts),
            "highest_open_severity": _highest([alert["severity"] for alert in open_alerts]),
            "score_trend": recent,
        }

    def fleet(self) -> dict:
        plants = [self.plant_summary(plant) for plant in self.plant_ids()]
        return {
            "source": self.index["source"],
            "sensors": self.index["sensors"],
            "plants": plants,
        }

    def telemetry(self, plant: str, seconds: int = 600) -> dict:
        timeline = self.timeline(plant)
        _, position, now = self.replay_position(plant)
        seconds = max(30, min(seconds, 3600))
        start = max(0, position - seconds + 1)
        points = {"time": [], "score": [], "alert": [], "label": [], "recording": []}
        sensors = {key: [] for key in timeline.recordings[0]["sensors"]}
        for current in range(start, position + 1):
            reading = self._reading(timeline, current)
            points["time"].append(now - (position - current))
            points["score"].append(reading["score"])
            points["alert"].append(int(reading["alert"]))
            points["label"].append(int(reading["label"]))
            points["recording"].append(reading["recording"])
            for key, value in reading["sensors"].items():
                sensors[key].append(value)
        return {
            "plant": plant,
            "threshold": self.threshold,
            "sensors_meta": self.index["sensors"],
            **points,
            "sensors": sensors,
        }

    # --- alerts ---

    def alerts(self, plant: str | None = None, limit: int = 200) -> list[dict]:
        plants = [plant] if plant else self.plant_ids()
        reviews = self._reviews()
        items = []
        for plant_id in plants:
            timeline = self.timeline(plant_id)
            cycle, position, now = self.replay_position(plant_id)
            node = self.node(plant_id)
            for number, episode in enumerate(timeline.episodes):
                if episode["start"] > position:
                    continue
                alert_id = f"{plant_id}:{cycle}:{number}"
                review = reviews.get(alert_id)
                ongoing = episode["end"] >= position
                items.append({
                    "id": alert_id,
                    "plant": plant_id,
                    "plant_name": node["display_name"],
                    "severity": severity(
                        episode["peak_score"] if not ongoing else max(
                            timeline.recordings[episode["recording_index"]]["score"][
                                episode["start"] - timeline.offsets[episode["recording_index"]]:
                                position - timeline.offsets[episode["recording_index"]] + 1
                            ]
                        ),
                        self.threshold,
                    ),
                    "started_at": now - (position - episode["start"]),
                    "ended_at": None if ongoing else now - (position - episode["end"]),
                    "ongoing": ongoing,
                    "duration_seconds": min(episode["end"], position) - episode["start"] + 1,
                    "peak_score": episode["peak_score"] if not ongoing else None,
                    "threshold": self.threshold,
                    "sensor": episode["sensor_deviation"],
                    "fault_type": episode["fault_type"],
                    "recording": episode["recording"],
                    "labelled_fault": episode["labelled_fault"],
                    "status": review["status"] if review else "open",
                    "note": review["note"] if review else "",
                    "reviewed_by": review["reviewer_name"] if review else None,
                    "reviewed_at": review["reviewed_at"] if review else None,
                })
        items.sort(key=lambda item: item["started_at"], reverse=True)
        return items[:limit]

    def _reviews(self) -> dict[str, dict]:
        with sqlite3.connect(self.path) as database:
            rows = database.execute(
                "SELECT alert_id, status, note, reviewer_name, reviewed_at FROM alert_reviews"
            ).fetchall()
        return {
            row[0]: {"status": row[1], "note": row[2], "reviewer_name": row[3], "reviewed_at": row[4]}
            for row in rows
        }

    def review_alert(self, alert_id: str, status: str, note: str, user: dict) -> dict:
        if status not in REVIEW_STATUSES:
            raise OperationsError(f"Unknown review status: {status}")
        plant = alert_id.split(":", 1)[0]
        if not any(alert["id"] == alert_id for alert in self.alerts(plant, limit=10_000)):
            raise OperationsError("That alert does not exist or has not happened yet.")
        with sqlite3.connect(self.path) as database:
            database.execute(
                """
                INSERT INTO alert_reviews(alert_id, client_id, status, note, reviewed_by, reviewer_name)
                VALUES(?, ?, ?, ?, ?, ?)
                ON CONFLICT(alert_id) DO UPDATE SET
                    status = excluded.status, note = excluded.note,
                    reviewed_by = excluded.reviewed_by, reviewer_name = excluded.reviewer_name,
                    reviewed_at = CURRENT_TIMESTAMP
                """,
                (alert_id, plant, status, note.strip()[:1000], user["user_id"], user["name"]),
            )
        return next(alert for alert in self.alerts(plant, limit=10_000) if alert["id"] == alert_id)

    # --- plant nodes ---

    def node(self, plant: str) -> dict:
        with sqlite3.connect(self.path) as database:
            row = database.execute(
                "SELECT display_name, location, capacity_mw, participates, notes, updated_at "
                "FROM plant_nodes WHERE client_id = ?",
                (plant,),
            ).fetchone()
        if row is None:
            number = plant.rsplit("_", 1)[-1]
            return {
                "client_id": plant, "display_name": f"Plant {number}", "location": "",
                "capacity_mw": None, "participates": True, "notes": "", "updated_at": None,
            }
        return {
            "client_id": plant, "display_name": row[0], "location": row[1], "capacity_mw": row[2],
            "participates": bool(row[3]), "notes": row[4], "updated_at": row[5],
        }

    def update_node(self, plant: str, changes: dict, user: dict) -> dict:
        if plant not in self.plant_ids():
            raise OperationsError(f"Unknown plant: {plant}")
        node = self.node(plant)
        for field in ("display_name", "location", "notes"):
            if changes.get(field) is not None:
                node[field] = str(changes[field]).strip()[:200]
        if not node["display_name"]:
            raise OperationsError("A plant needs a display name.")
        if "capacity_mw" in changes:
            capacity = changes["capacity_mw"]
            if capacity is not None and not 0 < float(capacity) < 10_000:
                raise OperationsError("Capacity must be between 0 and 10,000 MW.")
            node["capacity_mw"] = None if capacity is None else float(capacity)
        if changes.get("participates") is not None:
            node["participates"] = bool(changes["participates"])
            participating = [
                plant_id for plant_id in self.plant_ids()
                if (node["participates"] if plant_id == plant else self.node(plant_id)["participates"])
            ]
            if len(participating) < 2:
                raise OperationsError("Federated training needs at least two participating plants.")
        with sqlite3.connect(self.path) as database:
            database.execute(
                """
                INSERT INTO plant_nodes(client_id, display_name, location, capacity_mw, participates, notes, updated_by)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(client_id) DO UPDATE SET
                    display_name = excluded.display_name, location = excluded.location,
                    capacity_mw = excluded.capacity_mw, participates = excluded.participates,
                    notes = excluded.notes, updated_by = excluded.updated_by,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (plant, node["display_name"], node["location"], node["capacity_mw"],
                 int(node["participates"]), node["notes"], user["user_id"]),
            )
        return self.node(plant)

    def nodes(self, partition_summary: dict, client_rounds: list[dict]) -> list[dict]:
        """Every plant node with its data size and most recent federated round."""
        latest: dict[str, dict] = {}
        for row in client_rounds:
            if row["client_id"] and row["client_id"] not in latest:
                latest[row["client_id"]] = row
        nodes = []
        for plant in self.plant_ids():
            summary = partition_summary.get(plant, {})
            last = latest.get(plant)
            nodes.append({
                **self.node(plant),
                "windows": summary.get("windows"),
                "anomaly_rate": summary.get("anomaly_rate"),
                "last_round": last["round"] if last else None,
                "last_round_at": last["created_at"] if last else None,
                "last_loss": (last or {}).get("metrics", {}).get("loss"),
                "last_uplink_bytes": (last or {}).get("metrics", {}).get("uplink_bytes"),
            })
        return nodes

    # --- federation settings ---

    def federation_config(self) -> dict:
        with sqlite3.connect(self.path) as database:
            row = database.execute(
                "SELECT value_json, updated_at FROM settings WHERE key = 'federation'"
            ).fetchone()
        config = {**DEFAULT_FEDERATION_CONFIG, **(json.loads(row[0]) if row else {})}
        return {"config": config, "updated_at": row[1] if row else None, "defaults": DEFAULT_FEDERATION_CONFIG,
                "limits": FEDERATION_LIMITS, "choices": FEDERATION_CHOICES}

    def update_federation_config(self, changes: dict, user: dict) -> dict:
        config = self.federation_config()["config"]
        for key, value in changes.items():
            if key not in DEFAULT_FEDERATION_CONFIG:
                raise OperationsError(f"Unknown setting: {key}")
            if key in FEDERATION_CHOICES:
                if value not in FEDERATION_CHOICES[key]:
                    raise OperationsError(f"{key} must be one of {', '.join(FEDERATION_CHOICES[key])}.")
                config[key] = value
                continue
            low, high = FEDERATION_LIMITS[key]
            number = float(value)
            if not low <= number <= high:
                raise OperationsError(f"{key.replace('_', ' ')} must be between {low:g} and {high:g}.")
            config[key] = int(number) if isinstance(DEFAULT_FEDERATION_CONFIG[key], int) else number
        with sqlite3.connect(self.path) as database:
            database.execute(
                """
                INSERT INTO settings(key, value_json, updated_by) VALUES('federation', ?, ?)
                ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json,
                    updated_by = excluded.updated_by, updated_at = CURRENT_TIMESTAMP
                """,
                (json.dumps(config), user["user_id"]),
            )
        return self.federation_config()


SEVERITY_ORDER = ("medium", "high", "critical")


def _highest(severities: list[str]) -> str | None:
    ranked = [SEVERITY_ORDER.index(item) for item in severities if item in SEVERITY_ORDER]
    return SEVERITY_ORDER[max(ranked)] if ranked else None
