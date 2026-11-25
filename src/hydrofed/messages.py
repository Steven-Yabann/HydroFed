"""Conversations between plant operators and administrators.

Each plant has one conversation. Its operator and every administrator can
read and write in it. A message may point at one of the plant's alerts; the
alert's severity and start time are copied in when the message is sent, so
the reference still reads correctly after the replay loop moves on.
Each account remembers the last message it has read in each conversation,
which gives the unread counts in the sidebar.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from hydrofed.auth import ROLE_LABELS

MAX_MESSAGE_LENGTH = 2000
THREAD_LIMIT = 200

MESSAGES_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
    plant_id TEXT NOT NULL,
    author_id INTEGER,
    author_name TEXT NOT NULL,
    author_role TEXT NOT NULL,
    body TEXT NOT NULL,
    alert_json TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS messages_by_plant ON messages(plant_id, message_id);
CREATE TABLE IF NOT EXISTS message_reads (
    user_id INTEGER NOT NULL,
    plant_id TEXT NOT NULL,
    last_read_id INTEGER NOT NULL,
    PRIMARY KEY (user_id, plant_id)
);
"""

COLUMNS = "message_id, plant_id, author_id, author_name, author_role, body, alert_json, created_at"


class MessageError(Exception):
    """A message that cannot be sent."""


class MessageStore:
    def __init__(self, database_path: str | Path):
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as database:
            database.executescript(MESSAGES_SCHEMA)

    @staticmethod
    def _message(row: tuple, user: dict) -> dict:
        message_id, plant, author_id, author_name, author_role, body, alert_json, created_at = row
        return {
            "id": message_id,
            "plant": plant,
            "author": {
                "id": author_id,
                "name": author_name,
                "role": author_role,
                "role_label": ROLE_LABELS.get(author_role, author_role),
            },
            "mine": author_id == user["user_id"],
            "body": body,
            "alert": json.loads(alert_json) if alert_json else None,
            "created_at": created_at,
        }

    def post(self, plant: str, user: dict, body: str, alert: dict | None = None) -> dict:
        text = body.strip()
        if not text:
            raise MessageError("Write a message first.")
        if len(text) > MAX_MESSAGE_LENGTH:
            raise MessageError(f"Keep messages under {MAX_MESSAGE_LENGTH} characters.")
        with sqlite3.connect(self.path) as database:
            cursor = database.execute(
                "INSERT INTO messages(plant_id, author_id, author_name, author_role, body, alert_json) "
                "VALUES(?, ?, ?, ?, ?, ?)",
                (plant, user["user_id"], user["name"], user["role"], text, json.dumps(alert) if alert else None),
            )
            message_id = cursor.lastrowid
            self._mark_read(database, user, plant, message_id)
            row = database.execute(f"SELECT {COLUMNS} FROM messages WHERE message_id = ?", (message_id,)).fetchone()
        return self._message(row, user)

    def thread(self, plant: str, user: dict, limit: int = THREAD_LIMIT) -> list[dict]:
        """Return the latest messages, oldest first, and mark them read for this account."""
        with sqlite3.connect(self.path) as database:
            rows = database.execute(
                f"SELECT {COLUMNS} FROM messages WHERE plant_id = ? ORDER BY message_id DESC LIMIT ?",
                (plant, limit),
            ).fetchall()
            if rows:
                self._mark_read(database, user, plant, rows[0][0])
        return [self._message(row, user) for row in reversed(rows)]

    def summaries(self, plants: list[str], user: dict) -> list[dict]:
        """The latest message and unread count for each plant's conversation."""
        if not plants:
            return []
        marks = ",".join("?" for _ in plants)
        with sqlite3.connect(self.path) as database:
            latest = {
                row[1]: row
                for row in database.execute(
                    f"SELECT {COLUMNS} FROM messages WHERE message_id IN ("
                    f"SELECT MAX(message_id) FROM messages WHERE plant_id IN ({marks}) GROUP BY plant_id)",
                    plants,
                )
            }
            unread = dict(database.execute(
                f"""
                SELECT m.plant_id, COUNT(*) FROM messages m
                LEFT JOIN message_reads r ON r.plant_id = m.plant_id AND r.user_id = ?
                WHERE m.plant_id IN ({marks})
                  AND m.message_id > COALESCE(r.last_read_id, 0)
                  AND (m.author_id IS NULL OR m.author_id != ?)
                GROUP BY m.plant_id
                """,
                (user["user_id"], *plants, user["user_id"]),
            ).fetchall())
        return [
            {
                "plant": plant,
                "last_message": self._message(latest[plant], user) if plant in latest else None,
                "unread": unread.get(plant, 0),
            }
            for plant in plants
        ]

    def unread_for(self, plant: str, user: dict) -> int:
        """How many messages from other people this account has not read in a conversation."""
        with sqlite3.connect(self.path) as database:
            return database.execute(
                """
                SELECT COUNT(*) FROM messages m
                LEFT JOIN message_reads r ON r.plant_id = m.plant_id AND r.user_id = ?
                WHERE m.plant_id = ? AND m.message_id > COALESCE(r.last_read_id, 0)
                  AND (m.author_id IS NULL OR m.author_id != ?)
                """,
                (user["user_id"], plant, user["user_id"]),
            ).fetchone()[0]

    @staticmethod
    def _mark_read(database: sqlite3.Connection, user: dict, plant: str, message_id: int) -> None:
        database.execute(
            """
            INSERT INTO message_reads(user_id, plant_id, last_read_id) VALUES(?, ?, ?)
            ON CONFLICT(user_id, plant_id) DO UPDATE SET
                last_read_id = MAX(last_read_id, excluded.last_read_id)
            """,
            (user["user_id"], plant, message_id),
        )
