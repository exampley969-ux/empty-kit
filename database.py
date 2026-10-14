"""
Vision Radar - Data Layer: database connection + schema
SQLite via the standard library (no extra dependency; matches Section 24's
"for local development ... keeps camera processing and data close to the
source"). Every table Section 23 lists has a home here; plus `users` for
Section 28's authentication/authorization and `event_entities`, a join table
so "events for entity X" is an exact indexed match instead of a text search.

Note: in this implementation an Entity and its Track are the same object
(the World Model's entity_id IS the tracker's track_id), so one `entities`
table covers both of Section 23's "Entity" and "Track".

Failures never pass silently (Section 26): any sqlite error flips the
"database" health component to ERROR ("DATA STORAGE UNAVAILABLE") and is
re-raised as DataStorageUnavailable; the next successful call flips it back.
"""

from __future__ import annotations

import sqlite3
import threading
from typing import Any, Iterable, List, Optional

from vision_radar.monitoring.health import ComponentStatus, HealthRegistry

SCHEMA = """
CREATE TABLE IF NOT EXISTS cameras (
    camera_id    TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    source_uri   TEXT NOT NULL,
    frame_width  INTEGER,
    frame_height INTEGER,
    orientation  TEXT NOT NULL DEFAULT 'landscape'
);
CREATE TABLE IF NOT EXISTS sessions (
    session_id    TEXT PRIMARY KEY,
    camera_id     TEXT REFERENCES cameras(camera_id) ON DELETE SET NULL,
    camera_uri    TEXT,
    state         TEXT NOT NULL,
    started_at    REAL,
    ended_at      REAL,
    frame_count   INTEGER NOT NULL DEFAULT 0,
    error_message TEXT
);
CREATE TABLE IF NOT EXISTS entities (
    entity_id   TEXT PRIMARY KEY,
    session_id  TEXT REFERENCES sessions(session_id) ON DELETE CASCADE,
    entity_type TEXT NOT NULL,
    first_seen  REAL,
    last_seen   REAL,
    last_state  TEXT,
    last_x      REAL,
    last_y      REAL,
    confidence  REAL
);
CREATE TABLE IF NOT EXISTS zones (
    zone_id      TEXT PRIMARY KEY,
    camera_id    TEXT REFERENCES cameras(camera_id) ON DELETE CASCADE,
    name         TEXT NOT NULL,
    polygon_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    event_id      TEXT PRIMARY KEY,
    session_id    TEXT REFERENCES sessions(session_id) ON DELETE CASCADE,
    event_type    TEXT NOT NULL,
    entity_type   TEXT,
    zone_id       TEXT,
    start_time    REAL NOT NULL,
    end_time      REAL,
    confidence    REAL NOT NULL,
    state         TEXT NOT NULL,
    evidence_json TEXT NOT NULL DEFAULT '{}',
    created_at    REAL
);
CREATE INDEX IF NOT EXISTS idx_events_start  ON events(start_time);
CREATE INDEX IF NOT EXISTS idx_events_type   ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_zone   ON events(zone_id);
CREATE INDEX IF NOT EXISTS idx_events_state  ON events(state);
CREATE TABLE IF NOT EXISTS event_entities (
    event_id  TEXT NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
    entity_id TEXT NOT NULL,
    PRIMARY KEY (event_id, entity_id)
);
CREATE INDEX IF NOT EXISTS idx_event_entities_entity ON event_entities(entity_id);
CREATE TABLE IF NOT EXISTS rules (
    rule_id         TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    definition_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS alerts (
    alert_id   TEXT PRIMARY KEY,
    rule_id    TEXT NOT NULL,
    rule_name  TEXT,
    event_id   TEXT NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
    message    TEXT NOT NULL,
    confidence REAL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts(created_at);
CREATE TABLE IF NOT EXISTS evidence (
    event_id              TEXT PRIMARY KEY REFERENCES events(event_id) ON DELETE CASCADE,
    timestamp             REAL NOT NULL,
    confidence            REAL,
    entity_location_json  TEXT,
    zone_geometry_json    TEXT,
    frame_snapshot_path   TEXT
);
CREATE TABLE IF NOT EXISTS system_state (
    key        TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_at REAL
);
CREATE TABLE IF NOT EXISTS users (
    user_id       TEXT PRIMARY KEY,
    username      TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role          TEXT NOT NULL
);
"""


class DataIntegrityError(Exception):
    """A write violated a constraint (bad foreign key, duplicate key...).
    The storage itself is healthy, so this does NOT flip health to ERROR -
    reporting 'DATA STORAGE UNAVAILABLE' for a caller's bad data would be a
    false alarm."""


class DataStorageUnavailable(Exception):
    """Raised (and surfaced as 'DATA STORAGE UNAVAILABLE') on any storage failure."""


class Database:
    def __init__(self, path: str = ":memory:", health: Optional[HealthRegistry] = None):
        self.path = path
        self.health = health
        self._lock = threading.RLock()
        try:
            self._conn = sqlite3.connect(path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.executescript(SCHEMA)
            self._conn.commit()
        except sqlite3.Error as exc:
            self._report_failure(exc)
            raise DataStorageUnavailable(str(exc)) from exc
        self._report_ok()

    def _safe_rollback(self) -> None:
        """Roll back, tolerating a connection that is itself broken/closed:
        an error handler must never raise a different, raw error of its own
        (which would bypass DataStorageUnavailable and the health report)."""
        try:
            self._conn.rollback()
        except sqlite3.Error:
            pass

    def _report_ok(self) -> None:
        if self.health is not None:
            self.health.set("database", ComponentStatus.OK)

    def _report_failure(self, exc: Exception) -> None:
        if self.health is not None:
            self.health.set("database", ComponentStatus.ERROR, detail=str(exc))

    def execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            try:
                cur = self._conn.execute(sql, tuple(params))
                self._conn.commit()
            except sqlite3.IntegrityError as exc:
                self._safe_rollback()
                raise DataIntegrityError(str(exc)) from exc
            except sqlite3.Error as exc:
                self._safe_rollback()
                self._report_failure(exc)
                raise DataStorageUnavailable(str(exc)) from exc
        self._report_ok()
        return cur

    def query_all(self, sql: str, params: Iterable[Any] = ()) -> List[sqlite3.Row]:
        with self._lock:
            try:
                rows = self._conn.execute(sql, tuple(params)).fetchall()
            except sqlite3.Error as exc:
                self._report_failure(exc)
                raise DataStorageUnavailable(str(exc)) from exc
        self._report_ok()
        return rows

    def query_one(self, sql: str, params: Iterable[Any] = ()) -> Optional[sqlite3.Row]:
        rows = self.query_all(sql, params)
        return rows[0] if rows else None

    def transaction(self, statements: List[tuple]) -> None:
        """Run several (sql, params) statements atomically - all or nothing."""
        with self._lock:
            try:
                for sql, params in statements:
                    self._conn.execute(sql, tuple(params))
                self._conn.commit()
            except sqlite3.IntegrityError as exc:
                self._safe_rollback()
                raise DataIntegrityError(str(exc)) from exc
            except sqlite3.Error as exc:
                self._safe_rollback()
                self._report_failure(exc)
                raise DataStorageUnavailable(str(exc)) from exc
        self._report_ok()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
