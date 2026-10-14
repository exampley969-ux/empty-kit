"""
Vision Radar - Data Layer: Repository
The only place in the system that knows SQL. Everything above it deals in
the domain objects built in earlier phases (Zone, Event, Rule, Alert,
Evidence, EntitySnapshot, SessionInfo).

Spec reference: PDF Section 23 (Data architecture), Section 18 (Event
history filters: date/time, event type, entity, zone, confidence, status),
Section 32 (Data: sessions / events / configurations persist).

Every query value is passed as a bound parameter - never interpolated into
SQL text - so user-supplied filters cannot inject SQL.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from vision_radar.data_layer.database import Database
from vision_radar.event_engine.evidence import Evidence
from vision_radar.event_engine.models import Event, EventState, EventType
from vision_radar.ingestion.models import SessionInfo
from vision_radar.rules_engine.models import Alert, Rule
from vision_radar.world_model.models import EntitySnapshot
from vision_radar.world_model.zone import Zone

MAX_QUERY_LIMIT = 1000


def _json_default(obj: Any):
    # numpy scalars sneak in from the vision stack; convert them, but refuse
    # anything else rather than silently stringifying unexpected data.
    if hasattr(obj, "item") and callable(obj.item):
        return obj.item()
    raise TypeError(f"Cannot store value of type {type(obj).__name__} as JSON")


def _dumps(value: Any) -> str:
    return json.dumps(value, default=_json_default)


@dataclass
class CameraRecord:
    camera_id: str
    name: str
    source_uri: str
    frame_width: Optional[int] = None
    frame_height: Optional[int] = None
    orientation: str = "landscape"


@dataclass
class EventQuery:
    """Section 18's filter set. All fields optional; combined with AND."""
    start_time_from: Optional[float] = None
    start_time_to: Optional[float] = None
    event_type: Optional[EventType] = None
    entity_id: Optional[str] = None
    entity_type: Optional[str] = None
    zone_id: Optional[str] = None
    min_confidence: Optional[float] = None
    state: Optional[EventState] = None
    limit: int = 100
    offset: int = 0


class Repository:
    def __init__(self, db: Database):
        self.db = db

    # ---------------- cameras ----------------
    def add_camera(self, cam: CameraRecord) -> None:
        self.db.execute(
            "INSERT INTO cameras (camera_id, name, source_uri, frame_width, frame_height, orientation) "
            "VALUES (?,?,?,?,?,?)",
            (cam.camera_id, cam.name, cam.source_uri, cam.frame_width, cam.frame_height, cam.orientation))

    def get_camera(self, camera_id: str) -> Optional[CameraRecord]:
        r = self.db.query_one("SELECT * FROM cameras WHERE camera_id = ?", (camera_id,))
        return CameraRecord(**dict(r)) if r else None

    def list_cameras(self) -> List[CameraRecord]:
        return [CameraRecord(**dict(r)) for r in self.db.query_all("SELECT * FROM cameras ORDER BY name")]

    def delete_camera(self, camera_id: str) -> bool:
        return self.db.execute("DELETE FROM cameras WHERE camera_id = ?", (camera_id,)).rowcount > 0

    # ---------------- sessions ----------------
    def save_session(self, info: SessionInfo, camera_id: Optional[str] = None) -> None:
        self.db.execute(
            "INSERT INTO sessions (session_id, camera_id, camera_uri, state, started_at, ended_at, frame_count, error_message) "
            "VALUES (?,?,?,?,?,?,?,?) "
            "ON CONFLICT(session_id) DO UPDATE SET state=excluded.state, started_at=excluded.started_at, "
            "ended_at=excluded.ended_at, frame_count=excluded.frame_count, error_message=excluded.error_message",
            (info.session_id, camera_id, info.camera_uri, info.state.value, info.started_at,
             info.ended_at, info.frame_count, info.error_message))

    def get_session(self, session_id: str) -> Optional[dict]:
        r = self.db.query_one("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
        return dict(r) if r else None

    def list_sessions(self, limit: int = 100) -> List[dict]:
        rows = self.db.query_all("SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?", (self._limit(limit),))
        return [dict(r) for r in rows]

    # ---------------- entities ----------------
    def upsert_entity(self, e: EntitySnapshot, session_id: Optional[str]) -> None:
        self.db.execute(
            "INSERT INTO entities (entity_id, session_id, entity_type, first_seen, last_seen, last_state, last_x, last_y, confidence) "
            "VALUES (?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(entity_id) DO UPDATE SET last_seen=excluded.last_seen, last_state=excluded.last_state, "
            "last_x=excluded.last_x, last_y=excluded.last_y, confidence=excluded.confidence",
            (e.entity_id, session_id, e.entity_type, e.first_seen, e.last_seen,
             e.lifecycle_state.value, e.position[0], e.position[1], e.confidence))

    def list_entities(self, session_id: Optional[str] = None, limit: int = 100) -> List[dict]:
        if session_id:
            rows = self.db.query_all("SELECT * FROM entities WHERE session_id = ? ORDER BY last_seen DESC LIMIT ?",
                                     (session_id, self._limit(limit)))
        else:
            rows = self.db.query_all("SELECT * FROM entities ORDER BY last_seen DESC LIMIT ?", (self._limit(limit),))
        return [dict(r) for r in rows]

    # ---------------- zones ----------------
    def save_zone(self, zone: Zone, camera_id: Optional[str] = None) -> None:
        self.db.execute(
            "INSERT INTO zones (zone_id, camera_id, name, polygon_json) VALUES (?,?,?,?) "
            "ON CONFLICT(zone_id) DO UPDATE SET camera_id=excluded.camera_id, name=excluded.name, "
            "polygon_json=excluded.polygon_json",
            (zone.zone_id, camera_id, zone.name, _dumps(zone.polygon)))

    @staticmethod
    def _row_to_zone(r) -> Zone:
        return Zone(zone_id=r["zone_id"], name=r["name"],
                    polygon=[tuple(p) for p in json.loads(r["polygon_json"])])

    def get_zone(self, zone_id: str) -> Optional[Zone]:
        r = self.db.query_one("SELECT * FROM zones WHERE zone_id = ?", (zone_id,))
        return self._row_to_zone(r) if r else None

    def list_zones(self, camera_id: Optional[str] = None) -> List[Zone]:
        if camera_id:
            rows = self.db.query_all("SELECT * FROM zones WHERE camera_id = ? ORDER BY name", (camera_id,))
        else:
            rows = self.db.query_all("SELECT * FROM zones ORDER BY name")
        return [self._row_to_zone(r) for r in rows]

    def delete_zone(self, zone_id: str) -> bool:
        return self.db.execute("DELETE FROM zones WHERE zone_id = ?", (zone_id,)).rowcount > 0

    # ---------------- events ----------------
    def save_event(self, event: Event, session_id: Optional[str] = None) -> None:
        """Upsert - an event is saved again each time its lifecycle state
        changes (ACTIVE -> RESOLVED -> ARCHIVED). Event row + its entity
        links are written atomically."""
        self.db.transaction([
            ("INSERT INTO events (event_id, session_id, event_type, entity_type, zone_id, start_time, end_time, "
             "confidence, state, evidence_json, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?) "
             "ON CONFLICT(event_id) DO UPDATE SET end_time=excluded.end_time, confidence=excluded.confidence, "
             "state=excluded.state, evidence_json=excluded.evidence_json",
             (event.event_id, session_id, event.event_type.value, event.entity_type, event.zone_id,
              event.start_time, event.end_time, event.confidence, event.state.value,
              _dumps(event.evidence), event.created_at)),
            ("DELETE FROM event_entities WHERE event_id = ?", (event.event_id,)),
            *[("INSERT INTO event_entities (event_id, entity_id) VALUES (?,?)", (event.event_id, eid))
              for eid in event.entity_ids],
        ])

    def _entity_ids_for(self, event_ids: List[str]) -> Dict[str, List[str]]:
        result: Dict[str, List[str]] = {eid: [] for eid in event_ids}
        for i in range(0, len(event_ids), 500):
            chunk = event_ids[i:i + 500]
            marks = ",".join("?" * len(chunk))
            for r in self.db.query_all(
                    f"SELECT event_id, entity_id FROM event_entities WHERE event_id IN ({marks}) ORDER BY entity_id", chunk):
                result[r["event_id"]].append(r["entity_id"])
        return result

    @staticmethod
    def _row_to_event(r, entity_ids: List[str]) -> Event:
        return Event(
            event_id=r["event_id"], event_type=EventType(r["event_type"]), entity_ids=entity_ids,
            entity_type=r["entity_type"], zone_id=r["zone_id"], start_time=r["start_time"],
            end_time=r["end_time"], confidence=r["confidence"], state=EventState(r["state"]),
            evidence=json.loads(r["evidence_json"] or "{}"), created_at=r["created_at"])

    def get_event(self, event_id: str) -> Optional[Event]:
        r = self.db.query_one("SELECT * FROM events WHERE event_id = ?", (event_id,))
        if not r:
            return None
        return self._row_to_event(r, self._entity_ids_for([event_id])[event_id])

    def query_events(self, q: EventQuery) -> List[Event]:
        sql = "SELECT DISTINCT e.* FROM events e"
        where, params = [], []
        if q.entity_id is not None:
            sql += " JOIN event_entities ee ON ee.event_id = e.event_id"
            where.append("ee.entity_id = ?"); params.append(q.entity_id)
        if q.start_time_from is not None:
            where.append("e.start_time >= ?"); params.append(q.start_time_from)
        if q.start_time_to is not None:
            where.append("e.start_time <= ?"); params.append(q.start_time_to)
        if q.event_type is not None:
            where.append("e.event_type = ?"); params.append(q.event_type.value)
        if q.entity_type is not None:
            where.append("e.entity_type = ?"); params.append(q.entity_type)
        if q.zone_id is not None:
            where.append("e.zone_id = ?"); params.append(q.zone_id)
        if q.min_confidence is not None:
            where.append("e.confidence >= ?"); params.append(q.min_confidence)
        if q.state is not None:
            where.append("e.state = ?"); params.append(q.state.value)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY e.start_time DESC, e.event_id LIMIT ? OFFSET ?"
        params += [self._limit(q.limit), max(0, int(q.offset))]

        rows = self.db.query_all(sql, params)
        ids = [r["event_id"] for r in rows]
        entity_map = self._entity_ids_for(ids)
        return [self._row_to_event(r, entity_map[r["event_id"]]) for r in rows]

    # ---------------- rules ----------------
    def save_rule(self, rule: Rule) -> None:
        self.db.execute(
            "INSERT INTO rules (rule_id, name, definition_json) VALUES (?,?,?) "
            "ON CONFLICT(rule_id) DO UPDATE SET name=excluded.name, definition_json=excluded.definition_json",
            (rule.rule_id, rule.name, _dumps(rule.to_definition())))

    @staticmethod
    def _row_to_rule(r) -> Rule:
        return Rule.from_definition(r["name"], json.loads(r["definition_json"]), rule_id=r["rule_id"])

    def get_rule(self, rule_id: str) -> Optional[Rule]:
        r = self.db.query_one("SELECT * FROM rules WHERE rule_id = ?", (rule_id,))
        return self._row_to_rule(r) if r else None

    def list_rules(self) -> List[Rule]:
        return [self._row_to_rule(r) for r in self.db.query_all("SELECT * FROM rules ORDER BY name")]

    def delete_rule(self, rule_id: str) -> bool:
        return self.db.execute("DELETE FROM rules WHERE rule_id = ?", (rule_id,)).rowcount > 0

    # ---------------- alerts ----------------
    def save_alert(self, alert: Alert) -> None:
        self.db.execute(
            "INSERT INTO alerts (alert_id, rule_id, rule_name, event_id, message, confidence, created_at) "
            "VALUES (?,?,?,?,?,?,?) ON CONFLICT(alert_id) DO NOTHING",
            (alert.alert_id, alert.rule_id, alert.rule_name, alert.event_id, alert.message,
             alert.confidence, alert.created_at))

    def list_alerts(self, limit: int = 100, since: Optional[float] = None) -> List[Alert]:
        sql = ("SELECT a.*, e.event_type, e.zone_id FROM alerts a JOIN events e ON e.event_id = a.event_id")
        params: List[Any] = []
        if since is not None:
            sql += " WHERE a.created_at >= ?"; params.append(since)
        sql += " ORDER BY a.created_at DESC LIMIT ?"
        params.append(self._limit(limit))
        rows = self.db.query_all(sql, params)
        entity_map = self._entity_ids_for(list({r["event_id"] for r in rows}))
        return [Alert(
            alert_id=r["alert_id"], rule_id=r["rule_id"], rule_name=r["rule_name"], event_id=r["event_id"],
            event_type=EventType(r["event_type"]), entity_ids=entity_map[r["event_id"]], zone_id=r["zone_id"],
            confidence=r["confidence"], message=r["message"], created_at=r["created_at"]) for r in rows]

    # ---------------- evidence ----------------
    def save_evidence(self, ev: Evidence) -> None:
        self.db.execute(
            "INSERT INTO evidence (event_id, timestamp, confidence, entity_location_json, zone_geometry_json, frame_snapshot_path) "
            "VALUES (?,?,?,?,?,?) ON CONFLICT(event_id) DO UPDATE SET timestamp=excluded.timestamp, "
            "confidence=excluded.confidence, entity_location_json=excluded.entity_location_json, "
            "zone_geometry_json=excluded.zone_geometry_json, frame_snapshot_path=excluded.frame_snapshot_path",
            (ev.event_id, ev.timestamp, ev.confidence,
             _dumps(ev.entity_location) if ev.entity_location is not None else None,
             _dumps(ev.zone_geometry) if ev.zone_geometry is not None else None,
             ev.frame_snapshot_path))

    def get_evidence(self, event_id: str) -> Optional[Evidence]:
        r = self.db.query_one("SELECT * FROM evidence WHERE event_id = ?", (event_id,))
        if not r:
            return None
        return Evidence(
            event_id=r["event_id"], timestamp=r["timestamp"], confidence=r["confidence"],
            entity_location=json.loads(r["entity_location_json"]) if r["entity_location_json"] else None,
            zone_geometry=[tuple(p) for p in json.loads(r["zone_geometry_json"])] if r["zone_geometry_json"] else None,
            frame_snapshot_path=r["frame_snapshot_path"])

    # ---------------- system state ----------------
    def set_state(self, key: str, value: Any) -> None:
        self.db.execute(
            "INSERT INTO system_state (key, value_json, updated_at) VALUES (?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_at=excluded.updated_at",
            (key, _dumps(value), time.time()))

    def get_state(self, key: str, default: Any = None) -> Any:
        r = self.db.query_one("SELECT value_json FROM system_state WHERE key = ?", (key,))
        return json.loads(r["value_json"]) if r else default

    # ---------------- users ----------------
    def add_user(self, user_id: str, username: str, password_hash: str, role: str) -> None:
        self.db.execute("INSERT INTO users (user_id, username, password_hash, role) VALUES (?,?,?,?)",
                        (user_id, username, password_hash, role))

    def get_user_by_username(self, username: str) -> Optional[dict]:
        r = self.db.query_one("SELECT * FROM users WHERE username = ?", (username,))
        return dict(r) if r else None

    def get_user(self, user_id: str) -> Optional[dict]:
        r = self.db.query_one("SELECT * FROM users WHERE user_id = ?", (user_id,))
        return dict(r) if r else None

    def list_users(self) -> List[dict]:
        return [dict(r) for r in self.db.query_all("SELECT user_id, username, role FROM users ORDER BY username")]

    def user_count(self) -> int:
        return self.db.query_one("SELECT COUNT(*) AS n FROM users")["n"]

    # ---------------- retention support ----------------
    def evidence_with_files_older_than(self, cutoff: float) -> List[Tuple[str, str]]:
        rows = self.db.query_all(
            "SELECT event_id, frame_snapshot_path FROM evidence "
            "WHERE frame_snapshot_path IS NOT NULL AND timestamp < ?", (cutoff,))
        return [(r["event_id"], r["frame_snapshot_path"]) for r in rows]

    def clear_evidence_file_path(self, event_id: str) -> None:
        self.db.execute("UPDATE evidence SET frame_snapshot_path = NULL WHERE event_id = ?", (event_id,))

    def finished_event_ids_older_than(self, cutoff: float) -> List[str]:
        """RESOLVED/ARCHIVED events only - an ACTIVE event is still happening
        and must never be swept away by retention."""
        rows = self.db.query_all(
            "SELECT event_id FROM events WHERE state IN (?, ?) AND COALESCE(end_time, start_time) < ?",
            (EventState.RESOLVED.value, EventState.ARCHIVED.value, cutoff))
        return [r["event_id"] for r in rows]

    def evidence_file_paths_for(self, event_ids: List[str]) -> List[str]:
        paths: List[str] = []
        for i in range(0, len(event_ids), 500):
            chunk = event_ids[i:i + 500]
            marks = ",".join("?" * len(chunk))
            paths += [r["frame_snapshot_path"] for r in self.db.query_all(
                f"SELECT frame_snapshot_path FROM evidence WHERE frame_snapshot_path IS NOT NULL "
                f"AND event_id IN ({marks})", chunk)]
        return paths

    def delete_events(self, event_ids: List[str]) -> int:
        """Cascades to the events' alerts, evidence rows and entity links."""
        deleted = 0
        for i in range(0, len(event_ids), 500):
            chunk = event_ids[i:i + 500]
            marks = ",".join("?" * len(chunk))
            deleted += self.db.execute(f"DELETE FROM events WHERE event_id IN ({marks})", chunk).rowcount
        return deleted

    def count(self, table: str) -> int:
        allowed = {"cameras", "sessions", "entities", "zones", "events", "rules", "alerts", "evidence", "users"}
        if table not in allowed:
            raise ValueError(f"Unknown table '{table}'")
        return self.db.query_one(f"SELECT COUNT(*) AS n FROM {table}")["n"]

    @staticmethod
    def _limit(limit: int) -> int:
        return max(1, min(int(limit), MAX_QUERY_LIMIT))
