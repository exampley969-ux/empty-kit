"""Plain-dict (JSON-ready) views of domain objects. Never includes password
hashes, and never exposes server file-system paths."""

from __future__ import annotations

from typing import Optional

from vision_radar.data_layer.repository import CameraRecord
from vision_radar.event_engine.evidence import Evidence
from vision_radar.event_engine.models import Event
from vision_radar.rules_engine.models import Alert, Rule
from vision_radar.world_model.zone import Zone


def event_to_dict(e: Event, evidence: Optional[Evidence] = None) -> dict:
    d = {
        "event_id": e.event_id,
        "event_type": e.event_type.value,
        "state": e.state.value,
        "entity_ids": list(e.entity_ids),
        "entity_type": e.entity_type,
        "zone_id": e.zone_id,
        "start_time": e.start_time,
        "end_time": e.end_time,
        "duration_seconds": e.duration_seconds(),
        "confidence": e.confidence,
        "evidence": e.evidence,
    }
    if evidence is not None:
        d["evidence_record"] = {
            "timestamp": evidence.timestamp,
            "confidence": evidence.confidence,
            "entity_location": evidence.entity_location,
            "zone_geometry": [list(p) for p in evidence.zone_geometry] if evidence.zone_geometry else None,
            "has_snapshot": bool(evidence.frame_snapshot_path),
        }
    return d


def alert_to_dict(a: Alert) -> dict:
    return {
        "alert_id": a.alert_id, "rule_id": a.rule_id, "rule_name": a.rule_name,
        "event_id": a.event_id, "event_type": a.event_type.value,
        "entity_ids": list(a.entity_ids), "zone_id": a.zone_id,
        "confidence": a.confidence, "message": a.message, "created_at": a.created_at,
    }


def zone_to_dict(z: Zone) -> dict:
    return {"zone_id": z.zone_id, "name": z.name, "polygon": [list(p) for p in z.polygon]}


def rule_to_dict(r: Rule) -> dict:
    return {"rule_id": r.rule_id, "name": r.name, "definition": r.to_definition()}


def camera_to_dict(c: CameraRecord) -> dict:
    return {"camera_id": c.camera_id, "name": c.name, "source_uri": c.source_uri,
            "frame_width": c.frame_width, "frame_height": c.frame_height, "orientation": c.orientation}
