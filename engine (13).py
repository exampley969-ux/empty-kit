"""
Vision Radar - State & Change Engine
Detects Appeared/Disappeared/Moved/Zone-entered/Zone-exited/Dwell/
State-changed transitions by diffing this frame's entities (+ optional
zone occupancy) against what was tracked internally last frame.

Spec reference: PDF Section 9.

Caller contract (carried forward from Phase 5's own finding): pass
filtered, confirmed-or-better entities - and compute zone_occupancy from
that SAME filtered list - so single-frame detection noise can never
register a false APPEARED/ZONE_ENTERED/DWELL change. This engine trusts
its inputs; it does not re-filter by lifecycle state itself, to stay a
single-purpose diffing layer.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from vision_radar.world_model.models import EntitySnapshot
from vision_radar.world_model.zone import ZoneOccupancy

from .models import Change, ChangeType, StateChangeConfig


def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


class StateChangeEngine:
    def __init__(self, config: StateChangeConfig = None):
        self.config = config or StateChangeConfig()
        # entity_id -> {"entity_type", "last_moved_pos", "motion_state"}
        self._known_entities: Dict[str, dict] = {}
        # entity_id -> set of zone_ids currently occupied
        self._entity_zones: Dict[str, Set[str]] = {}
        # (entity_id, zone_id) -> timestamp first entered
        self._dwell_start: Dict[Tuple[str, str], float] = {}
        # (entity_id, zone_id) already reported a DWELL change this visit
        self._dwell_fired: Set[Tuple[str, str]] = set()

    def update(self, entities: List[EntitySnapshot],
               zone_occupancy: Optional[Dict[str, ZoneOccupancy]] = None,
               timestamp: float = 0.0) -> List[Change]:
        changes: List[Change] = []

        changes.extend(self._detect_appeared(entities, timestamp))
        changes.extend(self._detect_disappeared(entities, timestamp))
        changes.extend(self._detect_movement_and_state(entities, timestamp))

        if zone_occupancy is not None:
            changes.extend(self._detect_zone_changes(zone_occupancy, timestamp))
            changes.extend(self._detect_dwell(zone_occupancy, timestamp))

        return changes

    # --- Appeared / Disappeared -----------------------------------------
    def _detect_appeared(self, entities: List[EntitySnapshot], timestamp: float) -> List[Change]:
        changes = []
        for e in entities:
            if e.entity_id not in self._known_entities:
                self._known_entities[e.entity_id] = {
                    "entity_type": e.entity_type,
                    "last_moved_pos": e.position,
                    "motion_state": e.motion_state,
                }
                changes.append(Change(
                    change_type=ChangeType.APPEARED,
                    entity_id=e.entity_id, entity_type=e.entity_type,
                    timestamp=timestamp, position=e.position,
                ))
        return changes

    def _detect_disappeared(self, entities: List[EntitySnapshot], timestamp: float) -> List[Change]:
        changes = []
        current_ids = {e.entity_id for e in entities}
        for eid in list(self._known_entities.keys()):
            if eid not in current_ids:
                info = self._known_entities.pop(eid)
                # An entity that vanishes while inside a zone has, by
                # definition, left that zone. Without emitting ZONE_EXITED
                # here, the downstream ZONE_ENTRY event would stay ACTIVE
                # forever (found in Phase 9's long-running-service review).
                for zone_id in sorted(self._entity_zones.get(eid, set())):
                    changes.append(Change(
                        change_type=ChangeType.ZONE_EXITED,
                        entity_id=eid, entity_type=info["entity_type"], timestamp=timestamp,
                        zone_id=zone_id, extra={"reason": "entity_disappeared"},
                    ))
                changes.append(Change(
                    change_type=ChangeType.DISAPPEARED,
                    entity_id=eid, entity_type=info["entity_type"], timestamp=timestamp,
                ))
                self._cleanup_zone_state(eid)
        return changes

    def _cleanup_zone_state(self, entity_id: str) -> None:
        self._entity_zones.pop(entity_id, None)
        for key in [k for k in self._dwell_start if k[0] == entity_id]:
            del self._dwell_start[key]
            self._dwell_fired.discard(key)

    # --- Movement / motion-state change ----------------------------------
    def _detect_movement_and_state(self, entities: List[EntitySnapshot],
                                    timestamp: float) -> List[Change]:
        changes = []
        for e in entities:
            info = self._known_entities.get(e.entity_id)
            if info is None:
                continue  # defensive; _detect_appeared always populates this first

            dist = _distance(info["last_moved_pos"], e.position)
            if dist >= self.config.movement_threshold_px:
                changes.append(Change(
                    change_type=ChangeType.MOVED,
                    entity_id=e.entity_id, entity_type=e.entity_type, timestamp=timestamp,
                    position=e.position,
                    extra={"from": info["last_moved_pos"], "to": e.position, "distance": dist},
                ))
                info["last_moved_pos"] = e.position

            if info["motion_state"] != e.motion_state:
                changes.append(Change(
                    change_type=ChangeType.STATE_CHANGED,
                    entity_id=e.entity_id, entity_type=e.entity_type, timestamp=timestamp,
                    extra={"from": info["motion_state"].value, "to": e.motion_state.value},
                ))
                info["motion_state"] = e.motion_state

        return changes

    # --- Zone entry / exit / dwell ---------------------------------------
    def _current_entity_zone_map(self, zone_occupancy: Dict[str, ZoneOccupancy]) -> Dict[str, Set[str]]:
        mapping: Dict[str, Set[str]] = {}
        for zone_id, occ in zone_occupancy.items():
            for entity_id in occ.entity_ids:
                mapping.setdefault(entity_id, set()).add(zone_id)
        return mapping

    def _detect_zone_changes(self, zone_occupancy: Dict[str, ZoneOccupancy],
                              timestamp: float) -> List[Change]:
        changes = []
        current_map = self._current_entity_zone_map(zone_occupancy)
        all_entity_ids = set(self._entity_zones.keys()) | set(current_map.keys())

        for entity_id in all_entity_ids:
            prev_zones = self._entity_zones.get(entity_id, set())
            curr_zones = current_map.get(entity_id, set())
            entity_type = self._known_entities.get(entity_id, {}).get("entity_type", "unknown")

            for zone_id in curr_zones - prev_zones:
                changes.append(Change(
                    change_type=ChangeType.ZONE_ENTERED,
                    entity_id=entity_id, entity_type=entity_type, timestamp=timestamp,
                    zone_id=zone_id,
                ))
                self._dwell_start[(entity_id, zone_id)] = timestamp
                self._dwell_fired.discard((entity_id, zone_id))

            for zone_id in prev_zones - curr_zones:
                changes.append(Change(
                    change_type=ChangeType.ZONE_EXITED,
                    entity_id=entity_id, entity_type=entity_type, timestamp=timestamp,
                    zone_id=zone_id,
                ))
                self._dwell_start.pop((entity_id, zone_id), None)
                self._dwell_fired.discard((entity_id, zone_id))

            self._entity_zones[entity_id] = curr_zones

        return changes

    def _detect_dwell(self, zone_occupancy: Dict[str, ZoneOccupancy],
                       timestamp: float) -> List[Change]:
        changes = []
        current_map = self._current_entity_zone_map(zone_occupancy)

        for (entity_id, zone_id), start_time in list(self._dwell_start.items()):
            if (entity_id, zone_id) in self._dwell_fired:
                continue
            still_inside = zone_id in current_map.get(entity_id, set())
            if not still_inside:
                continue
            elapsed = timestamp - start_time
            if elapsed >= self.config.dwell_threshold_seconds:
                entity_type = self._known_entities.get(entity_id, {}).get("entity_type", "unknown")
                changes.append(Change(
                    change_type=ChangeType.DWELL,
                    entity_id=entity_id, entity_type=entity_type, timestamp=timestamp,
                    zone_id=zone_id, extra={"dwell_seconds": elapsed},
                ))
                self._dwell_fired.add((entity_id, zone_id))

        return changes
