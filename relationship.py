"""
Vision Radar - World Model: Relationship Engine
Detects relationships between pairs of entities: proximity ("near") and
co-movement ("moving together").

Spec reference: PDF Section 10 (Relationship engine - "Person near
vehicle... Two entities move together... Object remains near another
object").

Scope note: this engine detects and gives a *stable identity* to a
relationship for as long as it continuously holds, so the same
"A near B" doesn't get a new ID every single frame. It deliberately does
NOT do full event lifecycle/deduplication-across-gaps (Potential -> Active
-> Resolved, hysteresis for a one-frame flicker) - that's Phase 7's Event
Engine, which will consume these relationship observations as one of its
inputs. Here, if a relationship's condition stops holding even for one
frame, it ends; if the same pair satisfies the condition again later, it
gets a new relationship_id. This is an intentional scope boundary, not an
oversight - documented in PHASE5_README.md.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple

from vision_radar.world_model.models import EntitySnapshot, MotionState


class RelationshipType(str, Enum):
    NEAR = "NEAR"
    MOVING_TOGETHER = "MOVING_TOGETHER"


@dataclass
class RelationshipConfig:
    proximity_threshold_px: float = 150.0     # max center-to-center distance for NEAR
    comovement_speed_ratio_tolerance: float = 0.4   # |speed_a - speed_b| / max(speed_a,speed_b) must be <= this
    min_comovement_speed: float = 5.0          # both entities must be moving at least this fast (px/s)


@dataclass
class Relationship:
    relationship_id: str
    relationship_type: RelationshipType
    entity_a_id: str
    entity_b_id: str
    entity_a_type: str
    entity_b_type: str
    first_observed: float
    last_observed: float
    distance: float


def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _pair_key(id_a: str, id_b: str, rel_type: RelationshipType) -> frozenset:
    return frozenset({id_a, id_b, rel_type.value})


class RelationshipEngine:
    def __init__(self, config: RelationshipConfig = None):
        self.config = config or RelationshipConfig()
        self._active: Dict[frozenset, Relationship] = {}

    def update(self, entities: List[EntitySnapshot], timestamp: float) -> List[Relationship]:
        """Recompute all pairwise relationships for this frame. Returns the
        currently-active relationships (a fresh list, not history)."""
        current_keys = set()

        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                a, b = entities[i], entities[j]
                dist = _distance(a.position, b.position)

                if dist <= self.config.proximity_threshold_px:
                    self._record(RelationshipType.NEAR, a, b, dist, timestamp, current_keys)

                if self._is_comovement(a, b):
                    self._record(RelationshipType.MOVING_TOGETHER, a, b, dist, timestamp, current_keys)

        # Drop anything that didn't hold this frame - see module docstring
        # for why this doesn't attempt cross-gap persistence (Phase 7's job).
        for key in list(self._active.keys()):
            if key not in current_keys:
                del self._active[key]

        return list(self._active.values())

    def _is_comovement(self, a: EntitySnapshot, b: EntitySnapshot) -> bool:
        if a.motion_state != MotionState.MOVING or b.motion_state != MotionState.MOVING:
            return False
        if a.speed < self.config.min_comovement_speed or b.speed < self.config.min_comovement_speed:
            return False
        if a.direction != b.direction:
            return False
        faster, slower = max(a.speed, b.speed), min(a.speed, b.speed)
        ratio_diff = (faster - slower) / faster if faster > 0 else 0.0
        return ratio_diff <= self.config.comovement_speed_ratio_tolerance

    def _record(self, rel_type: RelationshipType, a: EntitySnapshot, b: EntitySnapshot,
                distance: float, timestamp: float, current_keys: set) -> None:
        key = _pair_key(a.entity_id, b.entity_id, rel_type)
        current_keys.add(key)

        if key in self._active:
            existing = self._active[key]
            existing.last_observed = timestamp
            existing.distance = distance
        else:
            self._active[key] = Relationship(
                relationship_id=str(uuid.uuid4()),
                relationship_type=rel_type,
                entity_a_id=a.entity_id,
                entity_b_id=b.entity_id,
                entity_a_type=a.entity_type,
                entity_b_type=b.entity_type,
                first_observed=timestamp,
                last_observed=timestamp,
                distance=distance,
            )
