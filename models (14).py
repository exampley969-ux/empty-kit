"""
Vision Radar - State & Change Engine
Shared data models.

Spec reference: PDF Section 9 (State engine - Appeared, Disappeared,
Movement, Zone entry, Zone exit, Dwell, State changed).

This layer answers "what changed?" by comparing each frame's World Model
(Phase 4) + Zone occupancy (Phase 5) against what was true last frame. It
does NOT do event lifecycle/deduplication/alerts - that's Phase 7's Event
Engine, which will consume the Change records this layer emits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple


class ChangeType(str, Enum):
    APPEARED = "APPEARED"
    DISAPPEARED = "DISAPPEARED"
    MOVED = "MOVED"
    ZONE_ENTERED = "ZONE_ENTERED"
    ZONE_EXITED = "ZONE_EXITED"
    DWELL = "DWELL"
    STATE_CHANGED = "STATE_CHANGED"


@dataclass
class StateChangeConfig:
    # A raw per-pixel position diff is not a meaningful "change" (Section 9
    # lists Movement as its own change type, distinct from every frame's
    # continuous position update) - this is the minimum cumulative
    # displacement since the entity's last reported MOVED change before a
    # new one fires. Interpretation choice, documented in PHASE6_README.md.
    movement_threshold_px: float = 30.0
    dwell_threshold_seconds: float = 5.0


@dataclass
class Change:
    """One detected state transition for one entity."""
    change_type: ChangeType
    entity_id: str
    entity_type: str
    timestamp: float
    zone_id: Optional[str] = None                    # for ZONE_ENTERED/EXITED/DWELL
    position: Optional[Tuple[float, float]] = None     # for APPEARED/MOVED
    extra: dict = field(default_factory=dict)          # e.g. {'from': .., 'to': ..} for STATE_CHANGED/MOVED/DWELL
