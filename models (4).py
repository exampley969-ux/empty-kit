"""
Vision Radar - World Model
Shared data models.

Spec reference: PDF Section 6 (World Model - the persistent, continuously
updated environment state, e.g. "Entity #17: Type Person, Position (x,y),
Velocity..., Direction NE, State Moving").

Zone assignment ("Zone: A" in the Section 6 example) is intentionally NOT
here - that's Phase 5. This phase only covers entity identity, position,
motion and current state; nothing about the environment's fixed geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple

from vision_radar.tracking.models import TrackState
from vision_radar.vision.models import BoundingBox


class MotionState(str, Enum):
    """Coarse motion state shown in the Section 6 World Model example
    ("State: Moving" / "State: Stationary") - distinct from the entity
    lifecycle state (DETECTED/TRACKED/etc, Section 7), which describes
    tracking confidence, not physical motion."""
    MOVING = "MOVING"
    STATIONARY = "STATIONARY"


@dataclass
class EntitySnapshot:
    """
    A live, point-in-time view of one tracked entity, as the rest of the
    system (dashboard, radar view, later the event engine) should consume
    it - everything computed and ready to read, no further derivation
    needed by callers.
    """
    entity_id: str                # == the underlying Track's track_id
    entity_type: str
    lifecycle_state: TrackState
    motion_state: MotionState
    bbox: BoundingBox
    position: Tuple[float, float]         # (x, y) center, image-space pixels
    velocity: Tuple[float, float]         # (vx, vy) px/sec, image-space
    speed: float                          # px/sec
    direction: str                        # 8-point compass label, or STATIONARY
    confidence: float
    first_seen: float
    last_seen: float
    dwell_seconds: float
    trajectory: List[Tuple[float, float]] = field(default_factory=list)

    def normalized_position(self, frame_width: int, frame_height: int) -> Tuple[float, float]:
        """Position as a fraction of frame size (0..1) - Section 8 spatial model."""
        if frame_width <= 0 or frame_height <= 0:
            return (0.0, 0.0)
        x, y = self.position
        return (x / frame_width, y / frame_height)
