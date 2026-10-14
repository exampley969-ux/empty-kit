"""
Vision Radar - Vision Engine: Tracking & Position
Shared data models.

Spec reference: PDF Section 6 (World Model - persistent entity context),
Section 7 (Entity lifecycle), Section 8 (Spatial model - coordinates,
trajectory, movement direction).
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, List, Optional

from vision_radar.vision.models import BoundingBox


class TrackState(str, Enum):
    """
    Entity lifecycle exactly as specified (Section 7):
    NOT PRESENT -> DETECTED -> CONFIRMED -> TRACKED -> TEMPORARILY LOST ->
    REACQUIRED -> EXITED.
    "NOT PRESENT" has no explicit state - it's simply the absence of a Track.
    """
    DETECTED = "DETECTED"
    CONFIRMED = "CONFIRMED"
    TRACKED = "TRACKED"
    TEMPORARILY_LOST = "TEMPORARILY_LOST"
    REACQUIRED = "REACQUIRED"
    EXITED = "EXITED"


@dataclass
class TrackedPoint:
    """One historical position sample, used to derive velocity/direction/trajectory."""
    x: float
    y: float
    timestamp: float


@dataclass
class TrackerConfig:
    """Tuning knobs for the IoU tracker's matching and lifecycle thresholds."""
    iou_match_threshold: float = 0.3     # minimum IoU to consider a detection<->track match
    confirm_hits: int = 2                # consecutive matched frames: DETECTED -> CONFIRMED
    tracked_hits: int = 4                # consecutive matched frames: CONFIRMED -> TRACKED
    max_missed_frames: int = 10          # consecutive misses before a track EXITS
    history_length: int = 30             # bounded trajectory history per track
    stationary_speed_threshold: float = 2.0  # px/sec below which we call it "stationary"


@dataclass
class Track:
    """Persistent identity for one entity across frames (Section 6/7)."""
    track_id: str
    entity_type: str
    bbox: BoundingBox
    state: TrackState
    confidence: float
    first_seen: float
    last_seen: float
    hits: int = 0                 # consecutive matched frames (resets on a miss)
    total_hits: int = 0           # lifetime matched-frame count
    missed_frames: int = 0        # consecutive missed frames (resets on a match)
    history: Deque[TrackedPoint] = field(default_factory=lambda: deque(maxlen=30))

    @classmethod
    def new(cls, entity_type: str, bbox: BoundingBox, confidence: float,
            timestamp: float, history_length: int) -> "Track":
        track = cls(
            track_id=str(uuid.uuid4()),
            entity_type=entity_type,
            bbox=bbox,
            state=TrackState.DETECTED,
            confidence=confidence,
            first_seen=timestamp,
            last_seen=timestamp,
            hits=1,
            total_hits=1,
            missed_frames=0,
            history=deque(maxlen=history_length),
        )
        cx, cy = bbox.x + bbox.width / 2, bbox.y + bbox.height / 2
        track.history.append(TrackedPoint(cx, cy, timestamp))
        return track

    def center(self) -> tuple:
        return (self.bbox.x + self.bbox.width / 2, self.bbox.y + self.bbox.height / 2)

    def dwell_seconds(self) -> float:
        return self.last_seen - self.first_seen
