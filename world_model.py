"""
Vision Radar - World Model
Maintains the live, queryable "what exists right now" view of the
environment, built directly from the tracker's output each frame.

Spec reference: PDF Section 6 (World Model), Section 8 (Spatial model).

This is intentionally a thin transformation + query layer, not a second
source of truth: it never mutates or re-decides anything the tracker (Phase
3) already decided about identity or lifecycle. It only derives a couple of
presentation-ready fields (motion_state, normalized position on request) and
keeps the *latest* snapshot per entity queryable by ID - satisfying the
Phase-4 acceptance goal that "current state is continuously updated" without
introducing any independent state of its own.
"""

from __future__ import annotations

from typing import Dict, List

from vision_radar.tracking.models import Track, TrackState
from vision_radar.tracking.position import compute_velocity, compute_speed, compute_direction_label

from .models import EntitySnapshot, MotionState


class WorldModel:
    def __init__(self, stationary_speed_threshold: float = 2.0, velocity_window: int = 5):
        self.stationary_speed_threshold = stationary_speed_threshold
        self.velocity_window = velocity_window
        self._entities: Dict[str, EntitySnapshot] = {}

    def update(self, tracks: List[Track]) -> List[EntitySnapshot]:
        """
        Rebuild the live snapshot set from the tracker's current output.
        Tracks in EXITED state are dropped - they are no longer part of
        "what exists right now" (Section 6). Called once per frame with
        IoUTracker.update()'s return value.
        """
        current_ids = set()
        for track in tracks:
            if track.state == TrackState.EXITED:
                continue
            snapshot = self._build_snapshot(track)
            self._entities[snapshot.entity_id] = snapshot
            current_ids.add(snapshot.entity_id)

        # Drop anything no longer reported as active (e.g. it exited since
        # the last update call) so stale entities never linger.
        for stale_id in list(self._entities.keys()):
            if stale_id not in current_ids:
                del self._entities[stale_id]

        return self.get_all_entities()

    def _build_snapshot(self, track: Track) -> EntitySnapshot:
        history = list(track.history)
        vx, vy = compute_velocity(history, window=self.velocity_window)
        speed = compute_speed(vx, vy)
        direction = compute_direction_label(vx, vy, self.stationary_speed_threshold)
        motion_state = (
            MotionState.STATIONARY if speed < self.stationary_speed_threshold
            else MotionState.MOVING
        )

        return EntitySnapshot(
            entity_id=track.track_id,
            entity_type=track.entity_type,
            lifecycle_state=track.state,
            motion_state=motion_state,
            bbox=track.bbox,
            position=track.center(),
            velocity=(vx, vy),
            speed=speed,
            direction=direction,
            confidence=track.confidence,
            first_seen=track.first_seen,
            last_seen=track.last_seen,
            dwell_seconds=track.dwell_seconds(),
            trajectory=[(p.x, p.y) for p in history],
        )

    # --- queries --------------------------------------------------------
    def get_entity(self, entity_id: str) -> EntitySnapshot:
        return self._entities[entity_id]

    def get_all_entities(self) -> List[EntitySnapshot]:
        return list(self._entities.values())

    def count_by_type(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for e in self._entities.values():
            counts[e.entity_type] = counts.get(e.entity_type, 0) + 1
        return counts

    def count_by_motion_state(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for e in self._entities.values():
            counts[e.motion_state.value] = counts.get(e.motion_state.value, 0) + 1
        return counts
