"""
Vision Radar - Vision Engine: Tracking & Position
IoU tracker: assigns persistent identity to detections across frames.

Spec reference: PDF Section 5 (Tracking: maintain identity across frames -
"Frame 1 -> Person #17, Frame 2 -> Person #17..."), Section 7 (Entity
lifecycle state machine).

Design: matching is per entity_type (a "car" detection can never match a
"person" track) and uses the Hungarian algorithm (optimal assignment) on an
IoU cost matrix, rather than greedy nearest-match, so identity swaps between
two crossing entities of the same type are minimized.
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from vision_radar.vision.models import BoundingBox, Detection

from .models import Track, TrackedPoint, TrackerConfig, TrackState


class IoUTracker:
    def __init__(self, config: TrackerConfig = None):
        self.config = config or TrackerConfig()
        self._tracks: Dict[str, Track] = {}

    def active_tracks(self) -> List[Track]:
        """All tracks not yet EXITED (includes DETECTED/CONFIRMED/TRACKED/TEMPORARILY_LOST/REACQUIRED)."""
        return [t for t in self._tracks.values() if t.state != TrackState.EXITED]

    def get_track(self, track_id: str) -> Track:
        return self._tracks[track_id]

    def purge_exited(self) -> int:
        """Forget EXITED tracks. update() deliberately keeps them readable for
        callers that want a track's final state; a long-running service should
        call this once per frame after everything has consumed it, or every
        track ever created stays in memory forever."""
        ids = [tid for tid, t in self._tracks.items() if t.state == TrackState.EXITED]
        for tid in ids:
            del self._tracks[tid]
        return len(ids)

    def update(self, detections: List[Detection], timestamp: float = None) -> List[Track]:
        """
        Advance the tracker by one frame. Returns the current active tracks
        (post-update). `timestamp` defaults to wall-clock time if omitted -
        pass frame.captured_at explicitly for reproducible results.
        """
        if timestamp is None:
            timestamp = time.time()

        candidates = [t for t in self._tracks.values() if t.state != TrackState.EXITED]

        matched_track_ids = set()
        matched_detection_idxs = set()

        # Match independently within each entity_type so different types
        # never compete for the same assignment slot.
        by_type_dets: Dict[str, List[int]] = {}
        for i, d in enumerate(detections):
            by_type_dets.setdefault(d.entity_type, []).append(i)

        by_type_tracks: Dict[str, List[Track]] = {}
        for t in candidates:
            by_type_tracks.setdefault(t.entity_type, []).append(t)

        for entity_type, det_idxs in by_type_dets.items():
            type_tracks = by_type_tracks.get(entity_type, [])
            if not type_tracks:
                continue  # all these detections are unmatched -> new tracks, handled below

            cost_matrix = np.ones((len(type_tracks), len(det_idxs)), dtype=np.float64)
            for ti, track in enumerate(type_tracks):
                for di, det_idx in enumerate(det_idxs):
                    iou = track.bbox.iou(detections[det_idx].bbox)
                    cost_matrix[ti, di] = 1.0 - iou

            row_idx, col_idx = linear_sum_assignment(cost_matrix)

            for ti, di in zip(row_idx, col_idx):
                iou = 1.0 - cost_matrix[ti, di]
                if iou < self.config.iou_match_threshold:
                    continue  # assignment exists mathematically, but too weak to count as a real match
                track = type_tracks[ti]
                det_idx = det_idxs[di]
                self._apply_match(track, detections[det_idx], timestamp)
                matched_track_ids.add(track.track_id)
                matched_detection_idxs.add(det_idx)

        # Unmatched existing tracks -> a miss this frame
        for track in candidates:
            if track.track_id not in matched_track_ids:
                self._apply_miss(track)

        # Unmatched detections -> brand new tracks
        for i, det in enumerate(detections):
            if i not in matched_detection_idxs:
                new_track = Track.new(
                    entity_type=det.entity_type,
                    bbox=det.bbox,
                    confidence=det.confidence,
                    timestamp=timestamp,
                    history_length=self.config.history_length,
                )
                self._tracks[new_track.track_id] = new_track

        return self.active_tracks()

    # --- state machine ------------------------------------------------
    def _apply_match(self, track: Track, detection: Detection, timestamp: float) -> None:
        track.bbox = detection.bbox
        track.confidence = detection.confidence
        track.last_seen = timestamp
        track.hits += 1
        track.total_hits += 1
        track.missed_frames = 0

        cx, cy = track.center()
        track.history.append(TrackedPoint(cx, cy, timestamp))

        if track.state in (TrackState.TEMPORARILY_LOST,):
            track.state = TrackState.REACQUIRED
        elif track.state == TrackState.REACQUIRED:
            track.state = TrackState.TRACKED
        elif track.state == TrackState.DETECTED and track.hits >= self.config.confirm_hits:
            track.state = TrackState.CONFIRMED
        elif track.state == TrackState.CONFIRMED and track.hits >= self.config.tracked_hits:
            track.state = TrackState.TRACKED
        # else: stays in its current state (e.g. already TRACKED, or DETECTED
        # but not yet enough hits to advance)

    def _apply_miss(self, track: Track) -> None:
        track.hits = 0
        track.missed_frames += 1

        if track.missed_frames > self.config.max_missed_frames:
            track.state = TrackState.EXITED
        elif track.state in (TrackState.CONFIRMED, TrackState.TRACKED,
                              TrackState.REACQUIRED, TrackState.TEMPORARILY_LOST):
            track.state = TrackState.TEMPORARILY_LOST
        elif track.state == TrackState.DETECTED:
            # A DETECTED track (not yet confirmed) that immediately misses
            # is more likely a false-positive blip than a real entity that
            # needs a recovery window - drop it straight away.
            track.state = TrackState.EXITED
