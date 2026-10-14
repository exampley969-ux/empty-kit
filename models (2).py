"""
Vision Radar - Vision Engine
Shared data models for detection & classification.

Spec reference: PDF Section 5 (Vision engine - Detection, Classification),
Section 14 (Confidence system).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class BoundingBox:
    """Pixel-space bounding box, top-left origin (matches OpenCV/image array convention)."""
    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    def iou(self, other: "BoundingBox") -> float:
        """Intersection-over-union with another box, used for de-duplication."""
        ix1 = max(self.x, other.x)
        iy1 = max(self.y, other.y)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        intersection = iw * ih
        if intersection == 0:
            return 0.0
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0

    def normalized(self, frame_width: int, frame_height: int) -> tuple:
        """Return (x, y, w, h) as fractions of frame size (0..1), per Section 8."""
        if frame_width <= 0 or frame_height <= 0:
            return (0.0, 0.0, 0.0, 0.0)
        return (
            self.x / frame_width,
            self.y / frame_height,
            self.width / frame_width,
            self.height / frame_height,
        )


@dataclass(frozen=True)
class Detection:
    """One classified detection produced by a single detector on a single frame."""
    entity_type: str        # e.g. "person" - open-ended, not an enum, so new
                             # types plug in without touching this model
    bbox: BoundingBox
    confidence: float        # 0.0-1.0 (Section 14 - never presented as certainty)
    detector_name: str        # which detector produced this (for evidence/debugging)


@dataclass
class FrameDetections:
    """All detections found in one frame, after pipeline aggregation (NMS etc.)."""
    session_id: str
    frame_id: int
    captured_at: float
    detections: List[Detection] = field(default_factory=list)

    def count_by_type(self) -> dict:
        counts: dict = {}
        for d in self.detections:
            counts[d.entity_type] = counts.get(d.entity_type, 0) + 1
        return counts
