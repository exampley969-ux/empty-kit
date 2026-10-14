"""
Vision Radar - World Model: Camera Calibration
Per-session calibration settings.

Spec reference: PDF Section 15 (Camera calibration): "Initial setup should
allow: camera resolution, scene boundaries, orientation, zone creation,
reference points, perspective configuration... measurements should
primarily be treated as image-space/spatial coordinates, unless a reliable
depth/calibration method is explicitly implemented. We should not falsely
claim real-world metres from a normal single camera."

This module deliberately does NOT provide any pixel-to-real-world-distance
conversion. That absence is the honesty guarantee from Section 15, enforced
in code (not just in a docstring) via `estimate_real_world_distance`, which
exists specifically so a future caller who reaches for it gets a clear
explanation instead of silently-wrong numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .zone import point_in_polygon


@dataclass
class CameraCalibration:
    frame_width: int
    frame_height: int
    orientation: str = "landscape"          # "landscape" | "portrait" - informational
    scene_boundary: Optional[List[Tuple[float, float]]] = None   # optional monitored-area polygon
    reference_points: List[Tuple[float, float]] = field(default_factory=list)  # informational only
    has_real_world_calibration: bool = False   # always False until a real method is implemented

    def to_normalized(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """Pixel coordinates -> (0..1) fraction of frame size."""
        if self.frame_width <= 0 or self.frame_height <= 0:
            return (0.0, 0.0)
        x, y = point
        return (x / self.frame_width, y / self.frame_height)

    def to_pixel(self, normalized_point: Tuple[float, float]) -> Tuple[float, float]:
        """(0..1) fraction of frame size -> pixel coordinates."""
        nx, ny = normalized_point
        return (nx * self.frame_width, ny * self.frame_height)

    def within_scene_boundary(self, point: Tuple[float, float]) -> bool:
        """True if no scene_boundary is defined (nothing excluded), or the
        point falls inside the defined boundary polygon."""
        if self.scene_boundary is None:
            return True
        return point_in_polygon(point, self.scene_boundary)

    def estimate_real_world_distance(self, point_a: Tuple[float, float],
                                      point_b: Tuple[float, float]) -> float:
        """
        Deliberately not implemented. A single uncalibrated camera cannot
        honestly convert pixel distance to real-world distance (perspective,
        unknown depth, lens distortion). Raising here - rather than quietly
        returning a plausible-looking but fabricated number - is the
        Section 15 requirement enforced in code.
        """
        raise NotImplementedError(
            "No real-world distance calibration is implemented (Section 15). "
            "Use pixel-space distance/normalized coordinates instead, or "
            "implement and explicitly enable a real calibration method "
            "before adding real-world units here."
        )
