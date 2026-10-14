"""
Vision Radar - Vision Engine: Tracking & Position
Position/motion estimation from a track's historical center points.

Spec reference: PDF Section 8 (Spatial model - movement direction,
trajectory), Section 6 (World Model example shows "Direction: NE").

Everything here operates in image-space pixel coordinates, never claiming
real-world units (Section 15 - camera calibration honesty rule carried
forward).
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from .models import Track, TrackedPoint

_COMPASS_LABELS = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]


def compute_velocity(history: List[TrackedPoint], window: int = 5) -> Tuple[float, float]:
    """
    Average velocity (px/sec) over the last `window` history points.
    Returns (vx, vy) in image-space pixels/sec. Note: image y grows
    downward, so a positive vy means moving toward the bottom of the frame.
    Returns (0.0, 0.0) if fewer than 2 points are available.
    """
    if len(history) < 2:
        return (0.0, 0.0)

    recent = list(history)[-window:]
    first, last = recent[0], recent[-1]
    dt = last.timestamp - first.timestamp
    if dt <= 0:
        return (0.0, 0.0)

    vx = (last.x - first.x) / dt
    vy = (last.y - first.y) / dt
    return (vx, vy)


def compute_speed(vx: float, vy: float) -> float:
    """Scalar speed (px/sec) from a velocity vector."""
    return math.hypot(vx, vy)


def compute_direction_label(vx: float, vy: float, stationary_threshold: float = 2.0) -> str:
    """
    8-point compass label from an image-space velocity vector, matching the
    style used in the spec's own World Model example ("Direction: NE").
    Image y grows downward, so "N" (up on screen) corresponds to vy < 0.
    Returns "STATIONARY" if speed is below the threshold.
    """
    speed = compute_speed(vx, vy)
    if speed < stationary_threshold:
        return "STATIONARY"

    # Flip y for a conventional compass (screen-up = north).
    angle = math.degrees(math.atan2(-vy, vx)) % 360
    index = int(((angle + 22.5) % 360) // 45)
    return _COMPASS_LABELS[index]


def compute_trajectory(track: Track) -> List[Tuple[float, float]]:
    """Ordered list of (x, y) center points for this track's bounded history,
    for radar-view rendering (Section 16)."""
    return [(p.x, p.y) for p in track.history]


def describe_motion(track: Track, window: int = 5, stationary_threshold: float = 2.0) -> dict:
    """Convenience bundle for dashboards/evidence payloads."""
    vx, vy = compute_velocity(list(track.history), window=window)
    return {
        "velocity_px_per_sec": (round(vx, 2), round(vy, 2)),
        "speed_px_per_sec": round(compute_speed(vx, vy), 2),
        "direction": compute_direction_label(vx, vy, stationary_threshold),
    }
