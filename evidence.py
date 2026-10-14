"""
Vision Radar - Event Engine: Evidence
Builds inspectable evidence for an event: timestamp, entity location,
zone geometry, confidence, and - when a real frame is available - an
actual annotated image saved to disk (not just a path placeholder).

Spec reference: PDF Section 20 (Evidence - "timestamp, relevant frame,
entity location, zone geometry, confidence... so the user can inspect why
the system generated the event").
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from vision_radar.event_engine.models import Event
from vision_radar.world_model.models import EntitySnapshot
from vision_radar.world_model.zone import Zone


@dataclass
class Evidence:
    event_id: str
    timestamp: float
    confidence: float
    entity_location: Optional[dict] = None          # {"position": (x,y), "bbox": (x,y,w,h)}
    zone_geometry: Optional[List[Tuple[float, float]]] = None
    frame_snapshot_path: Optional[str] = None         # real saved annotated image, if a frame was given


class EvidenceBuilder:
    def __init__(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

    def build(self, event: Event, entity: Optional[EntitySnapshot] = None,
              zone: Optional[Zone] = None, frame: Optional[np.ndarray] = None) -> Evidence:
        entity_location = None
        if entity is not None:
            b = entity.bbox
            entity_location = {"position": entity.position, "bbox": (b.x, b.y, b.width, b.height)}

        zone_geometry = zone.polygon if zone is not None else None

        frame_snapshot_path = None
        if frame is not None:
            frame_snapshot_path = self._save_annotated_frame(event, entity_location, zone_geometry, frame)

        return Evidence(
            event_id=event.event_id,
            timestamp=event.end_time if event.end_time is not None else event.start_time,
            confidence=event.confidence,
            entity_location=entity_location,
            zone_geometry=zone_geometry,
            frame_snapshot_path=frame_snapshot_path,
        )

    @staticmethod
    def _label_origin(x: int, y: int) -> Tuple[int, int]:
        """Where to draw the text label: just above the box, but tucked inside
        it when the box touches the top edge (otherwise the text is clipped
        off the image), and never at a negative x."""
        label_y = y - 8 if y - 8 >= 12 else y + 16
        return (max(0, x), max(12, label_y))

    def _save_annotated_frame(self, event: Event, entity_location: Optional[dict],
                               zone_geometry: Optional[List[Tuple[float, float]]],
                               frame: np.ndarray) -> str:
        annotated = frame.copy()

        if zone_geometry:
            pts = np.array(zone_geometry, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(annotated, [pts], isClosed=True, color=(0, 200, 0), thickness=2)

        if entity_location:
            x, y, w, h = entity_location["bbox"]
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
            label = f"{event.entity_type} {event.confidence:.2f}"
            cv2.putText(annotated, label, self._label_origin(x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        filename = f"{event.event_id}.jpg"
        path = os.path.join(self.output_dir, filename)
        cv2.imwrite(path, annotated)
        return path
