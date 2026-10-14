"""
Vision Radar - Vision Engine
Detection pipeline: runs one or more Detectors on a frame, filters low
confidence results, and removes duplicate/overlapping boxes (NMS) per
entity type - so one real person doesn't become five overlapping
"detections" from a sliding-window scan.

Spec reference: PDF Section 5 (Vision engine), Section 14 (Confidence system).
"""

from __future__ import annotations

import time
from typing import List

import cv2
import numpy as np

from .detector import Detector
from .models import Detection, FrameDetections


class DetectionPipeline:
    def __init__(self, detectors: List[Detector], min_confidence: float = 0.3,
                 nms_iou_threshold: float = 0.45):
        if not detectors:
            raise ValueError("DetectionPipeline requires at least one Detector")
        self.detectors = detectors
        self.min_confidence = min_confidence
        self.nms_iou_threshold = nms_iou_threshold

    def process(self, image: np.ndarray, session_id: str, frame_id: int) -> FrameDetections:
        raw: List[Detection] = []
        for detector in self.detectors:
            raw.extend(detector.detect(image))

        filtered = [d for d in raw if d.confidence >= self.min_confidence]
        deduped = self._nms_per_type(filtered)

        return FrameDetections(
            session_id=session_id,
            frame_id=frame_id,
            captured_at=time.time(),
            detections=deduped,
        )

    def _nms_per_type(self, detections: List[Detection]) -> List[Detection]:
        """Apply non-max suppression independently within each entity_type,
        so a 'person' box never suppresses a 'face' box even if they overlap."""
        if not detections:
            return []

        result: List[Detection] = []
        by_type: dict = {}
        for d in detections:
            by_type.setdefault(d.entity_type, []).append(d)

        for entity_type, group in by_type.items():
            if len(group) == 1:
                result.extend(group)
                continue
            boxes = [[d.bbox.x, d.bbox.y, d.bbox.width, d.bbox.height] for d in group]
            scores = [d.confidence for d in group]
            indices = cv2.dnn.NMSBoxes(
                boxes, scores,
                score_threshold=self.min_confidence,
                nms_threshold=self.nms_iou_threshold,
            )
            if len(indices) == 0:
                continue
            # cv2 returns either a flat array or array-of-arrays depending on version
            flat_indices = np.array(indices).flatten()
            for i in flat_indices:
                result.append(group[int(i)])

        return result
