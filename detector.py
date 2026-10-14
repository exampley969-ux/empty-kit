"""
Vision Radar - Vision Engine
Detector abstraction + concrete detectors.

Design rule (mirrors Section 4's VideoSource abstraction): `Detector` is a
generic interface for "find entities of some type in this image". Today we
ship one real, working implementation (HOGPersonDetector, person-only,
offline/no-download) plus one generic reusable one (HaarCascadeDetector,
usable for any entity a Haar cascade exists for). Adding a vehicle or other
object detector later means implementing this same interface - nothing else
in the system changes.

Honesty note (Section 14 - confidence system): neither the HOG SVM decision
value nor a Haar cascade's stage-reject weight is a calibrated probability.
Both are here passed through a sigmoid purely to squash them into a 0-1
range for storage/comparison. This is flagged, not hidden - see
`_sigmoid_confidence` docstring.
"""

from __future__ import annotations

import abc
import math
import os
from typing import List, Optional

import cv2
import numpy as np

from .models import BoundingBox, Detection


def _sigmoid_confidence(raw_score: float, midpoint: float = 0.0, scale: float = 1.0) -> float:
    """
    Squash an unbounded classifier score into (0, 1) via a sigmoid.

    IMPORTANT: this is a monotonic normalization, not a calibrated
    probability. A 0.9 here means "the classifier's internal score was well
    above its decision boundary", not "90% chance this is correct". Section
    14 requires we store and expose confidence, not that it be perfectly
    calibrated - this must not be presented to a user as a statistical
    probability without further calibration work.
    """
    z = (raw_score - midpoint) / scale
    return 1.0 / (1.0 + math.exp(-z))


class Detector(abc.ABC):
    """Abstract base for any single-entity-type (or multi-type) detector."""

    @abc.abstractmethod
    def detect(self, image: np.ndarray) -> List[Detection]:
        """Return all detections found in this single BGR image (numpy array)."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Short identifier used in Detection.detector_name and logs."""


class HOGPersonDetector(Detector):
    """
    Person detector using OpenCV's built-in HOG + Linear SVM pedestrian
    detector. The SVM weights ship inside opencv-python itself
    (`cv2.HOGDescriptor_getDefaultPeopleDetector()`) - no model download
    required, which matters in a network-restricted environment.
    """

    def __init__(self, hit_threshold: float = 0.0, win_stride=(8, 8),
                 padding=(8, 8), scale: float = 1.05):
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.hit_threshold = hit_threshold
        self.win_stride = win_stride
        self.padding = padding
        self.scale = scale

    @property
    def name(self) -> str:
        return "hog_person_detector"

    def detect(self, image: np.ndarray) -> List[Detection]:
        if image is None or image.size == 0:
            return []
        rects, weights = self._hog.detectMultiScale(
            image,
            hitThreshold=self.hit_threshold,
            winStride=self.win_stride,
            padding=self.padding,
            scale=self.scale,
        )
        detections: List[Detection] = []
        for (x, y, w, h), weight in zip(rects, weights):
            conf = _sigmoid_confidence(float(weight), midpoint=0.0, scale=1.0)
            detections.append(Detection(
                entity_type="person",
                bbox=BoundingBox(int(x), int(y), int(w), int(h)),
                confidence=conf,
                detector_name=self.name,
            ))
        return detections


class HaarCascadeDetector(Detector):
    """
    Generic detector backed by any OpenCV Haar cascade XML. Demonstrates the
    plug-in point for additional entity types: instantiate this with a
    different cascade + entity_type/name to detect faces, full bodies,
    license plates etc. without writing new pipeline code.
    """

    def __init__(self, cascade_filename: str, entity_type: str,
                 scale_factor: float = 1.1, min_neighbors: int = 5,
                 min_size=(30, 30)):
        cascade_path = os.path.join(cv2.data.haarcascades, cascade_filename)
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise ValueError(f"Could not load Haar cascade: {cascade_path}")
        self.entity_type = entity_type
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    @property
    def name(self) -> str:
        return f"haar_{self.entity_type}_detector"

    def detect(self, image: np.ndarray) -> List[Detection]:
        if image is None or image.size == 0:
            return []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        rects, reject_levels, level_weights = self._cascade.detectMultiScale3(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
            outputRejectLevels=True,
        )
        detections: List[Detection] = []
        for (x, y, w, h), weight in zip(rects, level_weights):
            # level_weight is a small positive number that grows with match
            # strength; scale chosen empirically so typical matches land
            # mid-to-high range rather than saturating at 1.0 immediately.
            conf = _sigmoid_confidence(float(weight), midpoint=2.0, scale=2.0)
            detections.append(Detection(
                entity_type=self.entity_type,
                bbox=BoundingBox(int(x), int(y), int(w), int(h)),
                confidence=conf,
                detector_name=self.name,
            ))
        return detections
