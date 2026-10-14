from typing import List

from vision_radar.vision.detector import Detector
from vision_radar.vision.models import Detection


class FakeDetector(Detector):
    """Returns a fixed, pre-programmed list of detections regardless of the
    input image - lets pipeline logic (filtering, NMS) be tested without
    depending on real image content or a real model's accuracy."""

    def __init__(self, fixed_detections: List[Detection], detector_name: str = "fake_detector"):
        self._fixed = fixed_detections
        self._name = detector_name

    @property
    def name(self) -> str:
        return self._name

    def detect(self, image) -> List[Detection]:
        return list(self._fixed)
