"""
Vision Radar - Vision Engine
YOLOv8-ONNX-backed detector.

Fills the gap flagged in PHASE2_README.md: a real, working detector for
entity types HOG/Haar cannot cover (vehicles, etc.), without changing the
Detector interface or the pipeline that consumes it.

Runs via onnxruntime (CPU) - no PyTorch/ultralytics needed at inference
time, which matters in this network-restricted sandbox.
"""

from __future__ import annotations

import ast
from typing import Dict, List, Optional

import cv2
import numpy as np
import onnxruntime as ort

from .detector import Detector
from .models import BoundingBox, Detection

# Fallback COCO class names, used only if a model has no embedded metadata.
_COCO_80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

VEHICLE_CLASS_NAMES = ["car", "motorcycle", "bus", "truck"]


def decode_yolov8_output(
    raw_output: np.ndarray,
    class_names: Dict[int, str],
    conf_threshold: float,
    scale: float,
    pad_x: float,
    pad_y: float,
    class_filter: Optional[set] = None,
) -> List[Detection]:
    """
    Pure post-processing function, decoupled from onnxruntime/cv2 image I/O
    so it can be unit-tested with hand-crafted arrays (no real model needed).

    raw_output: shape (84, N) or (1, 84, N) - already squeezed to (84, N) by
                the caller. Rows 0-3 are box [cx, cy, w, h] in the *model's*
                640x640 input space; rows 4.. are per-class scores (already
                sigmoid-activated by the YOLOv8 head, i.e. already 0-1).
    scale, pad_x, pad_y: letterbox parameters to map boxes back to the
                original image's coordinate space.
    """
    if raw_output.ndim == 3:
        raw_output = raw_output[0]

    boxes_xywh = raw_output[0:4, :]          # (4, N)
    class_scores = raw_output[4:, :]          # (num_classes, N)

    best_class_ids = np.argmax(class_scores, axis=0)         # (N,)
    best_scores = class_scores[best_class_ids, np.arange(class_scores.shape[1])]

    detections: List[Detection] = []
    keep = np.where(best_scores >= conf_threshold)[0]

    for idx in keep:
        class_id = int(best_class_ids[idx])
        class_name = class_names.get(class_id, f"class_{class_id}")
        if class_filter is not None and class_name not in class_filter:
            continue

        cx, cy, w, h = boxes_xywh[:, idx]
        # undo letterbox: subtract padding, divide by scale
        x1 = (cx - w / 2 - pad_x) / scale
        y1 = (cy - h / 2 - pad_y) / scale
        box_w = w / scale
        box_h = h / scale

        detections.append(Detection(
            entity_type=class_name,
            bbox=BoundingBox(int(round(x1)), int(round(y1)), int(round(box_w)), int(round(box_h))),
            confidence=float(best_scores[idx]),
            detector_name="yolov8_onnx",
        ))

    return detections


class YoloOnnxDetector(Detector):
    """
    Generic YOLOv8-ONNX detector. Instantiate with class_filter=None to get
    all 80 COCO classes, or restrict to a subset (e.g. vehicle classes) so
    it slots into the architecture as a focused, single-purpose Detector
    like HOGPersonDetector.
    """

    def __init__(self, model_path: str, class_filter: Optional[List[str]] = None,
                 conf_threshold: float = 0.4, input_size: int = 640,
                 detector_label: str = "yolov8_onnx"):
        self._session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.class_filter = set(class_filter) if class_filter else None
        self._detector_label = detector_label
        self.class_names = self._load_class_names()

    def _load_class_names(self) -> Dict[int, str]:
        meta = self._session.get_modelmeta().custom_metadata_map
        names_raw = meta.get("names")
        if names_raw:
            try:
                parsed = ast.literal_eval(names_raw)
                return {int(k): v for k, v in parsed.items()}
            except (ValueError, SyntaxError):
                pass
        return dict(enumerate(_COCO_80))

    @property
    def name(self) -> str:
        return self._detector_label

    def _letterbox(self, image: np.ndarray):
        h, w = image.shape[:2]
        scale = min(self.input_size / h, self.input_size / w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_x = (self.input_size - new_w) / 2
        pad_y = (self.input_size - new_h) / 2
        top, bottom = int(round(pad_y - 0.1)), int(round(pad_y + 0.1))
        left, right = int(round(pad_x - 0.1)), int(round(pad_x + 0.1))
        # ensure exact target size despite rounding
        top = max(top, 0)
        left = max(left, 0)
        bottom = self.input_size - new_h - top
        right = self.input_size - new_w - left

        padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                     cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return padded, scale, left, top

    def detect(self, image: np.ndarray) -> List[Detection]:
        if image is None or image.size == 0:
            return []

        padded, scale, pad_x, pad_y = self._letterbox(image)
        blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]   # NCHW

        outputs = self._session.run(None, {self._input_name: blob})
        raw = outputs[0][0]   # (84, 8400)

        return decode_yolov8_output(
            raw, self.class_names, self.conf_threshold,
            scale, pad_x, pad_y, self.class_filter,
        )


def VehicleDetector(model_path: str, conf_threshold: float = 0.4) -> YoloOnnxDetector:
    """Convenience factory: a YOLOv8-ONNX detector restricted to vehicle classes."""
    return YoloOnnxDetector(
        model_path,
        class_filter=VEHICLE_CLASS_NAMES,
        conf_threshold=conf_threshold,
        detector_label="yolov8_vehicle_detector",
    )
