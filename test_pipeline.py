import unittest

import numpy as np

from vision_radar.tests.fake_detectors import FakeDetector
from vision_radar.vision.models import BoundingBox, Detection
from vision_radar.vision.pipeline import DetectionPipeline

BLANK_IMAGE = np.zeros((100, 100, 3), dtype=np.uint8)


def det(entity_type, x, y, w, h, confidence, detector_name="fake"):
    return Detection(
        entity_type=entity_type,
        bbox=BoundingBox(x, y, w, h),
        confidence=confidence,
        detector_name=detector_name,
    )


class TestDetectionPipelineAggregation(unittest.TestCase):
    def test_combines_results_from_multiple_detectors(self):
        d1 = FakeDetector([det("person", 0, 0, 10, 10, 0.9)], detector_name="d1")
        d2 = FakeDetector([det("face", 50, 50, 5, 5, 0.8)], detector_name="d2")
        pipeline = DetectionPipeline([d1, d2], min_confidence=0.3)

        result = pipeline.process(BLANK_IMAGE, session_id="s1", frame_id=0)

        self.assertEqual(len(result.detections), 2)
        types = {d.entity_type for d in result.detections}
        self.assertEqual(types, {"person", "face"})
        self.assertEqual(result.session_id, "s1")
        self.assertEqual(result.frame_id, 0)

    def test_filters_out_low_confidence_detections(self):
        d = FakeDetector([
            det("person", 0, 0, 10, 10, 0.9),
            det("person", 50, 50, 10, 10, 0.1),   # below threshold
        ])
        pipeline = DetectionPipeline([d], min_confidence=0.3)

        result = pipeline.process(BLANK_IMAGE, "s1", 0)

        self.assertEqual(len(result.detections), 1)
        self.assertAlmostEqual(result.detections[0].confidence, 0.9)

    def test_empty_detectors_list_raises(self):
        with self.assertRaises(ValueError):
            DetectionPipeline([])


class TestDetectionPipelineNMS(unittest.TestCase):
    def test_overlapping_same_type_boxes_are_deduplicated(self):
        # Two boxes for the same "person", heavily overlapping - should
        # collapse to the higher-confidence one.
        d = FakeDetector([
            det("person", 10, 10, 40, 40, 0.95),
            det("person", 12, 12, 40, 40, 0.60),  # near-duplicate, lower conf
        ])
        pipeline = DetectionPipeline([d], min_confidence=0.3, nms_iou_threshold=0.3)

        result = pipeline.process(BLANK_IMAGE, "s1", 0)

        self.assertEqual(len(result.detections), 1)
        self.assertAlmostEqual(result.detections[0].confidence, 0.95)

    def test_non_overlapping_same_type_boxes_are_both_kept(self):
        d = FakeDetector([
            det("person", 0, 0, 10, 10, 0.9),
            det("person", 80, 80, 10, 10, 0.8),
        ])
        pipeline = DetectionPipeline([d], min_confidence=0.3, nms_iou_threshold=0.3)

        result = pipeline.process(BLANK_IMAGE, "s1", 0)

        self.assertEqual(len(result.detections), 2)

    def test_nms_does_not_suppress_across_different_entity_types(self):
        # A "face" box fully inside a "person" box should NOT be suppressed -
        # they are different entity types (Section 5: classification matters).
        d = FakeDetector([
            det("person", 0, 0, 50, 50, 0.9),
            det("face", 10, 10, 15, 15, 0.85),
        ])
        pipeline = DetectionPipeline([d], min_confidence=0.3, nms_iou_threshold=0.3)

        result = pipeline.process(BLANK_IMAGE, "s1", 0)

        self.assertEqual(len(result.detections), 2)
        types = {d.entity_type for d in result.detections}
        self.assertEqual(types, {"person", "face"})

    def test_count_by_type(self):
        d = FakeDetector([
            det("person", 0, 0, 10, 10, 0.9),
            det("person", 80, 80, 10, 10, 0.8),
            det("face", 40, 40, 5, 5, 0.7),
        ])
        pipeline = DetectionPipeline([d], min_confidence=0.3, nms_iou_threshold=0.3)

        result = pipeline.process(BLANK_IMAGE, "s1", 0)
        counts = result.count_by_type()

        self.assertEqual(counts.get("person"), 2)
        self.assertEqual(counts.get("face"), 1)


if __name__ == "__main__":
    unittest.main()
