import unittest

import numpy as np

from vision_radar.vision.onnx_yolo_detector import decode_yolov8_output


def make_raw_output(num_boxes=4, num_classes=80):
    """(84, N) array, all-zero class scores by default; tests override specific columns."""
    return np.zeros((4 + num_classes, num_boxes), dtype=np.float32)


class TestDecodeYolov8Output(unittest.TestCase):
    def test_single_high_confidence_box_is_kept(self):
        raw = make_raw_output(num_boxes=1)
        raw[0:4, 0] = [100, 100, 50, 80]  # cx, cy, w, h (model space, no letterbox pad/scale)
        raw[4 + 2, 0] = 0.9               # class id 2 = "car"

        names = {2: "car"}
        dets = decode_yolov8_output(raw, names, conf_threshold=0.4,
                                     scale=1.0, pad_x=0.0, pad_y=0.0)

        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0].entity_type, "car")
        self.assertAlmostEqual(dets[0].confidence, 0.9, places=5)
        # cx=100, w=50 -> x1 = 100-25 = 75
        self.assertEqual(dets[0].bbox.x, 75)
        self.assertEqual(dets[0].bbox.width, 50)

    def test_low_confidence_box_is_dropped(self):
        raw = make_raw_output(num_boxes=1)
        raw[0:4, 0] = [100, 100, 50, 80]
        raw[4 + 2, 0] = 0.1   # below default threshold

        dets = decode_yolov8_output(raw, {2: "car"}, conf_threshold=0.4,
                                     scale=1.0, pad_x=0.0, pad_y=0.0)
        self.assertEqual(len(dets), 0)

    def test_class_filter_excludes_non_matching_classes(self):
        raw = make_raw_output(num_boxes=2)
        raw[0:4, 0] = [50, 50, 20, 20]
        raw[4 + 0, 0] = 0.95   # class 0 = "person"
        raw[0:4, 1] = [150, 150, 30, 30]
        raw[4 + 2, 1] = 0.90   # class 2 = "car"

        names = {0: "person", 2: "car"}
        dets = decode_yolov8_output(raw, names, conf_threshold=0.4,
                                     scale=1.0, pad_x=0.0, pad_y=0.0,
                                     class_filter={"car", "truck", "bus", "motorcycle"})

        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0].entity_type, "car")

    def test_letterbox_undo_scale_and_padding(self):
        # model-space box at (210, 210) with w=h=100, was letterboxed with
        # scale=0.5 and pad=(10,10) -> original space should undo both.
        raw = make_raw_output(num_boxes=1)
        raw[0:4, 0] = [210, 210, 100, 100]
        raw[4 + 2, 0] = 0.99

        dets = decode_yolov8_output(raw, {2: "car"}, conf_threshold=0.4,
                                     scale=0.5, pad_x=10, pad_y=10)

        self.assertEqual(len(dets), 1)
        d = dets[0]
        # cx=210 -> x1_model = 210-50=160; minus pad(10)=150; /scale(0.5) = 300
        self.assertEqual(d.bbox.x, 300)
        self.assertEqual(d.bbox.width, 200)   # 100 / 0.5

    def test_picks_highest_scoring_class_per_box(self):
        raw = make_raw_output(num_boxes=1)
        raw[0:4, 0] = [50, 50, 20, 20]
        raw[4 + 2, 0] = 0.4   # car
        raw[4 + 7, 0] = 0.85  # truck - higher, should win

        dets = decode_yolov8_output(raw, {2: "car", 7: "truck"}, conf_threshold=0.3,
                                     scale=1.0, pad_x=0, pad_y=0)

        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0].entity_type, "truck")
        self.assertAlmostEqual(dets[0].confidence, 0.85, places=5)

    def test_no_boxes_above_threshold_returns_empty_list(self):
        raw = make_raw_output(num_boxes=5)
        dets = decode_yolov8_output(raw, {2: "car"}, conf_threshold=0.4,
                                     scale=1.0, pad_x=0, pad_y=0)
        self.assertEqual(dets, [])


if __name__ == "__main__":
    unittest.main()
