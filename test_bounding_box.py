import unittest

from vision_radar.vision.models import BoundingBox


class TestBoundingBox(unittest.TestCase):
    def test_area(self):
        box = BoundingBox(0, 0, 10, 20)
        self.assertEqual(box.area, 200)

    def test_iou_identical_boxes_is_one(self):
        a = BoundingBox(10, 10, 50, 50)
        b = BoundingBox(10, 10, 50, 50)
        self.assertAlmostEqual(a.iou(b), 1.0)

    def test_iou_no_overlap_is_zero(self):
        a = BoundingBox(0, 0, 10, 10)
        b = BoundingBox(100, 100, 10, 10)
        self.assertEqual(a.iou(b), 0.0)

    def test_iou_partial_overlap(self):
        a = BoundingBox(0, 0, 10, 10)   # area 100
        b = BoundingBox(5, 0, 10, 10)   # area 100, overlap 5x10=50
        # union = 100+100-50=150, iou=50/150=0.333...
        self.assertAlmostEqual(a.iou(b), 50 / 150, places=4)

    def test_normalized_coordinates(self):
        box = BoundingBox(50, 25, 100, 50)
        nx, ny, nw, nh = box.normalized(frame_width=200, frame_height=100)
        self.assertAlmostEqual(nx, 0.25)
        self.assertAlmostEqual(ny, 0.25)
        self.assertAlmostEqual(nw, 0.5)
        self.assertAlmostEqual(nh, 0.5)

    def test_normalized_guards_against_zero_frame_size(self):
        box = BoundingBox(0, 0, 10, 10)
        self.assertEqual(box.normalized(0, 0), (0.0, 0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
