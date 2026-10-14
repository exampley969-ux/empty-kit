import unittest

from vision_radar.world_model.calibration import CameraCalibration


class TestNormalization(unittest.TestCase):
    def test_to_normalized_and_back(self):
        cal = CameraCalibration(frame_width=640, frame_height=480)
        norm = cal.to_normalized((320, 240))
        self.assertAlmostEqual(norm[0], 0.5)
        self.assertAlmostEqual(norm[1], 0.5)

        pixel = cal.to_pixel(norm)
        self.assertAlmostEqual(pixel[0], 320)
        self.assertAlmostEqual(pixel[1], 240)

    def test_to_normalized_guards_zero_size(self):
        cal = CameraCalibration(frame_width=0, frame_height=0)
        self.assertEqual(cal.to_normalized((10, 10)), (0.0, 0.0))


class TestSceneBoundary(unittest.TestCase):
    def test_no_boundary_means_everything_is_within(self):
        cal = CameraCalibration(frame_width=640, frame_height=480)
        self.assertTrue(cal.within_scene_boundary((0, 0)))
        self.assertTrue(cal.within_scene_boundary((10000, 10000)))

    def test_defined_boundary_restricts_area(self):
        cal = CameraCalibration(
            frame_width=640, frame_height=480,
            scene_boundary=[(0, 0), (100, 0), (100, 100), (0, 100)],
        )
        self.assertTrue(cal.within_scene_boundary((50, 50)))
        self.assertFalse(cal.within_scene_boundary((500, 400)))


class TestRealWorldDistanceGuard(unittest.TestCase):
    def test_estimate_real_world_distance_is_not_implemented(self):
        cal = CameraCalibration(frame_width=640, frame_height=480)
        with self.assertRaises(NotImplementedError):
            cal.estimate_real_world_distance((0, 0), (100, 100))

    def test_has_real_world_calibration_defaults_false(self):
        cal = CameraCalibration(frame_width=640, frame_height=480)
        self.assertFalse(cal.has_real_world_calibration)


if __name__ == "__main__":
    unittest.main()
