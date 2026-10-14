import unittest

from vision_radar.tracking.models import Track, TrackedPoint
from vision_radar.tracking.position import (
    compute_direction_label, compute_speed, compute_trajectory, compute_velocity,
    describe_motion,
)
from vision_radar.vision.models import BoundingBox


def make_track_with_history(points):
    """points: list of (x, y, t)"""
    track = Track.new("person", BoundingBox(0, 0, 10, 10), 0.9, points[0][2], history_length=30)
    track.history.clear()
    for x, y, t in points:
        track.history.append(TrackedPoint(x, y, t))
    return track


class TestVelocity(unittest.TestCase):
    def test_insufficient_history_returns_zero(self):
        vx, vy = compute_velocity([TrackedPoint(0, 0, 0.0)])
        self.assertEqual((vx, vy), (0.0, 0.0))

    def test_moving_right_gives_positive_vx_zero_vy(self):
        history = [TrackedPoint(0, 50, 0.0), TrackedPoint(10, 50, 1.0)]
        vx, vy = compute_velocity(history)
        self.assertAlmostEqual(vx, 10.0)
        self.assertAlmostEqual(vy, 0.0)

    def test_moving_up_screen_gives_negative_vy(self):
        # image space: y decreases going "up" on screen
        history = [TrackedPoint(50, 100, 0.0), TrackedPoint(50, 80, 1.0)]
        vx, vy = compute_velocity(history)
        self.assertAlmostEqual(vx, 0.0)
        self.assertAlmostEqual(vy, -20.0)

    def test_zero_time_delta_returns_zero(self):
        history = [TrackedPoint(0, 0, 5.0), TrackedPoint(10, 10, 5.0)]
        vx, vy = compute_velocity(history)
        self.assertEqual((vx, vy), (0.0, 0.0))

    def test_speed_is_euclidean_norm(self):
        self.assertAlmostEqual(compute_speed(3.0, 4.0), 5.0)


class TestDirectionLabel(unittest.TestCase):
    def test_stationary_below_threshold(self):
        self.assertEqual(compute_direction_label(0.1, 0.1, stationary_threshold=2.0), "STATIONARY")

    def test_east(self):
        self.assertEqual(compute_direction_label(10.0, 0.0), "E")

    def test_north_is_negative_vy(self):
        self.assertEqual(compute_direction_label(0.0, -10.0), "N")

    def test_south_is_positive_vy(self):
        self.assertEqual(compute_direction_label(0.0, 10.0), "S")

    def test_north_east(self):
        self.assertEqual(compute_direction_label(10.0, -10.0), "NE")

    def test_south_west(self):
        self.assertEqual(compute_direction_label(-10.0, 10.0), "SW")


class TestTrajectoryAndDescribeMotion(unittest.TestCase):
    def test_trajectory_matches_history_points(self):
        track = make_track_with_history([(0, 0, 0.0), (5, 5, 1.0), (10, 10, 2.0)])
        traj = compute_trajectory(track)
        self.assertEqual(traj, [(0, 0), (5, 5), (10, 10)])

    def test_describe_motion_bundle(self):
        track = make_track_with_history([(0, 50, 0.0), (20, 50, 1.0)])
        motion = describe_motion(track)
        self.assertEqual(motion["direction"], "E")
        self.assertAlmostEqual(motion["speed_px_per_sec"], 20.0)
        self.assertEqual(motion["velocity_px_per_sec"], (20.0, 0.0))


if __name__ == "__main__":
    unittest.main()
