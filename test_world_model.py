import unittest

from vision_radar.tracking.models import TrackerConfig, TrackState
from vision_radar.tracking.tracker import IoUTracker
from vision_radar.vision.models import BoundingBox, Detection
from vision_radar.world_model.models import MotionState
from vision_radar.world_model.world_model import WorldModel


def det(entity_type, x, y, w=40, h=80, confidence=0.9):
    return Detection(entity_type=entity_type, bbox=BoundingBox(x, y, w, h),
                      confidence=confidence, detector_name="test")


class TestWorldModelBasics(unittest.TestCase):
    def test_update_produces_a_snapshot_per_active_track(self):
        tracker = IoUTracker()
        wm = WorldModel()

        tracks = tracker.update([det("person", 10, 10)], timestamp=0.0)
        entities = wm.update(tracks)

        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0].entity_type, "person")
        self.assertEqual(entities[0].lifecycle_state, TrackState.DETECTED)

    def test_snapshot_reflects_latest_tracker_state_each_call(self):
        cfg = TrackerConfig(confirm_hits=2, tracked_hits=3)
        tracker = IoUTracker(cfg)
        wm = WorldModel()

        for i in range(3):
            tracks = tracker.update([det("person", 10 + i * 5, 10)], timestamp=float(i))
            entities = wm.update(tracks)

        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0].lifecycle_state, TrackState.TRACKED)
        # position should reflect the LAST update, not the first
        self.assertAlmostEqual(entities[0].position[0], 10 + 2 * 5 + 20, delta=1)

    def test_get_entity_by_id(self):
        tracker = IoUTracker()
        wm = WorldModel()
        tracks = tracker.update([det("person", 10, 10)], timestamp=0.0)
        entities = wm.update(tracks)

        fetched = wm.get_entity(entities[0].entity_id)
        self.assertEqual(fetched.entity_id, entities[0].entity_id)

    def test_get_entity_unknown_id_raises_keyerror(self):
        wm = WorldModel()
        with self.assertRaises(KeyError):
            wm.get_entity("does-not-exist")


class TestExitedEntitiesAreRemoved(unittest.TestCase):
    def test_exited_track_disappears_from_world_model(self):
        cfg = TrackerConfig(confirm_hits=1, tracked_hits=1, max_missed_frames=1)
        tracker = IoUTracker(cfg)
        wm = WorldModel()

        tracks = tracker.update([det("person", 10, 10)], timestamp=0.0)
        wm.update(tracks)
        self.assertEqual(len(wm.get_all_entities()), 1)

        # two consecutive misses -> exceeds max_missed_frames=1 -> EXITED
        tracker.update([], timestamp=1.0)
        tracks = tracker.update([], timestamp=2.0)
        entities = wm.update(tracks)

        self.assertEqual(len(entities), 0)
        self.assertEqual(len(wm.get_all_entities()), 0)


class TestMotionState(unittest.TestCase):
    def test_stationary_entity_reports_stationary(self):
        tracker = IoUTracker()
        wm = WorldModel(stationary_speed_threshold=2.0)

        # same position every frame -> zero speed
        for i in range(4):
            tracks = tracker.update([det("person", 100, 100)], timestamp=float(i))
        entities = wm.update(tracks)

        self.assertEqual(entities[0].motion_state, MotionState.STATIONARY)
        self.assertEqual(entities[0].direction, "STATIONARY")

    def test_moving_entity_reports_moving_with_direction(self):
        tracker = IoUTracker()
        wm = WorldModel(stationary_speed_threshold=2.0)

        for i in range(4):
            tracks = tracker.update([det("person", 100 + i * 8, 100)], timestamp=float(i))
        entities = wm.update(tracks)

        self.assertEqual(entities[0].motion_state, MotionState.MOVING)
        self.assertEqual(entities[0].direction, "E")


class TestAggregateCounts(unittest.TestCase):
    def test_count_by_type(self):
        tracker = IoUTracker()
        wm = WorldModel()
        tracks = tracker.update(
            [det("person", 10, 10), det("person", 300, 10), det("car", 500, 500)],
            timestamp=0.0,
        )
        wm.update(tracks)

        counts = wm.count_by_type()
        self.assertEqual(counts.get("person"), 2)
        self.assertEqual(counts.get("car"), 1)

    def test_count_by_motion_state(self):
        tracker = IoUTracker()
        wm = WorldModel(stationary_speed_threshold=2.0)
        # one stationary, one moving slowly enough for IoU continuity
        # (see PHASE4_README known-limitation note on fast relative motion)
        for i in range(4):
            tracks = tracker.update(
                [det("person", 100, 100), det("person", 300 + i * 8, 300)],
                timestamp=float(i),
            )
        wm.update(tracks)

        counts = wm.count_by_motion_state()
        self.assertEqual(counts.get("STATIONARY"), 1)
        self.assertEqual(counts.get("MOVING"), 1)


class TestNormalizedPosition(unittest.TestCase):
    def test_normalized_position_scales_correctly(self):
        tracker = IoUTracker()
        wm = WorldModel()
        tracks = tracker.update([det("person", 100, 50, 40, 40)], timestamp=0.0)
        entities = wm.update(tracks)

        nx, ny = entities[0].normalized_position(frame_width=640, frame_height=480)
        # center = (100+20, 50+20) = (120, 70)
        self.assertAlmostEqual(nx, 120 / 640)
        self.assertAlmostEqual(ny, 70 / 480)

    def test_normalized_position_guards_zero_frame_size(self):
        tracker = IoUTracker()
        wm = WorldModel()
        tracks = tracker.update([det("person", 10, 10)], timestamp=0.0)
        entities = wm.update(tracks)
        self.assertEqual(entities[0].normalized_position(0, 0), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
