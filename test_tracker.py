import unittest

from vision_radar.tracking.models import TrackerConfig, TrackState
from vision_radar.tracking.tracker import IoUTracker
from vision_radar.vision.models import BoundingBox, Detection


def det(entity_type, x, y, w=40, h=80, confidence=0.9):
    return Detection(entity_type=entity_type, bbox=BoundingBox(x, y, w, h),
                      confidence=confidence, detector_name="test")


class TestNewTrackCreation(unittest.TestCase):
    def test_first_detection_creates_track_in_detected_state(self):
        tracker = IoUTracker()
        tracks = tracker.update([det("person", 10, 10)], timestamp=0.0)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].state, TrackState.DETECTED)
        self.assertEqual(tracks[0].entity_type, "person")


class TestIdentityPersistence(unittest.TestCase):
    def test_same_entity_moving_slightly_keeps_same_track_id(self):
        tracker = IoUTracker()
        t0 = tracker.update([det("person", 10, 10)], timestamp=0.0)
        track_id = t0[0].track_id

        # small movement each frame - should stay matched to the same track
        for i in range(1, 5):
            tracks = tracker.update([det("person", 10 + i * 2, 10)], timestamp=float(i))

        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].track_id, track_id)

    def test_two_simultaneous_entities_get_different_track_ids(self):
        tracker = IoUTracker()
        tracks = tracker.update([det("person", 10, 10), det("person", 300, 300)], timestamp=0.0)
        self.assertEqual(len(tracks), 2)
        self.assertNotEqual(tracks[0].track_id, tracks[1].track_id)

    def test_different_entity_types_never_share_a_track(self):
        tracker = IoUTracker(TrackerConfig(iou_match_threshold=0.1))
        # same location, different types - must NOT merge into one track
        tracker.update([det("person", 10, 10, 40, 80), det("car", 10, 10, 40, 80)], timestamp=0.0)
        tracks = tracker.update([det("person", 10, 10, 40, 80), det("car", 10, 10, 40, 80)], timestamp=1.0)
        self.assertEqual(len(tracks), 2)
        types = {t.entity_type for t in tracks}
        self.assertEqual(types, {"person", "car"})

    def test_far_apart_detection_of_same_type_does_not_match_becomes_new_track(self):
        tracker = IoUTracker()
        tracker.update([det("person", 10, 10, 40, 80)], timestamp=0.0)
        # completely different location next frame -> IoU ~0, should NOT
        # match the existing track; instead the old one misses and a new
        # one is created.
        tracks = tracker.update([det("person", 500, 500, 40, 80)], timestamp=1.0)
        # old track (now missed once, still DETECTED pre-confirm) exits
        # immediately per lifecycle rule; new track created for the new detection
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].state, TrackState.DETECTED)


class TestLifecycleProgression(unittest.TestCase):
    def test_progresses_detected_confirmed_tracked_with_enough_hits(self):
        cfg = TrackerConfig(confirm_hits=2, tracked_hits=4)
        tracker = IoUTracker(cfg)
        states = []
        for i in range(6):
            tracks = tracker.update([det("person", 10 + i, 10)], timestamp=float(i))
            states.append(tracks[0].state)

        self.assertEqual(states[0], TrackState.DETECTED)   # hit 1
        self.assertEqual(states[1], TrackState.CONFIRMED)  # hit 2 == confirm_hits
        self.assertEqual(states[2], TrackState.CONFIRMED)  # hit 3, not yet tracked_hits
        self.assertEqual(states[3], TrackState.TRACKED)     # hit 4 == tracked_hits
        self.assertEqual(states[4], TrackState.TRACKED)
        self.assertEqual(states[5], TrackState.TRACKED)


class TestMissesAndRecovery(unittest.TestCase):
    def _build_tracked_entity(self, tracker, cfg):
        track_id = None
        for i in range(cfg.tracked_hits):
            tracks = tracker.update([det("person", 100 + i, 100)], timestamp=float(i))
            track_id = tracks[0].track_id
        return track_id

    def test_established_track_goes_temporarily_lost_on_miss(self):
        cfg = TrackerConfig(confirm_hits=2, tracked_hits=3, max_missed_frames=5)
        tracker = IoUTracker(cfg)
        track_id = self._build_tracked_entity(tracker, cfg)

        # next frame: no detections at all -> the track should miss
        tracks = tracker.update([], timestamp=100.0)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].track_id, track_id)
        self.assertEqual(tracks[0].state, TrackState.TEMPORARILY_LOST)

    def test_reacquired_then_back_to_tracked(self):
        cfg = TrackerConfig(confirm_hits=2, tracked_hits=3, max_missed_frames=5)
        tracker = IoUTracker(cfg)
        track_id = self._build_tracked_entity(tracker, cfg)
        last_track = tracker.get_track(track_id)
        last_bbox = last_track.bbox

        tracker.update([], timestamp=100.0)  # miss -> TEMPORARILY_LOST

        # reappears near the same spot
        tracks = tracker.update(
            [det("person", last_bbox.x, last_bbox.y, last_bbox.width, last_bbox.height)],
            timestamp=101.0,
        )
        self.assertEqual(tracks[0].track_id, track_id)
        self.assertEqual(tracks[0].state, TrackState.REACQUIRED)

        tracks = tracker.update(
            [det("person", last_bbox.x, last_bbox.y, last_bbox.width, last_bbox.height)],
            timestamp=102.0,
        )
        self.assertEqual(tracks[0].state, TrackState.TRACKED)

    def test_exceeding_max_missed_frames_exits_the_track(self):
        cfg = TrackerConfig(confirm_hits=2, tracked_hits=3, max_missed_frames=2)
        tracker = IoUTracker(cfg)
        track_id = self._build_tracked_entity(tracker, cfg)

        for i in range(5):
            tracker.update([], timestamp=100.0 + i)

        active = tracker.active_tracks()
        self.assertEqual(len(active), 0)
        self.assertEqual(tracker.get_track(track_id).state, TrackState.EXITED)

    def test_unconfirmed_track_exits_immediately_on_first_miss(self):
        # A single DETECTED (not yet confirmed) hit that then misses is
        # treated as a likely false positive, not given a recovery window.
        cfg = TrackerConfig(confirm_hits=3, max_missed_frames=5)
        tracker = IoUTracker(cfg)
        tracker.update([det("person", 10, 10)], timestamp=0.0)  # DETECTED, hits=1
        tracker.update([], timestamp=1.0)  # immediate miss

        self.assertEqual(len(tracker.active_tracks()), 0)


if __name__ == "__main__":
    unittest.main()
