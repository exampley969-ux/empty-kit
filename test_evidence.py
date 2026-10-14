import os
import shutil
import tempfile
import unittest

import numpy as np

from vision_radar.event_engine.evidence import EvidenceBuilder
from vision_radar.event_engine.models import Event, EventType
from vision_radar.tracking.models import TrackState
from vision_radar.vision.models import BoundingBox
from vision_radar.world_model.models import EntitySnapshot, MotionState
from vision_radar.world_model.zone import Zone


def make_event(zone_id="z1", confidence=0.9):
    return Event.new(event_type=EventType.ZONE_ENTRY, entity_ids=["e1"], entity_type="person",
                     start_time=0.0, confidence=confidence, zone_id=zone_id)


def make_entity():
    return EntitySnapshot(
        entity_id="e1", entity_type="person", lifecycle_state=TrackState.TRACKED,
        motion_state=MotionState.MOVING, bbox=BoundingBox(50, 60, 40, 80),
        position=(70, 100), velocity=(5, 0), speed=5.0, direction="E",
        confidence=0.9, first_seen=0.0, last_seen=1.0, dwell_seconds=1.0, trajectory=[],
    )


class TestEvidenceBuilderWithoutFrame(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_evidence_without_frame_has_no_snapshot_path(self):
        builder = EvidenceBuilder(self.tmp_dir)
        event = make_event()
        evidence = builder.build(event, entity=make_entity())

        self.assertIsNone(evidence.frame_snapshot_path)
        self.assertEqual(evidence.entity_location["position"], (70, 100))
        self.assertEqual(evidence.confidence, 0.9)

    def test_evidence_uses_end_time_when_resolved(self):
        builder = EvidenceBuilder(self.tmp_dir)
        event = make_event()
        event.end_time = 25.0
        evidence = builder.build(event)
        self.assertEqual(evidence.timestamp, 25.0)

    def test_evidence_uses_start_time_when_still_active(self):
        builder = EvidenceBuilder(self.tmp_dir)
        event = make_event()   # end_time stays None
        evidence = builder.build(event)
        self.assertEqual(evidence.timestamp, event.start_time)

    def test_zone_geometry_included_when_zone_given(self):
        builder = EvidenceBuilder(self.tmp_dir)
        zone = Zone.from_rect("Restricted-A", 0, 0, 100, 100)
        evidence = builder.build(make_event(), zone=zone)
        self.assertEqual(evidence.zone_geometry, zone.polygon)


class TestEvidenceBuilderWithRealFrame(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_frame_snapshot_is_a_real_saved_image(self):
        builder = EvidenceBuilder(self.tmp_dir)
        frame = np.full((200, 300, 3), 120, dtype=np.uint8)
        zone = Zone.from_rect("Restricted-A", 10, 10, 150, 150)
        event = make_event()

        evidence = builder.build(event, entity=make_entity(), zone=zone, frame=frame)

        self.assertIsNotNone(evidence.frame_snapshot_path)
        self.assertTrue(os.path.exists(evidence.frame_snapshot_path))
        self.assertGreater(os.path.getsize(evidence.frame_snapshot_path), 0)

        # It must actually be a readable, correctly-sized image, not a
        # placeholder or empty file - reload it and verify.
        import cv2
        reloaded = cv2.imread(evidence.frame_snapshot_path)
        self.assertIsNotNone(reloaded)
        self.assertEqual(reloaded.shape[:2], (200, 300))

    def test_annotated_frame_differs_from_original_when_entity_present(self):
        # The bbox/zone overlay must actually change pixel data - otherwise
        # this "evidence" would be indistinguishable from a raw, unannotated
        # frame and would not actually show why the event fired.
        builder = EvidenceBuilder(self.tmp_dir)
        frame = np.full((200, 300, 3), 120, dtype=np.uint8)
        event = make_event()

        evidence = builder.build(event, entity=make_entity(), frame=frame)

        import cv2
        reloaded = cv2.imread(evidence.frame_snapshot_path)
        self.assertFalse(np.array_equal(reloaded, frame))

    def test_filename_is_keyed_by_event_id(self):
        builder = EvidenceBuilder(self.tmp_dir)
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        event = make_event()
        evidence = builder.build(event, frame=frame)
        self.assertIn(event.event_id, evidence.frame_snapshot_path)


if __name__ == "__main__":
    unittest.main()


class TestLabelPlacement(unittest.TestCase):
    """Found by looking at a real evidence image: the label was clipped when
    the box touched the top edge of the frame."""

    def test_label_sits_above_the_box_when_there_is_room(self):
        self.assertEqual(EvidenceBuilder._label_origin(50, 100), (50, 92))

    def test_label_moves_inside_the_box_at_the_top_edge(self):
        x, y = EvidenceBuilder._label_origin(50, 18)      # the real case: bbox y=18
        self.assertGreaterEqual(y, 12)                    # fully on-screen
        self.assertGreater(y, 18)                         # inside the box, not above it

    def test_label_is_never_placed_at_a_negative_position(self):
        x, y = EvidenceBuilder._label_origin(-4, -30)     # detector boxes can extend past the frame
        self.assertGreaterEqual(x, 0)
        self.assertGreaterEqual(y, 12)
