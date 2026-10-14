import unittest

from vision_radar.state_engine.engine import StateChangeEngine
from vision_radar.state_engine.models import ChangeType, StateChangeConfig
from vision_radar.tracking.models import TrackState
from vision_radar.vision.models import BoundingBox
from vision_radar.world_model.models import EntitySnapshot, MotionState
from vision_radar.world_model.zone import ZoneOccupancy, ZoneState


def make_entity(entity_id, x, y, entity_type="person", motion_state=MotionState.STATIONARY):
    return EntitySnapshot(
        entity_id=entity_id, entity_type=entity_type,
        lifecycle_state=TrackState.TRACKED, motion_state=motion_state,
        bbox=BoundingBox(int(x - 10), int(y - 10), 20, 20),
        position=(x, y), velocity=(0.0, 0.0), speed=0.0, direction="STATIONARY",
        confidence=0.9, first_seen=0.0, last_seen=0.0, dwell_seconds=0.0, trajectory=[],
    )


def occ_map(zone_id, zone_name, entity_ids):
    o = ZoneOccupancy(zone_id=zone_id, zone_name=zone_name, entity_ids=list(entity_ids))
    o.state = ZoneState.ACTIVE if entity_ids else ZoneState.IDLE
    return {zone_id: o}


class TestAppearedDisappeared(unittest.TestCase):
    def test_new_entity_fires_appeared_once(self):
        engine = StateChangeEngine()
        e = make_entity("e1", 10, 10)

        changes1 = engine.update([e], timestamp=0.0)
        changes2 = engine.update([e], timestamp=1.0)

        self.assertEqual([c.change_type for c in changes1], [ChangeType.APPEARED])
        self.assertEqual(changes2, [])  # no repeat APPEARED on the next frame

    def test_disappeared_fires_once_when_entity_vanishes(self):
        engine = StateChangeEngine()
        e = make_entity("e1", 10, 10)
        engine.update([e], timestamp=0.0)

        changes = engine.update([], timestamp=1.0)
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0].change_type, ChangeType.DISAPPEARED)
        self.assertEqual(changes[0].entity_id, "e1")

        # doesn't fire again - the entity is already forgotten
        changes2 = engine.update([], timestamp=2.0)
        self.assertEqual(changes2, [])


class TestMovement(unittest.TestCase):
    def test_small_movement_below_threshold_does_not_fire(self):
        engine = StateChangeEngine(StateChangeConfig(movement_threshold_px=30))
        engine.update([make_entity("e1", 0, 0)], timestamp=0.0)
        changes = engine.update([make_entity("e1", 5, 0)], timestamp=1.0)
        moved = [c for c in changes if c.change_type == ChangeType.MOVED]
        self.assertEqual(len(moved), 0)

    def test_movement_above_threshold_fires_and_resets_baseline(self):
        engine = StateChangeEngine(StateChangeConfig(movement_threshold_px=30))
        engine.update([make_entity("e1", 0, 0)], timestamp=0.0)
        changes = engine.update([make_entity("e1", 40, 0)], timestamp=1.0)
        moved = [c for c in changes if c.change_type == ChangeType.MOVED]
        self.assertEqual(len(moved), 1)
        self.assertEqual(moved[0].extra["to"], (40, 0))

        # small move from the NEW baseline (40,0) should not immediately re-fire
        changes2 = engine.update([make_entity("e1", 45, 0)], timestamp=2.0)
        moved2 = [c for c in changes2 if c.change_type == ChangeType.MOVED]
        self.assertEqual(len(moved2), 0)

    def test_appearing_entity_does_not_immediately_fire_moved(self):
        engine = StateChangeEngine(StateChangeConfig(movement_threshold_px=30))
        changes = engine.update([make_entity("e1", 100, 100)], timestamp=0.0)
        moved = [c for c in changes if c.change_type == ChangeType.MOVED]
        self.assertEqual(len(moved), 0)


class TestStateChanged(unittest.TestCase):
    def test_motion_state_transition_fires_state_changed(self):
        engine = StateChangeEngine()
        engine.update([make_entity("e1", 0, 0, motion_state=MotionState.STATIONARY)], timestamp=0.0)
        changes = engine.update([make_entity("e1", 0, 0, motion_state=MotionState.MOVING)], timestamp=1.0)

        state_changes = [c for c in changes if c.change_type == ChangeType.STATE_CHANGED]
        self.assertEqual(len(state_changes), 1)
        self.assertEqual(state_changes[0].extra, {"from": "STATIONARY", "to": "MOVING"})

    def test_no_state_changed_when_motion_state_stable(self):
        engine = StateChangeEngine()
        engine.update([make_entity("e1", 0, 0, motion_state=MotionState.MOVING)], timestamp=0.0)
        changes = engine.update([make_entity("e1", 1, 0, motion_state=MotionState.MOVING)], timestamp=1.0)
        state_changes = [c for c in changes if c.change_type == ChangeType.STATE_CHANGED]
        self.assertEqual(len(state_changes), 0)


class TestZoneEntryExit(unittest.TestCase):
    def test_zone_entered_fires_when_entity_first_appears_in_zone(self):
        engine = StateChangeEngine()
        e = make_entity("e1", 10, 10)
        engine.update([e], zone_occupancy=occ_map("z1", "Zone A", []), timestamp=0.0)

        changes = engine.update([e], zone_occupancy=occ_map("z1", "Zone A", ["e1"]), timestamp=1.0)
        entered = [c for c in changes if c.change_type == ChangeType.ZONE_ENTERED]
        self.assertEqual(len(entered), 1)
        self.assertEqual(entered[0].zone_id, "z1")

    def test_zone_exited_fires_when_entity_leaves(self):
        engine = StateChangeEngine()
        e = make_entity("e1", 10, 10)
        engine.update([e], zone_occupancy=occ_map("z1", "Zone A", ["e1"]), timestamp=0.0)

        changes = engine.update([e], zone_occupancy=occ_map("z1", "Zone A", []), timestamp=1.0)
        exited = [c for c in changes if c.change_type == ChangeType.ZONE_EXITED]
        self.assertEqual(len(exited), 1)

    def test_no_zone_change_while_staying_inside(self):
        engine = StateChangeEngine()
        e = make_entity("e1", 10, 10)
        engine.update([e], zone_occupancy=occ_map("z1", "Zone A", ["e1"]), timestamp=0.0)
        changes = engine.update([e], zone_occupancy=occ_map("z1", "Zone A", ["e1"]), timestamp=1.0)
        zone_changes = [c for c in changes if c.change_type in (ChangeType.ZONE_ENTERED, ChangeType.ZONE_EXITED)]
        self.assertEqual(len(zone_changes), 0)


class TestDwell(unittest.TestCase):
    def test_dwell_fires_once_after_threshold(self):
        engine = StateChangeEngine(StateChangeConfig(dwell_threshold_seconds=5.0))
        e = make_entity("e1", 10, 10)

        engine.update([e], zone_occupancy=occ_map("z1", "Zone A", ["e1"]), timestamp=0.0)  # enters
        c1 = engine.update([e], zone_occupancy=occ_map("z1", "Zone A", ["e1"]), timestamp=3.0)
        self.assertEqual([c for c in c1 if c.change_type == ChangeType.DWELL], [])

        c2 = engine.update([e], zone_occupancy=occ_map("z1", "Zone A", ["e1"]), timestamp=6.0)
        dwell_changes = [c for c in c2 if c.change_type == ChangeType.DWELL]
        self.assertEqual(len(dwell_changes), 1)
        self.assertAlmostEqual(dwell_changes[0].extra["dwell_seconds"], 6.0)

        # still inside afterwards - must NOT fire again
        c3 = engine.update([e], zone_occupancy=occ_map("z1", "Zone A", ["e1"]), timestamp=10.0)
        self.assertEqual([c for c in c3 if c.change_type == ChangeType.DWELL], [])

    def test_dwell_resets_after_exit_and_reentry(self):
        engine = StateChangeEngine(StateChangeConfig(dwell_threshold_seconds=5.0))
        e = make_entity("e1", 10, 10)

        engine.update([e], zone_occupancy=occ_map("z1", "Zone A", ["e1"]), timestamp=0.0)
        engine.update([e], zone_occupancy=occ_map("z1", "Zone A", ["e1"]), timestamp=6.0)  # dwell fires

        engine.update([e], zone_occupancy=occ_map("z1", "Zone A", []), timestamp=7.0)       # exits
        engine.update([e], zone_occupancy=occ_map("z1", "Zone A", ["e1"]), timestamp=8.0)   # re-enters

        c = engine.update([e], zone_occupancy=occ_map("z1", "Zone A", ["e1"]), timestamp=14.0)
        dwell_changes = [x for x in c if x.change_type == ChangeType.DWELL]
        self.assertEqual(len(dwell_changes), 1)  # fires again for the new visit


class TestDisappearCleansUpZoneState(unittest.TestCase):
    def test_disappearing_inside_a_zone_does_not_leave_dangling_dwell_state(self):
        engine = StateChangeEngine(StateChangeConfig(dwell_threshold_seconds=5.0))
        e = make_entity("e1", 10, 10)
        engine.update([e], zone_occupancy=occ_map("z1", "Zone A", ["e1"]), timestamp=0.0)

        # entity vanishes entirely (no longer in entities OR zone_occupancy)
        changes = engine.update([], zone_occupancy=occ_map("z1", "Zone A", []), timestamp=1.0)
        types = [c.change_type for c in changes]
        self.assertIn(ChangeType.DISAPPEARED, types)

        # internal dwell bookkeeping must be gone - no crash, no phantom dwell later
        self.assertNotIn(("e1", "z1"), engine._dwell_start)


if __name__ == "__main__":
    unittest.main()
