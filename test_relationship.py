import unittest

from vision_radar.tracking.models import TrackState
from vision_radar.vision.models import BoundingBox
from vision_radar.world_model.models import EntitySnapshot, MotionState
from vision_radar.world_model.relationship import (
    RelationshipConfig, RelationshipEngine, RelationshipType,
)


def make_entity(entity_id, x, y, entity_type="person", motion_state=MotionState.STATIONARY,
                 speed=0.0, direction="STATIONARY"):
    return EntitySnapshot(
        entity_id=entity_id, entity_type=entity_type,
        lifecycle_state=TrackState.TRACKED, motion_state=motion_state,
        bbox=BoundingBox(int(x - 10), int(y - 10), 20, 20),
        position=(x, y), velocity=(0.0, 0.0), speed=speed, direction=direction,
        confidence=0.9, first_seen=0.0, last_seen=0.0, dwell_seconds=0.0, trajectory=[],
    )


class TestProximity(unittest.TestCase):
    def test_close_entities_are_near(self):
        engine = RelationshipEngine(RelationshipConfig(proximity_threshold_px=100))
        a = make_entity("person_1", 100, 100, entity_type="person")
        b = make_entity("car_1", 150, 100, entity_type="car")

        rels = engine.update([a, b], timestamp=0.0)

        near = [r for r in rels if r.relationship_type == RelationshipType.NEAR]
        self.assertEqual(len(near), 1)
        self.assertEqual({near[0].entity_a_id, near[0].entity_b_id}, {"person_1", "car_1"})

    def test_far_entities_are_not_near(self):
        engine = RelationshipEngine(RelationshipConfig(proximity_threshold_px=100))
        a = make_entity("person_1", 0, 0)
        b = make_entity("car_1", 1000, 1000)

        rels = engine.update([a, b], timestamp=0.0)
        self.assertEqual(len(rels), 0)

    def test_relationship_id_stable_while_condition_holds(self):
        engine = RelationshipEngine(RelationshipConfig(proximity_threshold_px=100))
        a = make_entity("p1", 100, 100)
        b = make_entity("c1", 140, 100)

        rels1 = engine.update([a, b], timestamp=0.0)
        rels2 = engine.update([a, b], timestamp=1.0)

        self.assertEqual(rels1[0].relationship_id, rels2[0].relationship_id)
        self.assertEqual(rels2[0].last_observed, 1.0)
        self.assertEqual(rels2[0].first_observed, 0.0)

    def test_relationship_ends_and_gets_new_id_if_restarted(self):
        engine = RelationshipEngine(RelationshipConfig(proximity_threshold_px=50))
        a = make_entity("p1", 0, 0)
        b_near = make_entity("c1", 30, 0)
        b_far = make_entity("c1", 1000, 0)

        rels1 = engine.update([a, b_near], timestamp=0.0)
        first_id = rels1[0].relationship_id

        rels2 = engine.update([a, b_far], timestamp=1.0)
        self.assertEqual(len(rels2), 0)  # relationship ended

        rels3 = engine.update([a, b_near], timestamp=2.0)
        self.assertEqual(len(rels3), 1)
        self.assertNotEqual(rels3[0].relationship_id, first_id)  # documented scope: new id, not resumed


class TestCoMovement(unittest.TestCase):
    def test_same_direction_similar_speed_is_moving_together(self):
        cfg = RelationshipConfig(min_comovement_speed=5.0, comovement_speed_ratio_tolerance=0.3)
        engine = RelationshipEngine(cfg)
        a = make_entity("p1", 0, 0, motion_state=MotionState.MOVING, speed=20.0, direction="E")
        b = make_entity("p2", 500, 0, motion_state=MotionState.MOVING, speed=22.0, direction="E")

        rels = engine.update([a, b], timestamp=0.0)
        together = [r for r in rels if r.relationship_type == RelationshipType.MOVING_TOGETHER]
        self.assertEqual(len(together), 1)

    def test_different_direction_is_not_moving_together(self):
        cfg = RelationshipConfig(min_comovement_speed=5.0)
        engine = RelationshipEngine(cfg)
        a = make_entity("p1", 0, 0, motion_state=MotionState.MOVING, speed=20.0, direction="E")
        b = make_entity("p2", 500, 0, motion_state=MotionState.MOVING, speed=20.0, direction="W")

        rels = engine.update([a, b], timestamp=0.0)
        together = [r for r in rels if r.relationship_type == RelationshipType.MOVING_TOGETHER]
        self.assertEqual(len(together), 0)

    def test_stationary_entity_is_not_moving_together(self):
        cfg = RelationshipConfig(min_comovement_speed=5.0)
        engine = RelationshipEngine(cfg)
        a = make_entity("p1", 0, 0, motion_state=MotionState.MOVING, speed=20.0, direction="E")
        b = make_entity("p2", 500, 0, motion_state=MotionState.STATIONARY, speed=0.0, direction="STATIONARY")

        rels = engine.update([a, b], timestamp=0.0)
        together = [r for r in rels if r.relationship_type == RelationshipType.MOVING_TOGETHER]
        self.assertEqual(len(together), 0)

    def test_very_different_speed_is_not_moving_together(self):
        cfg = RelationshipConfig(min_comovement_speed=5.0, comovement_speed_ratio_tolerance=0.2)
        engine = RelationshipEngine(cfg)
        a = make_entity("p1", 0, 0, motion_state=MotionState.MOVING, speed=10.0, direction="E")
        b = make_entity("p2", 500, 0, motion_state=MotionState.MOVING, speed=50.0, direction="E")

        rels = engine.update([a, b], timestamp=0.0)
        together = [r for r in rels if r.relationship_type == RelationshipType.MOVING_TOGETHER]
        self.assertEqual(len(together), 0)


if __name__ == "__main__":
    unittest.main()
