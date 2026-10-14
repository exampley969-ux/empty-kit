import unittest

from vision_radar.tracking.models import TrackState
from vision_radar.vision.models import BoundingBox
from vision_radar.world_model.models import EntitySnapshot, MotionState
from vision_radar.world_model.zone import Zone, ZoneManager, ZoneState, point_in_polygon


def make_entity(entity_id, x, y, entity_type="person"):
    return EntitySnapshot(
        entity_id=entity_id, entity_type=entity_type,
        lifecycle_state=TrackState.TRACKED, motion_state=MotionState.STATIONARY,
        bbox=BoundingBox(int(x - 10), int(y - 10), 20, 20),
        position=(x, y), velocity=(0.0, 0.0), speed=0.0, direction="STATIONARY",
        confidence=0.9, first_seen=0.0, last_seen=0.0, dwell_seconds=0.0, trajectory=[],
    )


class TestPointInPolygon(unittest.TestCase):
    def test_point_inside_rectangle(self):
        rect = [(0, 0), (100, 0), (100, 100), (0, 100)]
        self.assertTrue(point_in_polygon((50, 50), rect))

    def test_point_outside_rectangle(self):
        rect = [(0, 0), (100, 0), (100, 100), (0, 100)]
        self.assertFalse(point_in_polygon((150, 50), rect))

    def test_point_inside_triangle(self):
        triangle = [(0, 0), (100, 0), (50, 100)]
        self.assertTrue(point_in_polygon((50, 30), triangle))
        self.assertFalse(point_in_polygon((5, 90), triangle))


class TestZoneConstruction(unittest.TestCase):
    def test_from_rect_contains_center_point(self):
        zone = Zone.from_rect("Entrance", x=0, y=0, width=200, height=100)
        self.assertTrue(zone.contains((100, 50)))
        self.assertFalse(zone.contains((300, 300)))

    def test_from_polygon_requires_at_least_three_points(self):
        with self.assertRaises(ValueError):
            Zone.from_polygon("Bad", [(0, 0), (10, 10)])

    def test_from_polygon_arbitrary_shape(self):
        zone = Zone.from_polygon("Wedge", [(0, 0), (100, 0), (50, 100)])
        self.assertTrue(zone.contains((50, 30)))


class TestZoneManagerOccupancy(unittest.TestCase):
    def test_empty_zone_is_idle(self):
        mgr = ZoneManager()
        zone = Zone.from_rect("A", 0, 0, 100, 100)
        mgr.add_zone(zone)

        occ = mgr.compute_occupancy([])
        self.assertEqual(occ[zone.zone_id].state, ZoneState.IDLE)
        self.assertEqual(occ[zone.zone_id].occupancy, 0)

    def test_entity_inside_zone_counted(self):
        mgr = ZoneManager()
        zone = Zone.from_rect("A", 0, 0, 100, 100)
        mgr.add_zone(zone)
        entities = [make_entity("e1", 50, 50)]

        occ = mgr.compute_occupancy(entities)
        self.assertEqual(occ[zone.zone_id].occupancy, 1)
        self.assertEqual(occ[zone.zone_id].state, ZoneState.ACTIVE)
        self.assertIn("e1", occ[zone.zone_id].entity_ids)

    def test_entity_outside_zone_not_counted(self):
        mgr = ZoneManager()
        zone = Zone.from_rect("A", 0, 0, 100, 100)
        mgr.add_zone(zone)
        entities = [make_entity("e1", 500, 500)]

        occ = mgr.compute_occupancy(entities)
        self.assertEqual(occ[zone.zone_id].occupancy, 0)
        self.assertEqual(occ[zone.zone_id].state, ZoneState.IDLE)

    def test_overlapping_zones_both_count_the_same_entity(self):
        mgr = ZoneManager()
        zone_a = Zone.from_rect("A", 0, 0, 100, 100)
        zone_b = Zone.from_rect("B", 50, 50, 100, 100)   # overlaps zone A
        mgr.add_zone(zone_a)
        mgr.add_zone(zone_b)
        entities = [make_entity("e1", 75, 75)]   # inside the overlap region

        occ = mgr.compute_occupancy(entities)
        self.assertEqual(occ[zone_a.zone_id].occupancy, 1)
        self.assertEqual(occ[zone_b.zone_id].occupancy, 1)

    def test_remove_and_get_zone(self):
        mgr = ZoneManager()
        zone = Zone.from_rect("A", 0, 0, 10, 10)
        mgr.add_zone(zone)
        self.assertEqual(len(mgr.list_zones()), 1)
        mgr.remove_zone(zone.zone_id)
        self.assertEqual(len(mgr.list_zones()), 0)
        with self.assertRaises(KeyError):
            mgr.get_zone(zone.zone_id)


if __name__ == "__main__":
    unittest.main()
