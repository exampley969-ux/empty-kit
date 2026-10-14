import unittest

from vision_radar.event_engine.engine import EventEngine
from vision_radar.event_engine.models import EventConfig, EventState, EventType
from vision_radar.state_engine.models import Change, ChangeType


def change(change_type, entity_id="e1", entity_type="person", timestamp=0.0,
           zone_id=None, position=None, extra=None):
    return Change(change_type=change_type, entity_id=entity_id, entity_type=entity_type,
                  timestamp=timestamp, zone_id=zone_id, position=position, extra=extra or {})


class TestInstantaneousEvents(unittest.TestCase):
    def test_appeared_creates_resolved_event_with_zero_duration(self):
        engine = EventEngine()
        events = engine.update([change(ChangeType.APPEARED, position=(1, 2))], timestamp=5.0)

        self.assertEqual(len(events), 1)
        e = events[0]
        self.assertEqual(e.event_type, EventType.APPEARED)
        self.assertEqual(e.state, EventState.RESOLVED)
        self.assertEqual(e.start_time, 5.0)
        self.assertEqual(e.end_time, 5.0)
        self.assertEqual(e.duration_seconds(), 0.0)
        self.assertEqual(e.entity_ids, ["e1"])

    def test_confidence_comes_from_lookup_when_provided(self):
        engine = EventEngine()
        events = engine.update([change(ChangeType.APPEARED)], timestamp=0.0,
                                confidence_lookup={"e1": 0.87})
        self.assertAlmostEqual(events[0].confidence, 0.87)

    def test_confidence_falls_back_to_default_when_missing(self):
        cfg = EventConfig(default_confidence=0.42)
        engine = EventEngine(cfg)
        events = engine.update([change(ChangeType.APPEARED)], timestamp=0.0)
        self.assertAlmostEqual(events[0].confidence, 0.42)

    def test_moved_event_carries_change_extra_as_evidence(self):
        engine = EventEngine()
        c = change(ChangeType.MOVED, extra={"from": (0, 0), "to": (50, 0), "distance": 50.0})
        events = engine.update([c], timestamp=1.0)
        self.assertEqual(events[0].evidence["distance"], 50.0)


class TestZoneEntryDeduplication(unittest.TestCase):
    def test_single_zone_entered_creates_one_active_event(self):
        engine = EventEngine()
        events = engine.update(
            [change(ChangeType.ZONE_ENTERED, zone_id="z1")], timestamp=10.0
        )
        self.assertEqual(len(events), 1)
        e = events[0]
        self.assertEqual(e.event_type, EventType.ZONE_ENTRY)
        self.assertEqual(e.state, EventState.ACTIVE)
        self.assertEqual(e.start_time, 10.0)
        self.assertIsNone(e.end_time)
        self.assertEqual(e.zone_id, "z1")

    def test_no_new_event_while_entity_remains_in_zone(self):
        # Simulates Section 13's own example: entering once should not spam
        # one event per frame while the entity stays inside.
        engine = EventEngine()
        engine.update([change(ChangeType.ZONE_ENTERED, zone_id="z1")], timestamp=0.0)

        # Phase 6 would not re-emit ZONE_ENTERED while still inside - so no
        # further zone changes arrive here either. Confirm no phantom event
        # appears from an empty changes list.
        events = engine.update([], timestamp=5.0)
        self.assertEqual(events, [])
        self.assertEqual(len(engine.list_events(event_type=EventType.ZONE_ENTRY)), 1)

    def test_zone_exit_resolves_the_same_event_with_correct_duration(self):
        engine = EventEngine()
        entered = engine.update(
            [change(ChangeType.ZONE_ENTERED, zone_id="z1")], timestamp=10.0
        )[0]

        exited = engine.update(
            [change(ChangeType.ZONE_EXITED, zone_id="z1")], timestamp=28.0
        )[0]

        self.assertEqual(exited.event_id, entered.event_id)   # same event, not a new one
        self.assertEqual(exited.state, EventState.RESOLVED)
        self.assertEqual(exited.end_time, 28.0)
        self.assertAlmostEqual(exited.duration_seconds(), 18.0)

    def test_one_full_entry_exit_cycle_produces_exactly_one_event_total(self):
        engine = EventEngine()
        engine.update([change(ChangeType.ZONE_ENTERED, zone_id="z1")], timestamp=0.0)
        engine.update([], timestamp=1.0)
        engine.update([], timestamp=2.0)
        engine.update([change(ChangeType.ZONE_EXITED, zone_id="z1")], timestamp=3.0)

        all_zone_events = engine.list_events(event_type=EventType.ZONE_ENTRY)
        self.assertEqual(len(all_zone_events), 1)

    def test_zone_exit_with_no_matching_open_entry_is_ignored_not_crashed(self):
        engine = EventEngine()
        events = engine.update(
            [change(ChangeType.ZONE_EXITED, zone_id="z1")], timestamp=0.0
        )
        self.assertEqual(events, [])

    def test_independent_entities_get_independent_zone_events(self):
        engine = EventEngine()
        e1 = engine.update(
            [change(ChangeType.ZONE_ENTERED, entity_id="e1", zone_id="z1")], timestamp=0.0
        )[0]
        e2 = engine.update(
            [change(ChangeType.ZONE_ENTERED, entity_id="e2", zone_id="z1")], timestamp=0.0
        )[0]
        self.assertNotEqual(e1.event_id, e2.event_id)

    def test_same_entity_two_different_zones_tracked_independently(self):
        engine = EventEngine()
        ea = engine.update(
            [change(ChangeType.ZONE_ENTERED, entity_id="e1", zone_id="zA")], timestamp=0.0
        )[0]
        eb = engine.update(
            [change(ChangeType.ZONE_ENTERED, entity_id="e1", zone_id="zB")], timestamp=0.0
        )[0]
        self.assertNotEqual(ea.event_id, eb.event_id)

        # exiting zA must not resolve zB's event
        engine.update([change(ChangeType.ZONE_EXITED, entity_id="e1", zone_id="zA")], timestamp=1.0)
        self.assertEqual(engine.get_event(eb.event_id).state, EventState.ACTIVE)
        self.assertEqual(engine.get_event(ea.event_id).state, EventState.RESOLVED)


class TestArchival(unittest.TestCase):
    def test_resolved_event_archives_after_configured_delay(self):
        cfg = EventConfig(archive_after_seconds=10.0)
        engine = EventEngine(cfg)
        engine.update([change(ChangeType.ZONE_ENTERED, zone_id="z1")], timestamp=0.0)
        engine.update([change(ChangeType.ZONE_EXITED, zone_id="z1")], timestamp=5.0)

        events = engine.list_events(event_type=EventType.ZONE_ENTRY)
        self.assertEqual(events[0].state, EventState.RESOLVED)

        # not yet past the archive delay (5s resolved + 9s < 10s threshold)
        engine.update([], timestamp=14.0)
        self.assertEqual(events[0].state, EventState.RESOLVED)

        # now past it (resolved at t=5, archive_after=10 -> archives at t>=15)
        engine.update([], timestamp=16.0)
        self.assertEqual(events[0].state, EventState.ARCHIVED)

    def test_active_events_are_never_auto_archived(self):
        cfg = EventConfig(archive_after_seconds=1.0)
        engine = EventEngine(cfg)
        events = engine.update([change(ChangeType.ZONE_ENTERED, zone_id="z1")], timestamp=0.0)
        engine.update([], timestamp=1000.0)   # far beyond archive_after_seconds
        self.assertEqual(engine.get_event(events[0].event_id).state, EventState.ACTIVE)


class TestQueries(unittest.TestCase):
    def test_list_events_filters_by_state_type_and_entity(self):
        engine = EventEngine()
        engine.update([change(ChangeType.APPEARED, entity_id="e1")], timestamp=0.0)
        engine.update([change(ChangeType.APPEARED, entity_id="e2")], timestamp=0.0)
        engine.update([change(ChangeType.ZONE_ENTERED, entity_id="e1", zone_id="z1")], timestamp=1.0)

        self.assertEqual(len(engine.list_events(event_type=EventType.APPEARED)), 2)
        self.assertEqual(len(engine.list_events(entity_id="e1")), 2)   # APPEARED + ZONE_ENTRY
        self.assertEqual(len(engine.list_events(state=EventState.ACTIVE)), 1)

    def test_get_event_unknown_id_raises_keyerror(self):
        engine = EventEngine()
        with self.assertRaises(KeyError):
            engine.get_event("nope")


if __name__ == "__main__":
    unittest.main()
