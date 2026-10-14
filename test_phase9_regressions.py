"""
Regression tests for four latent defects found while reviewing Phases 3-8
as a long-running backend service (Phase 9). None of them showed up in the
earlier short, fixed test sequences, because those sequences never had an
entity leave the scene while inside a zone, never touched the same event
twice, and never ran long enough for memory growth to matter.
"""
import unittest

from vision_radar.event_engine.engine import EventEngine
from vision_radar.event_engine.models import EventConfig, EventState, EventType
from vision_radar.rules_engine.engine import RuleEngine
from vision_radar.rules_engine.models import FieldCondition, Rule, RuleTrigger
from vision_radar.state_engine.engine import StateChangeEngine
from vision_radar.state_engine.models import ChangeType
from vision_radar.tracking.models import TrackerConfig
from vision_radar.tracking.tracker import IoUTracker
from vision_radar.tests.test_state_engine import make_entity, occ_map
from vision_radar.vision.models import BoundingBox, Detection


class TestDisappearInsideZoneResolvesTheEvent(unittest.TestCase):
    """Defect 1: an entity vanishing while inside a zone left its ZONE_ENTRY
    event ACTIVE forever."""

    def test_state_engine_emits_zone_exited_before_disappeared(self):
        engine = StateChangeEngine()
        e = make_entity("e1", 10, 10)
        engine.update([e], zone_occupancy=occ_map("z1", "A", ["e1"]), timestamp=0.0)

        changes = engine.update([], zone_occupancy=occ_map("z1", "A", []), timestamp=1.0)
        types = [c.change_type for c in changes]
        self.assertEqual(types, [ChangeType.ZONE_EXITED, ChangeType.DISAPPEARED])
        self.assertEqual(changes[0].zone_id, "z1")
        self.assertEqual(changes[0].extra["reason"], "entity_disappeared")

    def test_no_zone_exited_when_entity_was_not_in_a_zone(self):
        engine = StateChangeEngine()
        engine.update([make_entity("e1", 10, 10)], zone_occupancy=occ_map("z1", "A", []), timestamp=0.0)
        changes = engine.update([], zone_occupancy=occ_map("z1", "A", []), timestamp=1.0)
        self.assertEqual([c.change_type for c in changes], [ChangeType.DISAPPEARED])

    def test_full_chain_zone_entry_event_resolves_when_entity_vanishes(self):
        state_engine = StateChangeEngine()
        event_engine = EventEngine()
        e = make_entity("e1", 10, 10)

        changes = state_engine.update([e], zone_occupancy=occ_map("z1", "A", ["e1"]), timestamp=0.0)
        event_engine.update(changes, timestamp=0.0)
        self.assertEqual(len(event_engine.list_events(state=EventState.ACTIVE)), 1)

        changes = state_engine.update([], zone_occupancy=occ_map("z1", "A", []), timestamp=4.0)
        event_engine.update(changes, timestamp=4.0)

        self.assertEqual(event_engine.list_events(state=EventState.ACTIVE), [])
        zone_event = event_engine.list_events(event_type=EventType.ZONE_ENTRY)[0]
        self.assertEqual(zone_event.state, EventState.RESOLVED)
        self.assertAlmostEqual(zone_event.duration_seconds(), 4.0)


class TestRuleDeduplication(unittest.TestCase):
    """Defect 2: the same event is reported at ACTIVE, RESOLVED and ARCHIVED;
    a matching rule used to raise an alert each time."""

    def _zone_event_engine(self):
        state_engine = StateChangeEngine()
        event_engine = EventEngine(EventConfig(archive_after_seconds=5.0))
        return state_engine, event_engine

    def test_same_event_reported_twice_alerts_once(self):
        rules = RuleEngine()
        rules.add_rule(Rule.new(name="any zone entry", when=RuleTrigger(event_type=EventType.ZONE_ENTRY)))
        state_engine, event_engine = self._zone_event_engine()
        e = make_entity("e1", 10, 10)

        ev1 = event_engine.update(
            state_engine.update([e], zone_occupancy=occ_map("z1", "A", ["e1"]), timestamp=0.0), 0.0)
        ev2 = event_engine.update(
            state_engine.update([e], zone_occupancy=occ_map("z1", "A", []), timestamp=3.0), 3.0)

        self.assertEqual(len(rules.evaluate(ev1)), 1)   # at entry
        self.assertEqual(len(rules.evaluate(ev2)), 0)   # same event at exit: no repeat

    def test_duration_gated_rule_fires_once_when_event_resolves(self):
        rules = RuleEngine()
        rules.add_rule(Rule.new(
            name="long stay", when=RuleTrigger(event_type=EventType.ZONE_ENTRY),
            if_conditions=[FieldCondition("duration_seconds", ">", 2.0)]))
        state_engine, event_engine = self._zone_event_engine()
        e = make_entity("e1", 10, 10)

        ev1 = event_engine.update(
            state_engine.update([e], zone_occupancy=occ_map("z1", "A", ["e1"]), timestamp=0.0), 0.0)
        self.assertEqual(rules.evaluate(ev1), [])          # still ACTIVE: no duration yet

        ev2 = event_engine.update(
            state_engine.update([e], zone_occupancy=occ_map("z1", "A", []), timestamp=3.0), 3.0)
        self.assertEqual(len(rules.evaluate(ev2)), 1)      # RESOLVED, 3s > 2s
        self.assertEqual(rules.evaluate(ev2), [])          # re-report: still just once

    def test_archived_report_never_alerts_and_clears_dedup_memory(self):
        rules = RuleEngine()
        rules.add_rule(Rule.new(name="any", when=RuleTrigger(event_type=EventType.ZONE_ENTRY)))
        state_engine, event_engine = self._zone_event_engine()
        e = make_entity("e1", 10, 10)

        rules.evaluate(event_engine.update(
            state_engine.update([e], zone_occupancy=occ_map("z1", "A", ["e1"]), timestamp=0.0), 0.0))
        rules.evaluate(event_engine.update(
            state_engine.update([e], zone_occupancy=occ_map("z1", "A", []), timestamp=1.0), 1.0))
        self.assertEqual(len(rules._fired), 1)

        archived = event_engine.update([], timestamp=10.0)
        # APPEARED (instant) + ZONE_ENTRY (resolved) are both archived by now
        self.assertEqual(len(archived), 2)
        self.assertTrue(all(x.state == EventState.ARCHIVED for x in archived))
        self.assertEqual(rules.evaluate(archived), [])
        self.assertEqual(len(rules._fired), 0)             # no unbounded growth

    def test_removing_a_rule_forgets_its_fired_keys(self):
        rules = RuleEngine()
        rule = Rule.new(name="any", when=RuleTrigger(event_type=EventType.ZONE_ENTRY))
        rules.add_rule(rule)
        event_engine = EventEngine()
        state_engine = StateChangeEngine()
        rules.evaluate(event_engine.update(
            state_engine.update([make_entity("e1", 1, 1)],
                                zone_occupancy=occ_map("z1", "A", ["e1"]), timestamp=0.0), 0.0))
        rules.remove_rule(rule.rule_id)
        self.assertEqual(len(rules._fired), 0)


class TestArchivedEventsAreReportedAndPurgeable(unittest.TestCase):
    """Defects 3 & 4: archival must be visible to whoever persists events,
    and archived events must be droppable from memory."""

    def test_purge_archived_removes_only_archived(self):
        engine = EventEngine(EventConfig(archive_after_seconds=1.0))
        state_engine = StateChangeEngine()
        e = make_entity("e1", 1, 1)
        engine.update(state_engine.update([e], zone_occupancy=occ_map("z1", "A", ["e1"]), timestamp=0.0), 0.0)
        engine.update(state_engine.update([e], zone_occupancy=occ_map("z1", "A", []), timestamp=1.0), 1.0)
        # an unrelated, still-ACTIVE event on another entity
        e2 = make_entity("e2", 5, 5)
        engine.update(state_engine.update([e, e2], zone_occupancy=occ_map("z1", "A", ["e2"]), timestamp=1.5), 1.5)

        engine.update([], timestamp=5.0)   # archives the resolved one
        removed = engine.purge_archived()

        self.assertGreaterEqual(removed, 1)
        self.assertEqual(engine.list_events(state=EventState.ARCHIVED), [])
        self.assertEqual(len(engine.list_events(state=EventState.ACTIVE)), 1)

    def test_event_resolved_and_archived_in_one_call_is_reported_once(self):
        engine = EventEngine(EventConfig(archive_after_seconds=0.0))
        state_engine = StateChangeEngine()
        e = make_entity("e1", 1, 1)
        engine.update(state_engine.update([e], zone_occupancy=occ_map("z1", "A", ["e1"]), timestamp=0.0), 0.0)
        touched = engine.update(
            state_engine.update([e], zone_occupancy=occ_map("z1", "A", []), timestamp=1.0), 1.0)
        ids = [t.event_id for t in touched]
        self.assertEqual(len(ids), len(set(ids)))


class TestTrackerPurge(unittest.TestCase):
    def test_purge_exited_forgets_exited_tracks_only(self):
        tracker = IoUTracker(TrackerConfig(confirm_hits=1, tracked_hits=1, max_missed_frames=0))
        det = lambda x: Detection("person", BoundingBox(x, 0, 40, 80), 0.9, "t")

        tracker.update([det(0), det(500)], timestamp=0.0)
        tracker.update([det(500)], timestamp=1.0)     # first track misses -> exits
        self.assertEqual(len(tracker.active_tracks()), 1)

        removed = tracker.purge_exited()
        self.assertEqual(removed, 1)
        self.assertEqual(len(tracker._tracks), 1)      # only the live one remains


if __name__ == "__main__":
    unittest.main()


class TestSessionEndClosesOpenEvents(unittest.TestCase):
    """Defect 5 (found reproducing the long-run scenario before the real
    end-to-end run): a session ending while an entity was inside a zone left
    its ZONE_ENTRY event ACTIVE forever."""

    def test_engine_closes_open_zone_entries_honestly(self):
        engine = EventEngine()
        state = StateChangeEngine()
        e = make_entity("e1", 1, 1)
        engine.update(state.update([e], zone_occupancy=occ_map("z1", "A", ["e1"]), timestamp=10.0), 10.0)

        closed = engine.close_open_events(timestamp=25.0, reason="session_ended")

        self.assertEqual(len(closed), 1)
        ev = closed[0]
        self.assertEqual((ev.state, ev.end_time), (EventState.RESOLVED, 25.0))
        self.assertAlmostEqual(ev.duration_seconds(), 15.0)
        self.assertEqual(ev.evidence["closed_reason"], "session_ended")
        self.assertNotIn("exited_at", ev.evidence)          # the entity did not exit; we stopped looking
        self.assertEqual(engine.list_events(state=EventState.ACTIVE), [])
        self.assertEqual(engine.close_open_events(30.0), [])   # idempotent

    def test_nothing_to_close_returns_empty(self):
        self.assertEqual(EventEngine().close_open_events(1.0), [])
