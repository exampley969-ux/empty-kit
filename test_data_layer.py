import os
import shutil
import tempfile
import unittest

from vision_radar.data_layer.database import Database, DataIntegrityError, DataStorageUnavailable
from vision_radar.data_layer.repository import CameraRecord, EventQuery, Repository
from vision_radar.data_layer.retention import RetentionPolicy, apply_retention
from vision_radar.event_engine.evidence import Evidence
from vision_radar.event_engine.models import Event, EventState, EventType
from vision_radar.ingestion.models import SessionInfo, SessionState
from vision_radar.monitoring.health import ComponentStatus, HealthRegistry
from vision_radar.rules_engine.models import Alert, FieldCondition, Rule, RuleTrigger
from vision_radar.tests.test_state_engine import make_entity
from vision_radar.world_model.zone import Zone

DAY = 86400.0


def new_repo(health=None):
    return Repository(Database(":memory:", health=health))


def make_event(event_type=EventType.ZONE_ENTRY, entity_ids=("e1",), zone_id="z1", start=100.0,
               end=None, confidence=0.9, state=EventState.ACTIVE, entity_type="person"):
    ev = Event.new(event_type=event_type, entity_ids=list(entity_ids), entity_type=entity_type,
                   start_time=start, confidence=confidence, zone_id=zone_id)
    ev.end_time = end
    ev.state = state
    return ev


class TestDatabaseHealthAndErrors(unittest.TestCase):
    def test_healthy_database_reports_ok(self):
        health = HealthRegistry()
        Database(":memory:", health=health)
        self.assertEqual(health.get("database").status, ComponentStatus.OK)

    def test_unusable_database_path_raises_and_flags_storage_unavailable(self):
        health = HealthRegistry()
        with self.assertRaises(DataStorageUnavailable):
            Database("/nonexistent-dir-xyz/sub/db.sqlite", health=health)
        h = health.get("database")
        self.assertEqual(h.status, ComponentStatus.ERROR)
        self.assertEqual(h.message, "DATA STORAGE UNAVAILABLE")

    def test_failure_after_startup_flags_error_and_recovery_flags_ok(self):
        health = HealthRegistry()
        db = Database(":memory:", health=health)
        with self.assertRaises(DataStorageUnavailable):
            db.query_all("SELECT * FROM table_that_does_not_exist")
        self.assertEqual(health.get("database").status, ComponentStatus.ERROR)
        db.query_all("SELECT 1")
        self.assertEqual(health.get("database").status, ComponentStatus.OK)

    def test_integrity_error_is_not_reported_as_storage_unavailable(self):
        health = HealthRegistry()
        repo = new_repo(health)
        with self.assertRaises(DataIntegrityError):
            repo.save_event(make_event(), session_id="no-such-session")   # FK violation
        self.assertEqual(health.get("database").status, ComponentStatus.OK)

    def test_failed_multi_statement_write_is_rolled_back_completely(self):
        repo = new_repo()
        with self.assertRaises(DataIntegrityError):
            repo.save_event(make_event(), session_id="no-such-session")
        self.assertEqual(repo.count("events"), 0)
        self.assertEqual(repo.db.query_one("SELECT COUNT(*) n FROM event_entities")["n"], 0)


class TestCamerasZonesRulesState(unittest.TestCase):
    def test_camera_roundtrip_and_delete(self):
        repo = new_repo()
        repo.add_camera(CameraRecord("c1", "Front door", "0", 640, 480))
        self.assertEqual(repo.get_camera("c1").name, "Front door")
        self.assertEqual(len(repo.list_cameras()), 1)
        self.assertTrue(repo.delete_camera("c1"))
        self.assertIsNone(repo.get_camera("c1"))
        self.assertFalse(repo.delete_camera("c1"))

    def test_zone_roundtrip_preserves_polygon_as_tuples(self):
        repo = new_repo()
        zone = Zone.from_polygon("Wedge", [(0, 0), (100, 0), (50, 100)])
        repo.save_zone(zone)
        loaded = repo.get_zone(zone.zone_id)
        self.assertEqual(loaded.polygon, [(0, 0), (100, 0), (50, 100)])
        self.assertTrue(loaded.contains((50, 30)))          # still geometrically usable

    def test_zone_update_via_save_is_an_upsert(self):
        repo = new_repo()
        zone = Zone.from_rect("A", 0, 0, 10, 10)
        repo.save_zone(zone)
        zone.name = "Renamed"
        repo.save_zone(zone)
        self.assertEqual(repo.count("zones"), 1)
        self.assertEqual(repo.get_zone(zone.zone_id).name, "Renamed")

    def test_deleting_a_camera_deletes_its_zones(self):
        repo = new_repo()
        repo.add_camera(CameraRecord("c1", "Cam", "0"))
        repo.save_zone(Zone.from_rect("A", 0, 0, 10, 10), camera_id="c1")
        repo.delete_camera("c1")
        self.assertEqual(repo.count("zones"), 0)

    def test_rule_roundtrip_is_behaviourally_identical(self):
        repo = new_repo()
        rule = Rule.new(
            name="Vehicle dwell",
            when=RuleTrigger(event_type=EventType.DWELL, entity_type="car", zone_id="zB"),
            if_conditions=[FieldCondition("evidence.dwell_seconds", ">", 30.0),
                           FieldCondition("confidence", ">=", 0.6)])
        repo.save_rule(rule)
        loaded = repo.get_rule(rule.rule_id)

        hit = make_event(EventType.DWELL, zone_id="zB", entity_type="car", confidence=0.7)
        hit.evidence = {"dwell_seconds": 45.0}
        miss = make_event(EventType.DWELL, zone_id="zB", entity_type="car", confidence=0.7)
        miss.evidence = {"dwell_seconds": 5.0}
        self.assertTrue(loaded.matches(hit))
        self.assertFalse(loaded.matches(miss))
        self.assertEqual(loaded.to_definition(), rule.to_definition())

    def test_system_state_roundtrip_and_default(self):
        repo = new_repo()
        self.assertEqual(repo.get_state("missing", default=42), 42)
        repo.set_state("last_retention_run", {"at": 123.0, "deleted": 4})
        self.assertEqual(repo.get_state("last_retention_run"), {"at": 123.0, "deleted": 4})
        repo.set_state("last_retention_run", {"at": 999.0})
        self.assertEqual(repo.get_state("last_retention_run"), {"at": 999.0})


class TestSessionsAndEntities(unittest.TestCase):
    def test_session_persists_and_updates(self):
        repo = new_repo()
        info = SessionInfo(camera_uri="0", state=SessionState.RUNNING, started_at=10.0)
        repo.save_session(info)
        info.state, info.ended_at, info.frame_count = SessionState.STOPPED, 20.0, 55
        repo.save_session(info)
        row = repo.get_session(info.session_id)
        self.assertEqual((row["state"], row["frame_count"], row["ended_at"]), ("STOPPED", 55, 20.0))
        self.assertEqual(len(repo.list_sessions()), 1)

    def test_entity_upsert_keeps_first_seen_updates_last(self):
        repo = new_repo()
        repo.save_session(SessionInfo(session_id="s1", camera_uri="0", state=SessionState.RUNNING))
        e = make_entity("e1", 10, 20)
        e.first_seen, e.last_seen = 1.0, 1.0
        repo.upsert_entity(e, "s1")
        e2 = make_entity("e1", 50, 60)
        e2.first_seen, e2.last_seen = 999.0, 9.0      # a later snapshot must not rewrite first_seen
        repo.upsert_entity(e2, "s1")
        rows = repo.list_entities("s1")
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["first_seen"], rows[0]["last_seen"], rows[0]["last_x"]), (1.0, 9.0, 50))


class TestEvents(unittest.TestCase):
    def test_event_roundtrip_including_entity_ids_and_evidence(self):
        repo = new_repo()
        ev = make_event(entity_ids=("e1", "e2"))
        ev.evidence = {"distance": 42.5, "from": (1, 2)}
        repo.save_event(ev)
        loaded = repo.get_event(ev.event_id)
        self.assertEqual(loaded.event_type, EventType.ZONE_ENTRY)
        self.assertEqual(sorted(loaded.entity_ids), ["e1", "e2"])
        self.assertEqual(loaded.evidence["distance"], 42.5)
        self.assertEqual(loaded.state, EventState.ACTIVE)
        self.assertIsNone(repo.get_event("missing"))

    def test_lifecycle_updates_overwrite_same_row(self):
        repo = new_repo()
        ev = make_event()
        repo.save_event(ev)
        ev.end_time, ev.state = 118.0, EventState.RESOLVED
        repo.save_event(ev)
        ev.state = EventState.ARCHIVED
        repo.save_event(ev)
        self.assertEqual(repo.count("events"), 1)
        loaded = repo.get_event(ev.event_id)
        self.assertEqual((loaded.state, loaded.end_time), (EventState.ARCHIVED, 118.0))

    def test_numpy_scalars_in_evidence_are_stored_not_rejected(self):
        import numpy as np
        repo = new_repo()
        ev = make_event()
        ev.evidence = {"distance": np.float32(3.5)}
        repo.save_event(ev)
        self.assertAlmostEqual(repo.get_event(ev.event_id).evidence["distance"], 3.5)

    def test_unserialisable_evidence_is_rejected_loudly(self):
        repo = new_repo()
        ev = make_event()
        ev.evidence = {"bad": object()}
        with self.assertRaises(TypeError):
            repo.save_event(ev)


class TestEventQueryFilters(unittest.TestCase):
    def setUp(self):
        self.repo = new_repo()
        self.a = make_event(EventType.ZONE_ENTRY, ("e1",), "zA", start=100, end=110, confidence=0.9, state=EventState.RESOLVED)
        self.b = make_event(EventType.APPEARED, ("e2",), None, start=200, end=200, confidence=0.6, state=EventState.RESOLVED)
        self.c = make_event(EventType.ZONE_ENTRY, ("e2",), "zB", start=300, confidence=0.8, state=EventState.ACTIVE)
        for e in (self.a, self.b, self.c):
            self.repo.save_event(e)

    def ids(self, **kw):
        return {e.event_id for e in self.repo.query_events(EventQuery(**kw))}

    def test_no_filters_returns_all_newest_first(self):
        result = self.repo.query_events(EventQuery())
        self.assertEqual([e.event_id for e in result], [self.c.event_id, self.b.event_id, self.a.event_id])

    def test_each_filter_from_section_18(self):
        self.assertEqual(self.ids(start_time_from=150, start_time_to=250), {self.b.event_id})
        self.assertEqual(self.ids(event_type=EventType.ZONE_ENTRY), {self.a.event_id, self.c.event_id})
        self.assertEqual(self.ids(entity_id="e2"), {self.b.event_id, self.c.event_id})
        self.assertEqual(self.ids(zone_id="zA"), {self.a.event_id})
        self.assertEqual(self.ids(min_confidence=0.85), {self.a.event_id})
        self.assertEqual(self.ids(state=EventState.ACTIVE), {self.c.event_id})

    def test_filters_combine_with_and(self):
        self.assertEqual(self.ids(event_type=EventType.ZONE_ENTRY, entity_id="e2"), {self.c.event_id})
        self.assertEqual(self.ids(event_type=EventType.APPEARED, zone_id="zA"), set())

    def test_the_spec_example_zone_entries_between_two_times(self):
        # Section 18: "Show all zone-entry events between 2 PM and 4 PM."
        self.assertEqual(self.ids(event_type=EventType.ZONE_ENTRY, start_time_from=50, start_time_to=150),
                         {self.a.event_id})

    def test_limit_and_offset_paginate(self):
        page1 = self.repo.query_events(EventQuery(limit=2))
        page2 = self.repo.query_events(EventQuery(limit=2, offset=2))
        self.assertEqual(len(page1), 2)
        self.assertEqual(len(page2), 1)
        self.assertEqual(len({e.event_id for e in page1 + page2}), 3)

    def test_event_with_two_entities_is_returned_once_not_twice(self):
        multi = make_event(entity_ids=("e7", "e8"), start=400)
        self.repo.save_event(multi)
        result = self.repo.query_events(EventQuery(entity_id="e7"))
        self.assertEqual([e.event_id for e in result], [multi.event_id])

    def test_hostile_filter_values_cannot_inject_sql(self):
        evil = "e1' OR '1'='1"
        self.assertEqual(self.ids(entity_id=evil), set())
        self.assertEqual(self.ids(zone_id="zA'; DROP TABLE events; --"), set())
        self.assertEqual(self.repo.count("events"), 3)     # table still intact

    def test_entity_filter_is_an_exact_match_not_a_pattern(self):
        self.assertEqual(self.ids(entity_id="e%"), set())
        self.assertEqual(self.ids(entity_id="e_"), set())


class TestAlertsAndEvidence(unittest.TestCase):
    def test_alert_roundtrip_via_join(self):
        repo = new_repo()
        ev = make_event(entity_ids=("e1",), zone_id="z9")
        repo.save_event(ev)
        rule = Rule.new(name="R", when=RuleTrigger(event_type=EventType.ZONE_ENTRY))
        alert = Alert.from_match(rule, ev)
        repo.save_alert(alert)
        repo.save_alert(alert)                      # saving twice must not duplicate
        alerts = repo.list_alerts()
        self.assertEqual(len(alerts), 1)
        self.assertEqual((alerts[0].event_id, alerts[0].zone_id, alerts[0].entity_ids),
                         (ev.event_id, "z9", ["e1"]))
        self.assertEqual(alerts[0].event_type, EventType.ZONE_ENTRY)

    def test_alert_for_unknown_event_is_rejected(self):
        repo = new_repo()
        ev = make_event()
        alert = Alert.from_match(Rule.new(name="R", when=RuleTrigger(event_type=EventType.ZONE_ENTRY)), ev)
        with self.assertRaises(DataIntegrityError):
            repo.save_alert(alert)                  # event was never saved

    def test_evidence_roundtrip(self):
        repo = new_repo()
        ev = make_event()
        repo.save_event(ev)
        repo.save_evidence(Evidence(ev.event_id, 105.0, 0.9,
                                    entity_location={"position": (1.0, 2.0), "bbox": (0, 0, 5, 5)},
                                    zone_geometry=[(0, 0), (10, 0), (10, 10)],
                                    frame_snapshot_path="/x/y.jpg"))
        got = repo.get_evidence(ev.event_id)
        self.assertEqual(got.zone_geometry, [(0, 0), (10, 0), (10, 10)])
        self.assertEqual(got.frame_snapshot_path, "/x/y.jpg")
        self.assertIsNone(repo.get_evidence("nope"))

    def test_deleting_an_event_cascades_to_alerts_and_evidence(self):
        repo = new_repo()
        ev = make_event()
        repo.save_event(ev)
        repo.save_alert(Alert.from_match(Rule.new(name="R", when=RuleTrigger(event_type=EventType.ZONE_ENTRY)), ev))
        repo.save_evidence(Evidence(ev.event_id, 1.0, 0.9))
        repo.delete_events([ev.event_id])
        self.assertEqual((repo.count("alerts"), repo.count("evidence"), repo.count("events")), (0, 0, 0))


class TestRetention(unittest.TestCase):
    NOW = 100 * DAY

    def setUp(self):
        self.repo = new_repo()
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _image(self, name="e.jpg"):
        p = os.path.join(self.tmp, name)
        with open(p, "wb") as f:
            f.write(b"jpeg")
        return p

    def _evented(self, age_days, state, with_image=True, name="e.jpg"):
        t = self.NOW - age_days * DAY
        ev = make_event(start=t, end=t, state=state)
        self.repo.save_event(ev)
        path = self._image(name) if with_image else None
        self.repo.save_evidence(Evidence(ev.event_id, t, 0.9, frame_snapshot_path=path))
        return ev, path

    def test_invalid_policy_is_rejected(self):
        with self.assertRaises(ValueError):
            apply_retention(self.repo, RetentionPolicy(event_retention_days=0), self.tmp, now=self.NOW)

    def test_old_image_is_deleted_but_structured_evidence_and_event_remain(self):
        ev, path = self._evented(10, EventState.RESOLVED)      # 10d old: image (7d) expired, event (30d) not
        report = apply_retention(self.repo, RetentionPolicy(30, 7), self.tmp, now=self.NOW)
        self.assertEqual(report.evidence_images_deleted, 1)
        self.assertFalse(os.path.exists(path))
        self.assertIsNotNone(self.repo.get_event(ev.event_id))
        self.assertIsNone(self.repo.get_evidence(ev.event_id).frame_snapshot_path)

    def test_recent_image_and_event_are_kept(self):
        ev, path = self._evented(1, EventState.RESOLVED)
        report = apply_retention(self.repo, RetentionPolicy(30, 7), self.tmp, now=self.NOW)
        self.assertEqual((report.events_deleted, report.evidence_images_deleted), (0, 0))
        self.assertTrue(os.path.exists(path))

    def test_old_finished_event_is_deleted_with_its_image_and_alerts(self):
        ev, path = self._evented(40, EventState.ARCHIVED)
        self.repo.save_alert(Alert.from_match(Rule.new(name="R", when=RuleTrigger(event_type=EventType.ZONE_ENTRY)), ev))
        report = apply_retention(self.repo, RetentionPolicy(30, 7), self.tmp, now=self.NOW)
        self.assertEqual(report.events_deleted, 1)
        self.assertFalse(os.path.exists(path))
        self.assertEqual((self.repo.count("events"), self.repo.count("alerts"), self.repo.count("evidence")), (0, 0, 0))

    def test_active_events_are_never_deleted_however_old(self):
        ev, _ = self._evented(500, EventState.ACTIVE)
        report = apply_retention(self.repo, RetentionPolicy(30, 7), self.tmp, now=self.NOW)
        self.assertEqual(report.events_deleted, 0)
        self.assertIsNotNone(self.repo.get_event(ev.event_id))

    def test_retention_never_deletes_files_outside_the_evidence_directory(self):
        outside_dir = tempfile.mkdtemp()
        try:
            victim = os.path.join(outside_dir, "precious.txt")
            with open(victim, "w") as f:
                f.write("do not delete")
            ev = make_event(start=self.NOW - 50 * DAY, end=self.NOW - 50 * DAY, state=EventState.RESOLVED)
            self.repo.save_event(ev)
            self.repo.save_evidence(Evidence(ev.event_id, self.NOW - 50 * DAY, 0.9, frame_snapshot_path=victim))
            report = apply_retention(self.repo, RetentionPolicy(30, 7), self.tmp, now=self.NOW)
            self.assertTrue(os.path.exists(victim))
            self.assertGreaterEqual(report.images_skipped_outside_evidence_dir, 1)
        finally:
            shutil.rmtree(outside_dir, ignore_errors=True)

    def test_path_traversal_in_stored_path_is_refused(self):
        sneaky = os.path.join(self.tmp, "..", "escape.txt")
        outside = os.path.realpath(sneaky)
        with open(outside, "w") as f:
            f.write("x")
        try:
            ev = make_event(start=self.NOW - 50 * DAY, end=self.NOW - 50 * DAY, state=EventState.RESOLVED)
            self.repo.save_event(ev)
            self.repo.save_evidence(Evidence(ev.event_id, self.NOW - 50 * DAY, 0.9, frame_snapshot_path=sneaky))
            apply_retention(self.repo, RetentionPolicy(30, 7), self.tmp, now=self.NOW)
            self.assertTrue(os.path.exists(outside))
        finally:
            os.remove(outside)

    def test_missing_image_file_does_not_crash_retention(self):
        ev, path = self._evented(10, EventState.RESOLVED)
        os.remove(path)
        report = apply_retention(self.repo, RetentionPolicy(30, 7), self.tmp, now=self.NOW)
        self.assertEqual(report.evidence_images_deleted, 0)


class TestUsers(unittest.TestCase):
    def test_user_storage_and_uniqueness(self):
        repo = new_repo()
        repo.add_user("u1", "alice", "hash", "viewer")
        self.assertEqual(repo.get_user_by_username("alice")["user_id"], "u1")
        self.assertEqual(repo.user_count(), 1)
        with self.assertRaises(DataIntegrityError):
            repo.add_user("u2", "alice", "hash2", "viewer")   # duplicate username
        self.assertNotIn("password_hash", repo.list_users()[0])


if __name__ == "__main__":
    unittest.main()


class TestClosedDatabaseIsReportedNotCrashed(unittest.TestCase):
    """Found by the pipeline runner's tests: rollback() inside the error
    handler itself raised on a closed connection, leaking a raw sqlite3 error."""

    def test_every_operation_on_a_closed_database_raises_datastorageunavailable(self):
        health = HealthRegistry()
        db = Database(":memory:", health=health)
        db.close()
        for op in (lambda: db.execute("SELECT 1"),
                   lambda: db.query_all("SELECT 1"),
                   lambda: db.query_one("SELECT 1"),
                   lambda: db.transaction([("SELECT 1", ())])):
            with self.assertRaises(DataStorageUnavailable):
                op()
        self.assertEqual(health.get("database").message, "DATA STORAGE UNAVAILABLE")


class TestEventQueryEntityTypeFilter(unittest.TestCase):
    """entity_type added for Phase 10's natural-language search (Section 19
    needs to filter e.g. 'person' events without guessing one entity_id)."""

    def test_entity_type_filters_correctly(self):
        repo = new_repo()
        person = make_event(entity_type="person", start=1.0)
        car = make_event(entity_type="car", start=2.0)
        for e in (person, car):
            repo.save_event(e)
        self.assertEqual([e.event_id for e in repo.query_events(EventQuery(entity_type="person"))],
                         [person.event_id])
        self.assertEqual([e.event_id for e in repo.query_events(EventQuery(entity_type="car"))],
                         [car.event_id])

    def test_entity_type_combines_with_other_filters(self):
        repo = new_repo()
        a = make_event(entity_type="person", event_type=EventType.ZONE_ENTRY, start=1.0)
        b = make_event(entity_type="person", event_type=EventType.APPEARED, start=2.0)
        for e in (a, b):
            repo.save_event(e)
        got = repo.query_events(EventQuery(entity_type="person", event_type=EventType.ZONE_ENTRY))
        self.assertEqual([e.event_id for e in got], [a.event_id])
