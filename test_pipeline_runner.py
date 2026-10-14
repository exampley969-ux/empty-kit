"""
Integration tests for PipelineRunner using the REAL wiring
(build_pipeline_runner), real SQLite, real evidence files - with a scripted
camera and detector standing in for hardware and the neural network.
"""
import os
import shutil
import tempfile
import time
import unittest
from typing import List

import cv2

from vision_radar.data_layer.database import Database
from vision_radar.data_layer.repository import EventQuery, Repository
from vision_radar.event_engine.models import EventConfig, EventState, EventType
from vision_radar.ingestion.models import VideoSourceConfig
from vision_radar.ingestion.session import IngestionSession
from vision_radar.monitoring.health import ComponentStatus, HealthRegistry
from vision_radar.monitoring.performance import PerformanceMonitor
from vision_radar.pipeline.runner import build_pipeline_runner
from vision_radar.rules_engine.models import FieldCondition, Rule, RuleTrigger
from vision_radar.tests.fakes import FakeVideoSource
from vision_radar.tracking.models import TrackerConfig
from vision_radar.vision.detector import Detector
from vision_radar.vision.models import BoundingBox, Detection
from vision_radar.world_model.zone import Zone

W, H = 320, 240


def person(x, y=80, w=40, h=80, conf=0.9):
    return Detection("person", BoundingBox(x, y, w, h), conf, "scripted")


class ScriptedDetector(Detector):
    """Returns script[i] on the i-th call; can be told to blow up on chosen calls."""
    def __init__(self, script: List[List[Detection]], fail_on=()):
        self.script, self.fail_on, self.calls = script, set(fail_on), 0

    @property
    def name(self):
        return "scripted"

    def detect(self, image):
        i = self.calls
        self.calls += 1
        if i in self.fail_on:
            raise RuntimeError("model exploded")
        return list(self.script[i]) if i < len(self.script) else []


class DyingSource(FakeVideoSource):
    """Delivers `die_after` frames, then the 'camera is unplugged' for good."""
    def __init__(self, config, die_after=3):
        super().__init__(config, total_frames=10_000, width=W, height=H)
        self.die_after, self._dead = die_after, False

    def _open(self):
        if self._dead:
            raise ConnectionError("camera unplugged")
        super()._open()

    def _read_raw(self):
        if self._frames_emitted >= self.die_after:
            self._dead, self._is_open = True, False
            return False, None
        return super()._read_raw()

    def is_exhausted(self):
        return False


def walk_script(present=6, absent=4, x0=10, step=5):
    """A person visible for `present` frames, moving right, then gone."""
    return [[person(x0 + i * step)] for i in range(present)] + [[] for _ in range(absent)]


class RunnerCase(unittest.TestCase):
    TRACKER = TrackerConfig(confirm_hits=2, tracked_hits=3, max_missed_frames=2)

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.health = HealthRegistry()
        self.db = Database(":memory:", health=self.health)
        self.repo = Repository(self.db)
        self.monitor = PerformanceMonitor()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def make(self, detector, source=None, total_frames=10, zones=(), rules=(), event_config=None):
        cfg = VideoSourceConfig(source_uri="fake://0", max_reconnect_attempts=2, reconnect_backoff_seconds=0.001)
        source = source or FakeVideoSource(cfg, total_frames=total_frames, width=W, height=H)
        session = IngestionSession(source, camera_uri="fake://0")
        return build_pipeline_runner(
            session=session, detectors=[detector], repository=self.repo, monitor=self.monitor,
            health=self.health, evidence_dir=self.tmp, zones=list(zones), rules=list(rules),
            tracker_config=self.TRACKER, event_config=event_config)


class TestEndToEndPersistence(RunnerCase):
    def setUp(self):
        super().setUp()
        self.zone = Zone.from_rect("Restricted", 0, 0, 200, H)
        self.rule = Rule.new(name="Person enters restricted zone",
                             when=RuleTrigger(event_type=EventType.ZONE_ENTRY, entity_type="person",
                                              zone_id=self.zone.zone_id),
                             if_conditions=[FieldCondition("confidence", ">", 0.5)])
        self.runner = self.make(ScriptedDetector(walk_script()), zones=[self.zone], rules=[self.rule])
        self.runner.begin()
        self.runner.run()

    def events(self, event_type):
        return self.repo.query_events(EventQuery(event_type=event_type))

    def test_the_full_story_is_in_the_database(self):
        self.assertEqual(len(self.events(EventType.APPEARED)), 1)
        self.assertEqual(len(self.events(EventType.DISAPPEARED)), 1)
        zone_events = self.events(EventType.ZONE_ENTRY)
        self.assertEqual(len(zone_events), 1)

    def test_zone_entry_was_resolved_because_the_person_vanished_inside_the_zone(self):
        # Phase 9 review defect #1, exercised through the whole real stack.
        ev = self.events(EventType.ZONE_ENTRY)[0]
        self.assertEqual(ev.state, EventState.RESOLVED)
        self.assertIsNotNone(ev.end_time)
        self.assertGreaterEqual(ev.duration_seconds(), 0.0)
        self.assertEqual(self.repo.query_events(EventQuery(state=EventState.ACTIVE)), [])

    def test_exactly_one_alert_despite_the_event_being_reported_twice(self):
        # Phase 9 review defect #2: entry (ACTIVE) and exit (RESOLVED) both reach the rule engine.
        alerts = self.repo.list_alerts()
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].rule_id, self.rule.rule_id)
        self.assertEqual(alerts[0].zone_id, self.zone.zone_id)

    def test_alert_has_real_evidence_with_an_annotated_image_on_disk(self):
        alert = self.repo.list_alerts()[0]
        evidence = self.repo.get_evidence(alert.event_id)
        self.assertIsNotNone(evidence)
        self.assertTrue(os.path.exists(evidence.frame_snapshot_path))
        self.assertEqual(cv2.imread(evidence.frame_snapshot_path).shape[:2], (H, W))
        self.assertEqual(evidence.zone_geometry, self.zone.polygon)
        self.assertIn("position", evidence.entity_location)

    def test_session_row_reflects_the_finished_run(self):
        row = self.repo.get_session(self.runner.session_id)
        self.assertEqual(row["state"], "STOPPED")
        self.assertEqual(row["frame_count"], 10)
        self.assertIsNotNone(row["ended_at"])

    def test_entity_was_persisted(self):
        rows = self.repo.list_entities(self.runner.session_id)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["entity_type"], "person")

    def test_all_component_health_is_ok_after_a_clean_run(self):
        for component in ("camera", "vision", "tracking", "events", "database"):
            self.assertEqual(self.health.get(component).status, ComponentStatus.OK, component)

    def test_disappeared_event_carries_the_entitys_last_real_confidence_not_a_default(self):
        ev = self.events(EventType.DISAPPEARED)[0]
        self.assertAlmostEqual(ev.confidence, 0.9)      # not EventConfig.default_confidence (0.5)

    def test_metrics_were_recorded(self):
        s = self.monitor.snapshot()
        self.assertEqual(s["events_per_min"], self.repo.count("events"))
        self.assertIsNotNone(s["processing_ms_avg"])

    def test_exited_tracks_were_purged_from_memory(self):
        self.assertEqual(len(self.runner.tracker._tracks), 0)


class TestFailuresAreVisibleAndSurvivable(RunnerCase):
    def test_vision_crash_is_flagged_skipped_counted_and_recovers(self):
        runner = self.make(ScriptedDetector(walk_script(), fail_on={1}))
        runner.begin()
        f0, f1, f2 = (runner.session.next_frame() for _ in range(3))

        self.assertFalse(runner.process_frame(f0).skipped)
        self.assertEqual(self.health.get("vision").status, ComponentStatus.OK)

        r1 = runner.process_frame(f1)                                   # detector raises
        self.assertTrue(r1.skipped)
        h = self.health.get("vision")
        self.assertEqual((h.status, h.message), (ComponentStatus.ERROR, "VISION ENGINE ERROR"))
        self.assertIn("model exploded", h.detail)

        self.assertFalse(runner.process_frame(f2).skipped)              # next frame works
        self.assertEqual(self.health.get("vision").status, ComponentStatus.OK)
        self.assertEqual((runner.frames_processed, runner.frames_failed), (2, 1))
        runner.stop()

    def test_camera_that_cannot_connect_is_flagged_offline_and_the_session_records_it(self):
        cfg = VideoSourceConfig(source_uri="fake://0", max_reconnect_attempts=1, reconnect_backoff_seconds=0.001)
        runner = self.make(ScriptedDetector([]), source=FakeVideoSource(cfg, fail_at_open_times=99))
        from vision_radar.ingestion.exceptions import CameraConnectionError
        with self.assertRaises(CameraConnectionError):
            runner.begin()
        h = self.health.get("camera")
        self.assertEqual((h.status, h.message), (ComponentStatus.ERROR, "CAMERA OFFLINE"))
        self.assertEqual(self.repo.get_session(runner.session_id)["state"], "ERROR")

    def test_camera_dying_mid_run_is_detected(self):
        cfg = VideoSourceConfig(source_uri="fake://0", max_reconnect_attempts=2, reconnect_backoff_seconds=0.001)
        runner = self.make(ScriptedDetector(walk_script(present=10, absent=0)), source=DyingSource(cfg, die_after=3))
        runner.begin()
        runner.run()

        self.assertEqual(runner.frames_processed, 3)
        h = self.health.get("camera")
        self.assertEqual((h.status, h.message), (ComponentStatus.ERROR, "CAMERA OFFLINE"))
        self.assertEqual(self.repo.get_session(runner.session_id)["state"], "ERROR")

    def test_database_going_away_is_flagged_and_processing_continues(self):
        runner = self.make(ScriptedDetector(walk_script()))
        runner.begin()
        self.db.close()                                                  # storage vanishes mid-run
        runner.run()

        self.assertEqual(runner.frames_processed, 10)                    # the feed was not dropped
        self.assertGreater(runner.persist_failures, 0)                   # ...and the loss was counted
        h = self.health.get("database")
        self.assertEqual((h.status, h.message), (ComponentStatus.ERROR, "DATA STORAGE UNAVAILABLE"))

    def test_nothing_is_purged_from_memory_while_persistence_is_failing(self):
        runner = self.make(ScriptedDetector(walk_script()), event_config=EventConfig(archive_after_seconds=0.001))
        runner.begin()
        self.db.close()
        runner.run()
        # events that could not be saved must still be held, not thrown away
        self.assertGreater(len(runner.event_engine._events), 0)


class ClockedSource(FakeVideoSource):
    """Stamps frame N with t = N seconds, so lifecycle timing (archive delays)
    is deterministic instead of depending on how fast this machine runs the
    loop (an earlier version used the real clock and was flaky for exactly
    that reason)."""
    def read_frame(self, session_id, frame_id):
        frame = super().read_frame(session_id, frame_id)
        if frame is not None:
            frame.captured_at = float(frame_id)
        return frame


class TestLongRunMemoryIsBounded(RunnerCase):
    def _run(self, cycles):
        script = []
        for _ in range(cycles):
            script += walk_script(present=5, absent=4)
        cfg = VideoSourceConfig(source_uri="fake://0", max_reconnect_attempts=1, reconnect_backoff_seconds=0.001)
        source = ClockedSource(cfg, total_frames=len(script), width=W, height=H)
        zone = Zone.from_rect("Z", 0, 0, 250, H)
        runner = self.make(ScriptedDetector(script), source=source, zones=[zone],
                           event_config=EventConfig(archive_after_seconds=0.5))
        runner.begin()
        runner.run()
        return runner

    def test_many_appear_disappear_cycles_do_not_accumulate_state(self):
        cycles = 15
        runner = self._run(cycles)

        self.assertEqual(runner.frames_processed, cycles * 9)
        # everything that happened is in the database...
        self.assertEqual(len(self.repo.query_events(EventQuery(event_type=EventType.APPEARED, limit=1000))), cycles)
        self.assertEqual(len(self.repo.query_events(EventQuery(event_type=EventType.ZONE_ENTRY, limit=1000))), cycles)
        # ...but the running process holds none of it.
        self.assertEqual(len(runner.tracker._tracks), 0)
        self.assertEqual(len(runner.event_engine._events), 0)
        self.assertEqual(len(runner._counted_events), 0)
        self.assertEqual(len(runner.rule_engine._fired), 0)

    def test_retained_state_does_not_depend_on_how_long_the_service_has_run(self):
        # The real claim: memory is O(1) in run length, not just "small once".
        def retained(cycles):
            self.setUp()          # fresh DB/monitor/health per run
            r = self._run(cycles)
            return (len(r.tracker._tracks), len(r.event_engine._events),
                    len(r._counted_events), len(r.rule_engine._fired), len(r._last_conf), len(r._last_entity))
        self.assertEqual(retained(3), retained(30))


class TestSessionEndClosesOpenEvents(RunnerCase):
    def _run_person_who_never_leaves(self, rule=None):
        zone = Zone.from_rect("Z", 0, 0, 250, H)
        runner = self.make(ScriptedDetector(walk_script(present=10, absent=0)), zones=[zone],
                           rules=[rule] if rule else [])
        runner.begin()
        runner.run()
        return runner, zone

    def test_no_event_is_left_active_after_the_session_ends(self):
        runner, _ = self._run_person_who_never_leaves()
        self.assertEqual(self.repo.query_events(EventQuery(state=EventState.ACTIVE)), [])
        ev = self.repo.query_events(EventQuery(event_type=EventType.ZONE_ENTRY))[0]
        self.assertEqual(ev.state, EventState.RESOLVED)
        self.assertEqual(ev.evidence["closed_reason"], "session_ended")

    def test_a_duration_gated_rule_can_still_fire_when_the_event_is_closed_at_session_end(self):
        rule = Rule.new(name="any zone entry with a duration", when=RuleTrigger(event_type=EventType.ZONE_ENTRY),
                        if_conditions=[FieldCondition("duration_seconds", ">=", 0.0)])
        self._run_person_who_never_leaves(rule)
        self.assertEqual(len(self.repo.list_alerts()), 1)

    def test_stopping_a_live_session_also_closes_open_events(self):
        zone = Zone.from_rect("Z", 0, 0, 250, H)
        runner = self.make(ScriptedDetector([[person(10 + i)] for i in range(2000)]),
                           total_frames=1_000_000, zones=[zone])
        runner.start_background()
        deadline = time.time() + 5
        while runner.frames_processed < 8 and time.time() < deadline:
            time.sleep(0.01)
        runner.stop()
        self.assertEqual(self.repo.query_events(EventQuery(state=EventState.ACTIVE)), [])


class TestBackgroundRun(RunnerCase):
    def test_start_background_then_stop_shuts_down_cleanly(self):
        runner = self.make(ScriptedDetector([]), total_frames=1_000_000)
        runner.start_background()
        deadline = time.time() + 5
        while runner.frames_processed < 5 and time.time() < deadline:
            time.sleep(0.01)
        self.assertGreaterEqual(runner.frames_processed, 5)
        self.assertTrue(runner.is_running())

        runner.stop()

        self.assertFalse(runner.is_running())
        row = self.repo.get_session(runner.session_id)
        self.assertEqual(row["state"], "STOPPED")
        self.assertGreaterEqual(row["frame_count"], 5)

    def test_stop_before_run_is_safe_and_idempotent(self):
        runner = self.make(ScriptedDetector([]))
        runner.begin()
        runner.stop()
        runner.stop()
        self.assertEqual(self.repo.get_session(runner.session_id)["state"], "STOPPED")


if __name__ == "__main__":
    unittest.main()


class TestLatestFrame(RunnerCase):
    def test_latest_frame_is_a_real_decodable_jpeg_while_running(self):
        import cv2
        import numpy as np
        runner = self.make(ScriptedDetector(walk_script()))
        runner.begin()
        f = runner.session.next_frame()
        self.assertIsNone(runner.latest_frame_jpeg())        # nothing processed yet
        runner.process_frame(f)
        jpeg = runner.latest_frame_jpeg()
        self.assertIsNotNone(jpeg)
        decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape[:2], (H, W))

    def test_latest_frame_is_replaced_not_accumulated(self):
        # FakeVideoSource emits identical blank frames, so two real frames
        # from it aren't a useful "did it change" signal - exercise the
        # replace-not-append behaviour directly instead.
        import numpy as np
        runner = self.make(ScriptedDetector(walk_script()))
        runner.begin()
        runner._update_latest_frame(np.zeros((10, 10, 3), dtype=np.uint8))
        first = runner.latest_frame_jpeg()
        runner._update_latest_frame(np.full((10, 10, 3), 255, dtype=np.uint8))
        second = runner.latest_frame_jpeg()
        self.assertNotEqual(first, second)
        self.assertIsInstance(first, bytes)   # a single value held, not a growing buffer
