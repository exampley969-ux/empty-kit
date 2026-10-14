import os
import shutil
import tempfile
import time
import unittest

from vision_radar.api.errors import (
    CameraUnavailable, Conflict, NotFound, ValidationError, public_message, status_for,
)
from vision_radar.api.service import ApiService
from vision_radar.data_layer.database import DataIntegrityError, DataStorageUnavailable, Database
from vision_radar.data_layer.repository import EventQuery, Repository
from vision_radar.data_layer.retention import RetentionPolicy
from vision_radar.event_engine.models import Event, EventState, EventType
from vision_radar.event_engine.evidence import Evidence
from vision_radar.ingestion.models import VideoSourceConfig
from vision_radar.ingestion.session import IngestionSession
from vision_radar.monitoring.health import ComponentStatus, HealthRegistry
from vision_radar.monitoring.performance import PerformanceMonitor
from vision_radar.pipeline.runner import build_pipeline_runner
from vision_radar.security.auth import AuthenticationError, AuthService, PermissionDenied, Role
from vision_radar.tests.fakes import FakeVideoSource
from vision_radar.tests.test_pipeline_runner import H, W, ScriptedDetector, walk_script
from vision_radar.tracking.models import TrackerConfig

FAST = 1000
RECT = [[0, 0], [200, 0], [200, H], [0, H]]
SQUARE = {"when": {"event_type": "ZONE_ENTRY", "entity_type": "person"}, "if": []}


class ApiCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.health = HealthRegistry()
        self.repo = Repository(Database(":memory:", health=self.health))
        self.monitor = PerformanceMonitor()
        self.auth = AuthService(self.repo, secret_key="k" * 40, hash_iterations=FAST)
        self.script, self.frames, self.fail_open = walk_script(), 10, 0
        self.svc = ApiService(self.repo, self.auth, self.health, self.monitor, evidence_dir=self.tmp,
                              retention=RetentionPolicy(30, 7), runner_factory=self._factory)
        for name, role in (("admin", Role.ADMINISTRATOR), ("oper", Role.OPERATOR), ("view", Role.VIEWER)):
            self.auth.create_user(name, "password-123", role)
        self.admin, self.oper, self.view = (self.svc.login(n, "password-123")["token"]
                                            for n in ("admin", "oper", "view"))

    def tearDown(self):
        self.svc.shutdown()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _factory(self, camera, zones, rules):
        cfg = VideoSourceConfig(source_uri=camera.source_uri, max_reconnect_attempts=1, reconnect_backoff_seconds=0.001)
        source = FakeVideoSource(cfg, total_frames=self.frames, fail_at_open_times=self.fail_open, width=W, height=H)
        return build_pipeline_runner(
            session=IngestionSession(source, camera_uri=camera.source_uri),
            detectors=[ScriptedDetector(list(self.script))], repository=self.repo, monitor=self.monitor,
            health=self.health, evidence_dir=self.tmp, zones=zones, rules=rules, camera_id=camera.camera_id,
            tracker_config=TrackerConfig(confirm_hits=2, tracked_hits=3, max_missed_frames=2))

    def wait_until_finished(self, timeout=10):
        deadline = time.time() + timeout
        while any(r.is_running() for r in self.svc._runners.values()) and time.time() < deadline:
            time.sleep(0.01)


class TestAuthentication(ApiCase):
    def test_login_returns_a_working_token_and_role(self):
        r = self.svc.login("admin", "password-123")
        self.assertEqual((r["token_type"], r["role"], r["username"]), ("bearer", "administrator", "admin"))
        self.svc.get_status(r["token"])

    def test_bad_login_is_rejected(self):
        with self.assertRaises(AuthenticationError):
            self.svc.login("admin", "wrong-password")

    def test_every_endpoint_rejects_a_missing_or_invalid_token(self):
        calls = [
            lambda t: self.svc.get_status(t), lambda t: self.svc.list_events(t), lambda t: self.svc.get_event(t, "x"),
            lambda t: self.svc.list_alerts(t), lambda t: self.svc.list_entities(t), lambda t: self.svc.list_sessions(t),
            lambda t: self.svc.list_cameras(t), lambda t: self.svc.create_camera(t, "c", "0"),
            lambda t: self.svc.delete_camera(t, "x"), lambda t: self.svc.list_zones(t),
            lambda t: self.svc.create_zone(t, "z", RECT), lambda t: self.svc.delete_zone(t, "x"),
            lambda t: self.svc.list_rules(t), lambda t: self.svc.create_rule(t, "r", SQUARE),
            lambda t: self.svc.delete_rule(t, "x"), lambda t: self.svc.start_session(t, "x"),
            lambda t: self.svc.stop_session(t, "x"), lambda t: self.svc.create_user(t, "u", "password-123", "viewer"),
            lambda t: self.svc.list_users(t), lambda t: self.svc.run_retention(t),
            lambda t: self.svc.get_evidence_image_path(t, "x"),
        ]
        for i, call in enumerate(calls):
            for bad in (None, "", "not.a.token"):
                with self.assertRaises(AuthenticationError, msg=f"call #{i} token={bad!r}"):
                    call(bad)


class TestAuthorizationMatrix(ApiCase):
    """Section 28: administrator / operator / viewer are genuinely separated."""

    def setUp(self):
        super().setUp()
        cam = self.svc.create_camera(self.admin, "Cam", "fake://0")
        self.cam_id = cam["camera_id"]

    def outcome(self, call):
        try:
            call()
            return "allowed"
        except PermissionDenied:
            return "denied"
        except Exception:                       # any other error means authorization PASSED
            return "allowed"

    def test_matrix(self):
        # name: (call, viewer allowed?, operator allowed?, administrator allowed?)
        matrix = {
            "read status":   (lambda t: self.svc.get_status(t), True, True, True),
            "read events":   (lambda t: self.svc.list_events(t), True, True, True),
            "read zones":    (lambda t: self.svc.list_zones(t), True, True, True),
            "read rules":    (lambda t: self.svc.list_rules(t), True, True, True),
            "read cameras":  (lambda t: self.svc.list_cameras(t), True, True, True),
            "start session": (lambda t: self.svc.start_session(t, "no-such-camera"), False, True, True),
            "stop session":  (lambda t: self.svc.stop_session(t, "nope"), False, True, True),
            "create camera": (lambda t: self.svc.create_camera(t, "", ""), False, False, True),
            "delete camera": (lambda t: self.svc.delete_camera(t, "x"), False, False, True),
            "create zone":   (lambda t: self.svc.create_zone(t, "", []), False, False, True),
            "create rule":   (lambda t: self.svc.create_rule(t, "", {}), False, False, True),
            "run retention": (lambda t: self.svc.run_retention(t), False, False, True),
            "create user":   (lambda t: self.svc.create_user(t, "", "", "viewer"), False, False, True),
            "list users":    (lambda t: self.svc.list_users(t), False, False, True),
        }
        tokens = (self.view, self.oper, self.admin)
        for name, (call, *allowed) in matrix.items():
            got = [self.outcome(lambda tok=tok: call(tok)) == "allowed" for tok in tokens]
            self.assertEqual(got, allowed, f"{name}: (viewer, operator, admin)")


class TestValidation(ApiCase):
    def test_zone_validation(self):
        for bad in ([[0, 0], [1, 1]], "nope", [[0, "a"], [1, 1], [2, 2]], [[0], [1], [2]], None):
            with self.assertRaises(ValidationError, msg=repr(bad)):
                self.svc.create_zone(self.admin, "Z", bad)
        with self.assertRaises(ValidationError):
            self.svc.create_zone(self.admin, "  ", RECT)
        with self.assertRaises(ValidationError):
            self.svc.create_zone(self.admin, "Z", RECT, camera_id="no-such-camera")

    def test_camera_validation(self):
        for kw in (dict(name="", source_uri="0"), dict(name="c", source_uri=" "),
                   dict(name="c", source_uri="0", orientation="diagonal"),
                   dict(name="c", source_uri="0", frame_width=-5), dict(name="c", source_uri="0", frame_width=True)):
            with self.assertRaises(ValidationError, msg=str(kw)):
                self.svc.create_camera(self.admin, **kw)

    def test_rule_validation_rejects_bad_rules_at_creation(self):
        bad_defs = [
            {}, {"when": {}}, {"when": {"event_type": "NOT_A_TYPE"}},
            {"when": {"event_type": "ZONE_ENTRY"}, "if": [{"field": "confidence", "operator": "~=", "value": 1}]},
            {"when": {"event_type": "ZONE_ENTRY"}, "if": [{"field": "secret_field", "operator": ">", "value": 1}]},
            {"when": {"event_type": "ZONE_ENTRY"}, "if": [{"field": "confidence", "operator": ">", "value": "high"}]},
            {"when": {"event_type": "ZONE_ENTRY"}, "if": [{"field": "confidence"}]},
            {"when": {"event_type": "ZONE_ENTRY"}, "action": "LAUNCH_MISSILES"},
        ]
        for d in bad_defs:
            with self.assertRaises(ValidationError, msg=str(d)):
                self.svc.create_rule(self.admin, "R", d)
        self.assertEqual(self.svc.list_rules(self.admin), [])

    def test_rule_referring_to_a_missing_zone_is_rejected(self):
        with self.assertRaises(ValidationError):
            self.svc.create_rule(self.admin, "R", {"when": {"event_type": "ZONE_ENTRY", "zone_id": "ghost"}})

    def test_user_validation_and_duplicates(self):
        with self.assertRaises(ValidationError):
            self.svc.create_user(self.admin, "new", "short", "viewer")
        with self.assertRaises(ValidationError):
            self.svc.create_user(self.admin, "new", "password-123", "emperor")
        self.svc.create_user(self.admin, "new", "password-123", "viewer")
        with self.assertRaises(ValidationError):
            self.svc.create_user(self.admin, "new", "password-123", "viewer")


class TestConfigurationCrud(ApiCase):
    def test_camera_zone_rule_lifecycle(self):
        cam = self.svc.create_camera(self.admin, "Gate", "rtsp://x", 640, 480)
        zone = self.svc.create_zone(self.admin, "Restricted", RECT, camera_id=cam["camera_id"])
        rule = self.svc.create_rule(self.admin, "Entry", {"when": {"event_type": "ZONE_ENTRY", "zone_id": zone["zone_id"]},
                                                          "if": [{"field": "confidence", "operator": ">", "value": 0.5}]})
        self.assertEqual([c["name"] for c in self.svc.list_cameras(self.view)], ["Gate"])
        self.assertEqual(self.svc.list_zones(self.view, camera_id=cam["camera_id"])[0]["polygon"], [list(p) for p in RECT])
        self.assertEqual(self.svc.list_rules(self.view)[0]["definition"]["if"][0]["value"], 0.5)

        self.svc.delete_rule(self.admin, rule["rule_id"])
        self.svc.delete_zone(self.admin, zone["zone_id"])
        self.svc.delete_camera(self.admin, cam["camera_id"])
        for lister in (self.svc.list_cameras, self.svc.list_zones, self.svc.list_rules):
            self.assertEqual(lister(self.view), [])

    def test_deleting_missing_things_is_notfound(self):
        for call in (self.svc.delete_camera, self.svc.delete_zone, self.svc.delete_rule):
            with self.assertRaises(NotFound):
                call(self.admin, "ghost")

    def test_user_listing_never_exposes_password_hashes(self):
        users = self.svc.list_users(self.admin)
        self.assertEqual({u["username"] for u in users}, {"admin", "oper", "view"})
        self.assertTrue(all("password_hash" not in u for u in users))


class TestSessionsEndToEnd(ApiCase):
    def setUp(self):
        super().setUp()
        self.cam = self.svc.create_camera(self.admin, "Gate", "fake://0")["camera_id"]
        self.zone = self.svc.create_zone(self.admin, "Restricted", RECT, camera_id=self.cam)["zone_id"]
        self.svc.create_rule(self.admin, "Person in restricted zone", {
            "when": {"event_type": "ZONE_ENTRY", "entity_type": "person", "zone_id": self.zone},
            "if": [{"field": "confidence", "operator": ">", "value": 0.5}]})

    def run_session(self):
        started = self.svc.start_session(self.oper, self.cam)
        self.wait_until_finished()
        return started

    def test_operator_can_run_a_session_and_the_results_are_queryable_through_the_api(self):
        started = self.run_session()
        self.assertEqual(started["state"], "RUNNING")

        events = self.svc.list_events(self.view, EventQuery(event_type=EventType.ZONE_ENTRY))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["state"], "RESOLVED")
        self.assertEqual(events[0]["zone_id"], self.zone)

        alerts = self.svc.list_alerts(self.view)
        self.assertEqual(len(alerts), 1)                                   # deduplicated
        self.assertEqual(alerts[0]["event_id"], events[0]["event_id"])

        detail = self.svc.get_event(self.view, events[0]["event_id"])
        self.assertTrue(detail["evidence_record"]["has_snapshot"])
        self.assertNotIn("frame_snapshot_path", str(detail))               # no server paths leaked

        sessions = self.svc.list_sessions(self.view)
        self.assertEqual((sessions[0]["state"], sessions[0]["frame_count"]), ("STOPPED", 10))
        self.assertEqual(len(self.svc.list_entities(self.view, session_id=started["session_id"])), 1)

    def test_evidence_image_is_served_from_inside_the_evidence_directory_only(self):
        self.run_session()
        event_id = self.svc.list_alerts(self.view)[0]["event_id"]
        path = self.svc.get_evidence_image_path(self.view, event_id)
        self.assertTrue(os.path.isfile(path))
        self.assertTrue(os.path.realpath(path).startswith(os.path.realpath(self.tmp) + os.sep))

    def test_a_tampered_evidence_path_cannot_be_used_to_read_arbitrary_files(self):
        self.run_session()
        event_id = self.svc.list_alerts(self.view)[0]["event_id"]
        self.repo.db.execute("UPDATE evidence SET frame_snapshot_path = ? WHERE event_id = ?", ("/etc/passwd", event_id))
        with self.assertRaises(NotFound):
            self.svc.get_evidence_image_path(self.view, event_id)
        self.repo.db.execute("UPDATE evidence SET frame_snapshot_path = ? WHERE event_id = ?",
                             (os.path.join(self.tmp, "..", "..", "etc", "passwd"), event_id))
        with self.assertRaises(NotFound):
            self.svc.get_evidence_image_path(self.view, event_id)

    def test_status_reports_healthy_components_and_metrics_after_a_run(self):
        self.run_session()
        status = self.svc.get_status(self.view)
        for comp in ("camera", "vision", "tracking", "events", "database"):
            self.assertEqual(status["components"][comp]["status"], "OK", comp)
        self.assertEqual(status["overall"], "OK")
        self.assertIn("fps", status["performance"])
        self.assertEqual(status["active_sessions"], [])                    # finished session was reaped

    def test_second_session_on_a_running_camera_is_a_conflict_and_stop_works(self):
        self.frames = 1_000_000
        self.svc.start_session(self.oper, self.cam)
        with self.assertRaises(Conflict):
            self.svc.start_session(self.oper, self.cam)
        with self.assertRaises(Conflict):
            self.svc.delete_camera(self.admin, self.cam)                    # can't yank a live camera
        sid = self.svc.get_status(self.view)["active_sessions"][0]
        self.assertEqual(self.svc.stop_session(self.oper, sid)["state"], "STOPPED")
        self.assertEqual(self.svc.get_status(self.view)["active_sessions"], [])
        self.svc.delete_camera(self.admin, self.cam)                        # now allowed

    def test_unknown_camera_and_unknown_session(self):
        with self.assertRaises(NotFound):
            self.svc.start_session(self.oper, "ghost")
        with self.assertRaises(NotFound):
            self.svc.stop_session(self.oper, "ghost")

    def test_unreachable_camera_reports_camera_offline(self):
        self.fail_open = 99
        with self.assertRaises(CameraUnavailable) as ctx:
            self.svc.start_session(self.oper, self.cam)
        self.assertIn("CAMERA OFFLINE", str(ctx.exception))
        self.assertEqual(self.svc.get_status(self.view)["components"]["camera"]["message"], "CAMERA OFFLINE")

    def test_default_factory_without_a_model_fails_clearly_instead_of_crashing(self):
        svc = ApiService(self.repo, self.auth, self.health, self.monitor, evidence_dir=self.tmp)   # default factory, no model
        with self.assertRaises(CameraUnavailable):
            svc.start_session(self.oper, self.cam)

    def test_shutdown_stops_running_sessions(self):
        self.frames = 1_000_000
        self.svc.start_session(self.oper, self.cam)
        self.svc.shutdown()
        self.assertEqual(self.svc._runners, {})
        self.assertEqual(self.svc.list_sessions(self.view)[0]["state"], "STOPPED")


class TestEventsAndAlertsQueries(ApiCase):
    def test_get_event_unknown_is_notfound(self):
        with self.assertRaises(NotFound):
            self.svc.get_event(self.view, "ghost")

    def test_list_events_filters_pass_through(self):
        for start, et in ((100.0, EventType.APPEARED), (200.0, EventType.ZONE_ENTRY)):
            ev = Event.new(et, ["e1"], "person", start, 0.9, zone_id=None)
            ev.end_time, ev.state = start, EventState.RESOLVED
            self.repo.save_event(ev)
        got = self.svc.list_events(self.view, EventQuery(event_type=EventType.ZONE_ENTRY))
        self.assertEqual([e["start_time"] for e in got], [200.0])


class TestRetentionThroughTheApi(ApiCase):
    def test_admin_runs_retention_and_the_run_is_recorded(self):
        old = time.time() - 60 * 86400
        ev = Event.new(EventType.APPEARED, ["e1"], "person", old, 0.9)
        ev.end_time, ev.state = old, EventState.RESOLVED
        self.repo.save_event(ev)
        result = self.svc.run_retention(self.admin)
        self.assertEqual(result["events_deleted"], 1)
        self.assertEqual(self.repo.count("events"), 0)
        self.assertEqual(self.repo.get_state("last_retention_run")["events_deleted"], 1)


class TestApiLatencyIsMeasured(ApiCase):
    def test_successful_and_rejected_calls_both_record_latency(self):
        self.assertIsNone(PerformanceMonitor().snapshot()["api_latency_ms_avg"])
        self.svc.get_status(self.view)
        with self.assertRaises(PermissionDenied):
            self.svc.create_camera(self.view, "c", "0")
        with self.assertRaises(AuthenticationError):
            self.svc.list_events("garbage")
        self.assertIsNotNone(self.monitor.snapshot()["api_latency_ms_avg"])
        self.assertGreaterEqual(len(self.monitor._api), 4)                  # 3 logins + these calls


class TestErrorMapping(unittest.TestCase):
    def test_status_codes(self):
        cases = [(NotFound("x"), 404), (ValidationError("x"), 422), (Conflict("x"), 409),
                 (CameraUnavailable("x"), 503), (AuthenticationError("x"), 401), (PermissionDenied("x"), 403),
                 (DataStorageUnavailable("x"), 503), (DataIntegrityError("x"), 422), (ValueError("x"), 422),
                 (RuntimeError("x"), 500)]
        for exc, code in cases:
            self.assertEqual(status_for(exc), code, type(exc).__name__)

    def test_internal_details_never_reach_the_client(self):
        leaky = DataStorageUnavailable("no such table: users at /var/lib/vision/db.sqlite")
        self.assertEqual(public_message(leaky), "DATA STORAGE UNAVAILABLE")
        self.assertEqual(public_message(RuntimeError("secret /etc/shadow stack trace")), "Internal server error")
        self.assertNotIn("sqlite", public_message(DataIntegrityError("UNIQUE constraint failed: users.username")))


if __name__ == "__main__":
    unittest.main()


class TestLatestFrameEndpoint(ApiCase):
    def test_only_a_running_session_has_a_live_frame(self):
        from vision_radar.api.errors import NotFound
        cam = self.svc.create_camera(self.admin, "Gate", "fake://0")
        with self.assertRaises(NotFound):
            self.svc.get_latest_frame(self.view, "no-such-session")

    def test_live_frame_is_available_while_running_and_gone_once_finished(self):
        from vision_radar.api.errors import NotFound
        self.frames = 1_000_000
        cam = self.svc.create_camera(self.admin, "Gate", "fake://0")
        started = self.svc.start_session(self.oper, cam["camera_id"])
        deadline = time.time() + 5
        frame = None
        while time.time() < deadline:
            try:
                frame = self.svc.get_latest_frame(self.view, started["session_id"])
                break
            except NotFound:
                time.sleep(0.02)
        self.assertIsNotNone(frame)
        self.assertEqual(frame[:2], b"\xff\xd8")             # real JPEG magic bytes
        self.svc.stop_session(self.oper, started["session_id"])
        with self.assertRaises(NotFound):
            self.svc.get_latest_frame(self.view, started["session_id"])   # no stale frame after it ends
