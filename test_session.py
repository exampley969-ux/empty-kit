import time
import unittest

from vision_radar.ingestion.exceptions import SessionError
from vision_radar.ingestion.models import SessionState, VideoSourceConfig
from vision_radar.ingestion.session import IngestionSession, SessionManager
from vision_radar.tests.fakes import FakeVideoSource


def make_config(**overrides):
    defaults = dict(source_uri="fake://0", max_reconnect_attempts=2,
                     reconnect_backoff_seconds=0.01)
    defaults.update(overrides)
    return VideoSourceConfig(**defaults)


class TestSessionLifecycle(unittest.TestCase):
    def test_start_sets_state_running(self):
        source = FakeVideoSource(make_config(), total_frames=5)
        session = IngestionSession(source, camera_uri="fake://0")
        session.start()
        self.assertEqual(session.info.state, SessionState.RUNNING)
        self.assertIsNotNone(session.info.started_at)

    def test_double_start_raises(self):
        source = FakeVideoSource(make_config(), total_frames=5)
        session = IngestionSession(source, camera_uri="fake://0")
        session.start()
        with self.assertRaises(SessionError):
            session.start()

    def test_stop_sets_state_stopped_and_end_time(self):
        source = FakeVideoSource(make_config(), total_frames=5)
        session = IngestionSession(source, camera_uri="fake://0")
        session.start()
        session.stop()
        self.assertEqual(session.info.state, SessionState.STOPPED)
        self.assertIsNotNone(session.info.ended_at)

    def test_reading_frame_without_start_raises(self):
        source = FakeVideoSource(make_config(), total_frames=5)
        session = IngestionSession(source, camera_uri="fake://0")
        with self.assertRaises(SessionError):
            session.next_frame()

    def test_frame_count_increments_and_stops_at_exhaustion(self):
        source = FakeVideoSource(make_config(), total_frames=3)
        session = IngestionSession(source, camera_uri="fake://0")
        session.start()
        collected = list(session.frames())
        self.assertEqual(len(collected), 3)
        self.assertEqual(session.info.frame_count, 3)

    def test_status_report_contains_expected_keys(self):
        source = FakeVideoSource(make_config(), total_frames=3)
        session = IngestionSession(source, camera_uri="fake://0")
        session.start()
        session.next_frame()
        status = session.status()
        for key in ("session_id", "camera_uri", "session_state", "camera_status",
                    "frame_count", "fps_actual", "error_message"):
            self.assertIn(key, status)
        self.assertEqual(status["frame_count"], 1)
        self.assertEqual(status["session_state"], "RUNNING")


class TestSessionManager(unittest.TestCase):
    def _factory(self, total_frames=5):
        def factory(config):
            return FakeVideoSource(config, total_frames=total_frames)
        return factory

    def test_start_session_registers_and_runs(self):
        manager = SessionManager()
        session = manager.start_session(make_config(), self._factory())
        self.assertEqual(session.info.state, SessionState.RUNNING)
        self.assertEqual(len(manager.list_sessions()), 1)

    def test_stop_session_by_id(self):
        manager = SessionManager()
        session = manager.start_session(make_config(), self._factory())
        manager.stop_session(session.info.session_id)
        self.assertEqual(session.info.state, SessionState.STOPPED)

    def test_multiple_independent_sessions(self):
        manager = SessionManager()
        s1 = manager.start_session(make_config(source_uri="fake://0"), self._factory())
        s2 = manager.start_session(make_config(source_uri="fake://1"), self._factory())
        self.assertNotEqual(s1.info.session_id, s2.info.session_id)
        self.assertEqual(len(manager.list_sessions()), 2)
        manager.stop_session(s1.info.session_id)
        statuses = {s["session_id"]: s["session_state"] for s in manager.list_sessions()}
        self.assertEqual(statuses[s1.info.session_id], "STOPPED")
        self.assertEqual(statuses[s2.info.session_id], "RUNNING")

    def test_unknown_session_id_raises(self):
        manager = SessionManager()
        with self.assertRaises(SessionError):
            manager.stop_session("does-not-exist")


if __name__ == "__main__":
    unittest.main()
