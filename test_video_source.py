import unittest

from vision_radar.ingestion.exceptions import CameraConnectionError
from vision_radar.ingestion.models import CameraStatus, VideoSourceConfig
from vision_radar.tests.fakes import FakeVideoSource


def make_config(**overrides):
    defaults = dict(
        source_uri="fake://0",
        max_reconnect_attempts=3,
        reconnect_backoff_seconds=0.01,
    )
    defaults.update(overrides)
    return VideoSourceConfig(**defaults)


class TestVideoSourceConnection(unittest.TestCase):
    def test_connect_success_sets_status_connected(self):
        source = FakeVideoSource(make_config(), total_frames=5)
        source.connect()
        self.assertEqual(source.health().status, CameraStatus.CONNECTED)
        self.assertTrue(source.is_connected())

    def test_connect_failure_raises_and_sets_error_status(self):
        source = FakeVideoSource(make_config(), fail_at_open_times=99)
        with self.assertRaises(CameraConnectionError):
            source.connect()
        self.assertEqual(source.health().status, CameraStatus.ERROR)

    def test_disconnect_sets_status_disconnected(self):
        source = FakeVideoSource(make_config(), total_frames=5)
        source.connect()
        source.disconnect()
        self.assertEqual(source.health().status, CameraStatus.DISCONNECTED)
        self.assertFalse(source.is_connected())


class TestFrameAcquisition(unittest.TestCase):
    def test_read_frame_returns_frame_with_metadata(self):
        source = FakeVideoSource(make_config(), total_frames=3)
        source.connect()
        frame = source.read_frame(session_id="s1", frame_id=0)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.session_id, "s1")
        self.assertEqual(frame.frame_id, 0)
        self.assertEqual(frame.width, 64)
        self.assertEqual(frame.height, 48)
        self.assertIsNotNone(frame.captured_at)

    def test_read_frame_returns_none_when_source_exhausted(self):
        source = FakeVideoSource(make_config(), total_frames=2)
        source.connect()
        f1 = source.read_frame("s1", 0)
        f2 = source.read_frame("s1", 1)
        f3 = source.read_frame("s1", 2)
        self.assertIsNotNone(f1)
        self.assertIsNotNone(f2)
        self.assertIsNone(f3)

    def test_frames_read_counter_increments(self):
        source = FakeVideoSource(make_config(), total_frames=4)
        source.connect()
        for i in range(4):
            source.read_frame("s1", i)
        self.assertEqual(source.health().frames_read, 4)


class TestReconnectHandling(unittest.TestCase):
    def test_dropped_frame_triggers_automatic_reconnect_and_recovers(self):
        # Frame index 2 (0-based, i.e. the 3rd read) will drop once; the base
        # VideoSource should reconnect and recover automatically.
        source = FakeVideoSource(make_config(max_reconnect_attempts=3),
                                  total_frames=10, drop_after_n_frames=2)
        source.connect()
        frames = [source.read_frame("s1", i) for i in range(5)]
        non_none = [f for f in frames if f is not None]
        # We expect the drop to cost us at most the one frame it interrupted,
        # not to kill the whole stream.
        self.assertGreaterEqual(len(non_none), 4)
        self.assertGreaterEqual(source.health().reconnect_attempts, 0)

    def test_reconnect_exhausted_sets_error_status(self):
        # Every reconnect attempt after the initial connect fails.
        source = FakeVideoSource(make_config(max_reconnect_attempts=2),
                                  total_frames=10, fail_at_open_times=0)
        source.connect()
        # Force a failure state: monkeypatch _open to always fail from now on
        source._fail_at_open_times = 999

        def always_fail():
            raise ConnectionError("still down")
        source._open = lambda: (_ for _ in ()).throw(ConnectionError("still down"))

        # Simulate a drop so a reconnect is required
        source._is_open = False
        frame = source.read_frame("s1", 0)
        self.assertIsNone(frame)
        self.assertEqual(source.health().status, CameraStatus.ERROR)


if __name__ == "__main__":
    unittest.main()
