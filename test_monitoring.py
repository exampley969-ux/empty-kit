import unittest

from vision_radar.monitoring.health import ComponentStatus, HealthRegistry
from vision_radar.monitoring.performance import PerformanceMonitor


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


class TestHealthRegistry(unittest.TestCase):
    def test_unreported_components_are_unknown_and_stale(self):
        h = HealthRegistry().get("camera")
        self.assertEqual(h.status, ComponentStatus.UNKNOWN)
        self.assertTrue(h.stale)

    def test_spec_failure_messages(self):
        reg = HealthRegistry()
        reg.set("camera", ComponentStatus.ERROR)
        reg.set("vision", ComponentStatus.ERROR)
        reg.set("database", ComponentStatus.ERROR)
        reg.set("tracking", ComponentStatus.DEGRADED)
        self.assertEqual(reg.get("camera").message, "CAMERA OFFLINE")
        self.assertEqual(reg.get("vision").message, "VISION ENGINE ERROR")
        self.assertEqual(reg.get("database").message, "DATA STORAGE UNAVAILABLE")
        self.assertEqual(reg.get("tracking").message, "TRACKING DEGRADED")

    def test_healthy_components_carry_no_error_message(self):
        reg = HealthRegistry()
        reg.set("camera", ComponentStatus.OK)
        self.assertIsNone(reg.get("camera").message)

    def test_unknown_component_rejected(self):
        with self.assertRaises(ValueError):
            HealthRegistry().set("toaster", ComponentStatus.OK)

    def test_stale_data_is_flagged_not_presented_as_live(self):
        clock = FakeClock()
        reg = HealthRegistry(stale_after_seconds=10, clock=clock)
        reg.set("camera", ComponentStatus.OK)
        self.assertFalse(reg.get("camera").stale)
        clock.advance(11)
        h = reg.get("camera")
        self.assertEqual(h.status, ComponentStatus.OK)      # last known value is kept...
        self.assertTrue(h.stale)                             # ...but marked stale

    def test_overall_rollup(self):
        clock = FakeClock()
        reg = HealthRegistry(stale_after_seconds=10, clock=clock)
        self.assertEqual(reg.overall(), ComponentStatus.DEGRADED)     # nothing reported yet
        for c in ("camera", "vision", "tracking", "events", "database"):
            reg.set(c, ComponentStatus.OK)
        self.assertEqual(reg.overall(), ComponentStatus.OK)
        reg.set("tracking", ComponentStatus.DEGRADED)
        self.assertEqual(reg.overall(), ComponentStatus.DEGRADED)
        reg.set("camera", ComponentStatus.ERROR)
        self.assertEqual(reg.overall(), ComponentStatus.ERROR)

    def test_a_component_that_stops_reporting_stops_counting_as_ok(self):
        clock = FakeClock()
        reg = HealthRegistry(stale_after_seconds=10, clock=clock)
        for c in ("camera", "vision", "tracking", "events", "database"):
            reg.set(c, ComponentStatus.OK)
        clock.advance(30)
        self.assertEqual(reg.overall(), ComponentStatus.DEGRADED)      # silence != health


class TestPerformanceMonitor(unittest.TestCase):
    def test_fps_from_frame_timestamps(self):
        clock = FakeClock()
        mon = PerformanceMonitor(window_seconds=5, clock=clock)
        for _ in range(11):              # 11 frames, 0.1s apart -> 10 fps
            mon.record_frame(20.0, detections=1, tracked=1)
            clock.advance(0.1)
        self.assertAlmostEqual(mon.snapshot()["fps"], 10.0, delta=0.6)

    def test_average_processing_latency(self):
        clock = FakeClock()
        mon = PerformanceMonitor(clock=clock)
        for ms in (10.0, 20.0, 30.0):
            mon.record_frame(ms, 0, 0)
            clock.advance(0.1)
        self.assertEqual(mon.snapshot()["processing_ms_avg"], 20.0)

    def test_counts_reflect_latest_frame(self):
        mon = PerformanceMonitor(clock=FakeClock())
        mon.record_frame(5, detections=7, tracked=3)
        s = mon.snapshot()
        self.assertEqual((s["detections"], s["tracked"]), (7, 3))

    def test_old_frames_fall_out_of_the_window(self):
        clock = FakeClock()
        mon = PerformanceMonitor(window_seconds=5, clock=clock)
        mon.record_frame(10, 0, 0)
        clock.advance(30)
        s = mon.snapshot()
        self.assertEqual(s["fps"], 0.0)
        self.assertIsNone(s["processing_ms_avg"])

    def test_events_per_minute_counts_a_rolling_minute(self):
        clock = FakeClock()
        mon = PerformanceMonitor(clock=clock)
        mon.record_events(4)
        self.assertEqual(mon.snapshot()["events_per_min"], 4)
        clock.advance(61)
        self.assertEqual(mon.snapshot()["events_per_min"], 0)

    def test_api_latency_average_and_none_when_idle(self):
        clock = FakeClock()
        mon = PerformanceMonitor(clock=clock)
        self.assertIsNone(mon.snapshot()["api_latency_ms_avg"])
        mon.record_api_latency(10.0)
        mon.record_api_latency(30.0)
        self.assertEqual(mon.snapshot()["api_latency_ms_avg"], 20.0)

    def test_system_metrics_are_real_and_gpu_is_honestly_unavailable(self):
        s = PerformanceMonitor().snapshot()
        self.assertGreater(s["memory_mb"], 1.0)
        self.assertGreaterEqual(s["cpu_percent"], 0.0)
        self.assertIsNone(s["gpu_percent"])

    def test_every_section_27_metric_is_present(self):
        s = PerformanceMonitor().snapshot()
        for key in ("fps", "processing_ms_avg", "cpu_percent", "memory_mb", "gpu_percent",
                    "detections", "tracked", "events_per_min", "api_latency_ms_avg"):
            self.assertIn(key, s)


if __name__ == "__main__":
    unittest.main()
