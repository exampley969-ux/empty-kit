"""
Vision Radar - Monitoring: performance
The system watches itself.

Spec reference: PDF Section 27 (FPS, processing latency, CPU/GPU usage,
memory, detection count, tracking count, event rate, camera health, API
latency).

Honesty notes:
  * GPU usage is reported as None ("not available"), never a made-up number:
    the inference stack here is CPU-only (onnxruntime CPUExecutionProvider)
    and no GPU probe exists in this build.
  * cpu_percent is machine-wide utilisation since the previous snapshot (the
    first snapshot after startup reads ~0, an artefact of how psutil samples).
  * Camera health comes from the HealthRegistry, merged in by the API layer.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable, Deque, Optional, Tuple

import psutil


class PerformanceMonitor:
    def __init__(self, window_seconds: float = 5.0, clock: Callable[[], float] = time.time):
        self.window_seconds = window_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._frames: Deque[Tuple[float, float]] = deque()    # (timestamp, processing_ms)
        self._events: Deque[float] = deque()                    # timestamps
        self._api: Deque[Tuple[float, float]] = deque()         # (timestamp, latency_ms)
        self._latest_detections = 0
        self._latest_tracked = 0
        self._process = psutil.Process()
        psutil.cpu_percent(interval=None)     # prime the sampler

    def record_frame(self, processing_ms: float, detections: int, tracked: int) -> None:
        now = self._clock()
        with self._lock:
            self._frames.append((now, processing_ms))
            self._latest_detections = detections
            self._latest_tracked = tracked
            self._trim(now)

    def record_events(self, count: int = 1) -> None:
        now = self._clock()
        with self._lock:
            self._events.extend([now] * count)
            self._trim(now)

    def record_api_latency(self, latency_ms: float) -> None:
        now = self._clock()
        with self._lock:
            self._api.append((now, latency_ms))
            self._trim(now)

    def _trim(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._frames and self._frames[0][0] < cutoff:
            self._frames.popleft()
        while self._api and self._api[0][0] < cutoff:
            self._api.popleft()
        event_cutoff = now - 60.0          # the events/min figure needs a full minute
        while self._events and self._events[0] < event_cutoff:
            self._events.popleft()

    def snapshot(self) -> dict:
        now = self._clock()
        with self._lock:
            self._trim(now)
            frames, api = list(self._frames), list(self._api)
            events_last_minute = len(self._events)
            detections, tracked = self._latest_detections, self._latest_tracked

        # fps = frames per second over the window actually covered by samples
        fps = 0.0
        if len(frames) >= 2:
            span = frames[-1][0] - frames[0][0]
            fps = (len(frames) - 1) / span if span > 0 else 0.0

        return {
            "fps": round(fps, 2),
            "processing_ms_avg": round(sum(m for _, m in frames) / len(frames), 2) if frames else None,
            "cpu_percent": psutil.cpu_percent(interval=None),
            "memory_mb": round(self._process.memory_info().rss / (1024 * 1024), 1),
            "gpu_percent": None,                              # not available in this build
            "detections": detections,
            "tracked": tracked,
            "events_per_min": events_last_minute,
            "api_latency_ms_avg": round(sum(m for _, m in api) / len(api), 2) if api else None,
        }
