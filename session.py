"""
Vision Radar - Video Ingestion Layer
Session management: start/stop, frame acquisition loop, camera status,
multi-session tracking.

Spec reference: PDF Section 4 (Video ingestion layer), Section 23 (Data
architecture: Camera -> Session -> Frames/Observations).
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Dict, Iterator, Optional

from .exceptions import SessionError
from .models import CameraStatus, Frame, SessionInfo, SessionState, VideoSourceConfig
from .video_source import VideoSource

logger = logging.getLogger("vision_radar.ingestion")


class IngestionSession:
    """
    Owns one VideoSource for the lifetime of one session. A session is the
    unit the rest of the system (Vision Engine onward) attaches to.
    """

    def __init__(self, source: VideoSource, camera_uri: str, session_id: Optional[str] = None):
        self._source = source
        self.info = SessionInfo(
            session_id=session_id or str(uuid.uuid4()),
            camera_uri=camera_uri,
            state=SessionState.IDLE,
        )
        self._fps_window: list[float] = []

    # --- lifecycle -----------------------------------------------------
    def start(self) -> None:
        if self.info.state == SessionState.RUNNING:
            raise SessionError(f"Session {self.info.session_id} is already running")

        self.info.state = SessionState.STARTING
        try:
            self._source.connect()
        except Exception as exc:
            self.info.state = SessionState.ERROR
            self.info.error_message = str(exc)
            raise

        self.info.state = SessionState.RUNNING
        self.info.started_at = time.time()
        self.info.error_message = None
        logger.info("Session %s started (camera=%s)", self.info.session_id, self.info.camera_uri)

    def stop(self) -> None:
        if self.info.state not in (SessionState.RUNNING, SessionState.ERROR):
            raise SessionError(
                f"Session {self.info.session_id} is not running (state={self.info.state})"
            )
        self.info.state = SessionState.STOPPING
        self._source.disconnect()
        self.info.state = SessionState.STOPPED
        self.info.ended_at = time.time()
        logger.info(
            "Session %s stopped (frames=%s, duration=%.1fs)",
            self.info.session_id, self.info.frame_count, self.info.duration_seconds() or 0.0,
        )

    # --- frame acquisition ------------------------------------------------
    def next_frame(self) -> Optional[Frame]:
        """Read the next frame. Returns None when the source is exhausted."""
        if self.info.state != SessionState.RUNNING:
            raise SessionError(
                f"Cannot read frame: session {self.info.session_id} is not running "
                f"(state={self.info.state})"
            )

        frame = self._source.read_frame(self.info.session_id, self.info.frame_count)

        if frame is None:
            health = self._source.health()
            if health.status == CameraStatus.ERROR:
                self.info.state = SessionState.ERROR
                self.info.error_message = health.last_error or "camera offline"
            return None

        self.info.frame_count += 1
        self._track_fps(frame.captured_at)
        return frame

    def frames(self) -> Iterator[Frame]:
        """Convenience generator: yields frames until the source is exhausted."""
        while True:
            frame = self.next_frame()
            if frame is None:
                return
            yield frame

    def _track_fps(self, ts: float) -> None:
        self._fps_window.append(ts)
        cutoff = ts - 5.0
        self._fps_window = [t for t in self._fps_window if t >= cutoff]
        if len(self._fps_window) >= 2:
            span = self._fps_window[-1] - self._fps_window[0]
            self._source.health().fps_actual = (
                (len(self._fps_window) - 1) / span if span > 0 else 0.0
            )

    # --- status -------------------------------------------------------
    def status(self) -> dict:
        health = self._source.health()
        return {
            "session_id": self.info.session_id,
            "camera_uri": self.info.camera_uri,
            "session_state": self.info.state.value,
            "camera_status": health.status.value,
            "frame_count": self.info.frame_count,
            "frames_dropped": health.frames_dropped,
            "fps_actual": round(health.fps_actual, 2),
            "reconnect_attempts": health.reconnect_attempts,
            "started_at": self.info.started_at,
            "duration_seconds": self.info.duration_seconds(),
            "error_message": self.info.error_message,
        }


class SessionManager:
    """Tracks all active/past ingestion sessions (in-memory for Phase 1;
    Phase 9 - Data Layer will persist SessionInfo to the database)."""

    def __init__(self):
        self._sessions: Dict[str, IngestionSession] = {}

    def start_session(self, config: VideoSourceConfig, source_factory) -> IngestionSession:
        """
        source_factory: callable(config) -> VideoSource, so this manager stays
        independent of any concrete VideoSource implementation (Section 4).
        """
        source = source_factory(config)
        session = IngestionSession(source=source, camera_uri=config.source_uri)
        session.start()
        self._sessions[session.info.session_id] = session
        return session

    def stop_session(self, session_id: str) -> None:
        session = self.get_session(session_id)
        session.stop()

    def get_session(self, session_id: str) -> IngestionSession:
        if session_id not in self._sessions:
            raise SessionError(f"Unknown session_id: {session_id}")
        return self._sessions[session_id]

    def list_sessions(self) -> list:
        return [s.status() for s in self._sessions.values()]

    def remove_stopped(self, session_id: str) -> None:
        session = self._sessions.get(session_id)
        if session and session.info.state in (SessionState.STOPPED, SessionState.ERROR):
            del self._sessions[session_id]
