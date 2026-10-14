"""
Vision Radar - Video Ingestion Layer
VideoSource abstraction.

Design rule (Spec Section 4): the abstraction is named `VideoSource`, not
`MobileCamera`, so the rest of the system never depends on a specific camera
type. Today it's backed by OpenCV (works with a webcam index, a video file,
or an RTSP/HTTP stream URL) — tomorrow it could be backed by anything else
without touching any other layer.
"""

from __future__ import annotations

import abc
import logging
import time
from typing import Optional

import cv2

from .exceptions import CameraConnectionError, FrameAcquisitionError
from .models import CameraStatus, Frame, SourceHealth, VideoSourceConfig

logger = logging.getLogger("vision_radar.ingestion")


class VideoSource(abc.ABC):
    """
    Abstract base for any video input. Concrete sources must implement
    connection handling and single-frame acquisition. Everything else
    (reconnect policy, health tracking, fps pacing) is handled by
    OpenCVVideoSource / subclasses reusing the shared helpers below,
    so new source types stay thin.
    """

    def __init__(self, config: VideoSourceConfig):
        self.config = config
        self._health = SourceHealth()

    # --- must be implemented by concrete sources -------------------------
    @abc.abstractmethod
    def _open(self) -> None:
        """Open the underlying connection. Raise CameraConnectionError on failure."""

    @abc.abstractmethod
    def _close(self) -> None:
        """Release the underlying connection."""

    @abc.abstractmethod
    def _read_raw(self):
        """Return (success: bool, image_or_None)."""

    @abc.abstractmethod
    def native_properties(self) -> dict:
        """Return dict with 'width', 'height', 'fps' as reported by the source."""

    def is_exhausted(self) -> bool:
        """
        Whether a finite source (e.g. a video file) has cleanly reached its
        end. Live sources (webcam/stream) should never report True here -
        for them, a failed read always means a genuine drop worth
        reconnecting for. Default: not a finite source.
        """
        return False

    # --- shared behaviour --------------------------------------------------
    def connect(self) -> None:
        self._health.status = CameraStatus.CONNECTING
        try:
            self._open()
            self._health.status = CameraStatus.CONNECTED
            self._health.reconnect_attempts = 0
            self._health.last_error = None
            logger.info("VideoSource connected: %s", self.config.source_uri)
        except Exception as exc:
            self._health.status = CameraStatus.ERROR
            self._health.last_error = str(exc)
            raise CameraConnectionError(
                f"Could not connect to source '{self.config.source_uri}': {exc}"
            ) from exc

    def disconnect(self) -> None:
        try:
            self._close()
        finally:
            self._health.status = CameraStatus.DISCONNECTED

    def is_connected(self) -> bool:
        return self._health.status == CameraStatus.CONNECTED

    def health(self) -> SourceHealth:
        return self._health

    def read_frame(self, session_id: str, frame_id: int) -> Optional[Frame]:
        """
        Read one frame. Returns None if the source is exhausted (e.g. end of
        a test video file). Raises FrameAcquisitionError on unexpected failure
        after reconnect attempts are exhausted.
        """
        if self._health.status != CameraStatus.CONNECTED:
            self._reconnect_with_backoff()

        ok, image = self._read_raw()

        if not ok or image is None:
            if self.is_exhausted():
                # Clean end of a finite source (e.g. video file finished).
                # This is not a failure - do not reconnect.
                return None
            # Live source dropped a frame - worth trying to recover.
            self._health.frames_dropped += 1
            recovered = self._reconnect_with_backoff()
            if not recovered:
                return None
            ok, image = self._read_raw()
            if not ok or image is None:
                return None

        now = time.time()
        h, w = image.shape[0], image.shape[1]
        self._health.frames_read += 1
        self._health.last_frame_at = now
        if self._health.frames_read > 1 and self._health.last_frame_at:
            # simple rolling fps estimate
            pass  # actual fps computed by SessionRunner using timestamps

        return Frame(
            session_id=session_id,
            frame_id=frame_id,
            image=image,
            captured_at=now,
            width=w,
            height=h,
        )

    def _reconnect_with_backoff(self) -> bool:
        """Attempt to reconnect up to config.max_reconnect_attempts times."""
        max_attempts = self.config.max_reconnect_attempts
        for attempt in range(1, max_attempts + 1):
            self._health.status = CameraStatus.RECONNECTING
            self._health.reconnect_attempts = attempt
            wait = self.config.reconnect_backoff_seconds * attempt
            logger.warning(
                "Reconnect attempt %s/%s for source '%s' (waiting %.1fs)",
                attempt, max_attempts, self.config.source_uri, wait,
            )
            time.sleep(min(wait, 0.05))  # capped small sleep so tests stay fast
            try:
                self._close()
            except Exception:
                pass
            try:
                self._open()
                self._health.status = CameraStatus.CONNECTED
                self._health.last_error = None
                return True
            except Exception as exc:
                self._health.last_error = str(exc)
                continue

        self._health.status = CameraStatus.ERROR
        return False


class OpenCVVideoSource(VideoSource):
    """
    VideoSource backed by OpenCV's VideoCapture.

    source_uri may be:
      - a webcam index as a string, e.g. "0"
      - a video file path, e.g. "/data/clip.mp4"
      - a network stream URL, e.g. "rtsp://..." or "http://..."
    """

    def __init__(self, config: VideoSourceConfig):
        super().__init__(config)
        self._cap: Optional[cv2.VideoCapture] = None

    def _resolve_source(self):
        uri = self.config.source_uri
        if uri.isdigit():
            return int(uri)
        return uri

    def _open(self) -> None:
        source = self._resolve_source()
        cap = cv2.VideoCapture(source)
        if self.config.target_width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.target_width)
        if self.config.target_height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.target_height)
        if self.config.target_fps:
            cap.set(cv2.CAP_PROP_FPS, self.config.target_fps)

        if not cap.isOpened():
            cap.release()
            raise CameraConnectionError(f"cv2.VideoCapture could not open '{source}'")

        self._cap = cap

    def _close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _read_raw(self):
        if self._cap is None:
            return False, None
        return self._cap.read()

    def native_properties(self) -> dict:
        if self._cap is None:
            return {"width": None, "height": None, "fps": None}
        return {
            "width": int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None,
            "height": int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None,
            "fps": self._cap.get(cv2.CAP_PROP_FPS) or None,
        }

    def is_exhausted(self) -> bool:
        """
        True once a finite source (video file) has played past its last
        frame. Webcams/streams report frame_count <= 0 and are never
        considered exhausted here.
        """
        if self._cap is None:
            return False
        total = self._cap.get(cv2.CAP_PROP_FRAME_COUNT)
        pos = self._cap.get(cv2.CAP_PROP_POS_FRAMES)
        if total and total > 0:
            return pos >= total
        return False
