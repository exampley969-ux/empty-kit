"""Vision Radar - Video Ingestion Layer exceptions."""


class VideoSourceError(Exception):
    """Base exception for all video source errors."""


class CameraConnectionError(VideoSourceError):
    """Raised when a camera/source cannot be opened or reconnected."""


class FrameAcquisitionError(VideoSourceError):
    """Raised when a frame cannot be read from an otherwise-connected source."""


class SessionError(Exception):
    """Raised for invalid session lifecycle operations (e.g. stopping a session that never started)."""
