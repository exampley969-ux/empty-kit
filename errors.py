"""
Vision Radar - API errors
One place mapping every failure the service can raise to an HTTP status, so
the web adapter has no logic of its own to get wrong.
"""

from __future__ import annotations

from vision_radar.data_layer.database import DataIntegrityError, DataStorageUnavailable
from vision_radar.security.auth import AuthenticationError, PermissionDenied


class ApiError(Exception):
    status_code = 500


class NotFound(ApiError):
    status_code = 404


class ValidationError(ApiError):
    status_code = 422


class Conflict(ApiError):
    status_code = 409


class CameraUnavailable(ApiError):
    status_code = 503


def status_for(exc: Exception) -> int:
    if isinstance(exc, ApiError):
        return exc.status_code
    if isinstance(exc, AuthenticationError):
        return 401
    if isinstance(exc, PermissionDenied):
        return 403
    if isinstance(exc, DataStorageUnavailable):
        return 503
    if isinstance(exc, DataIntegrityError):
        return 422
    if isinstance(exc, ValueError):
        return 422
    return 500


def public_message(exc: Exception) -> str:
    """Text safe to return to a client. Storage failures and unknown errors
    are reported by their spec'd/generic name only - never with internal
    detail (SQL text, file paths, stack traces)."""
    if isinstance(exc, DataStorageUnavailable):
        return "DATA STORAGE UNAVAILABLE"
    if isinstance(exc, DataIntegrityError):
        return "The request conflicts with existing data"
    if isinstance(exc, (ApiError, AuthenticationError, PermissionDenied, ValueError)):
        return str(exc)
    return "Internal server error"
