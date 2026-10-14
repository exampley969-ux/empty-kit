"""
Vision Radar - FastAPI adapter
A deliberately thin routing layer: every route is one call into ApiService,
which holds all the behaviour (auth, validation, orchestration) and is fully
tested without a web framework. This file only maps HTTP <-> service calls.

Run (after `pip install fastapi uvicorn`):
    uvicorn vision_radar.api.fastapi_app:create_app_from_env --factory

Deliberately NOT included here: CORS. Which browser origin may call the API
is a deployment decision (the React dashboard's URL); add
`fastapi.middleware.cors.CORSMiddleware` with that explicit origin.
"""

from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import Depends, FastAPI, Header, Query, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from vision_radar.api.errors import ApiError, ValidationError, public_message, status_for
from vision_radar.api.service import ApiService
from vision_radar.data_layer.database import DataIntegrityError, DataStorageUnavailable
from vision_radar.data_layer.repository import EventQuery
from vision_radar.event_engine.models import EventState, EventType
from vision_radar.security.auth import AuthenticationError, PermissionDenied


class LoginRequest(BaseModel):
    username: str
    password: str


class CameraCreate(BaseModel):
    name: str
    source_uri: str
    frame_width: Optional[int] = None
    frame_height: Optional[int] = None
    orientation: str = "landscape"


class ZoneCreate(BaseModel):
    name: str
    polygon: List[List[float]]
    camera_id: Optional[str] = None


class RuleCreate(BaseModel):
    name: str
    definition: dict


class SessionStart(BaseModel):
    camera_id: str


class UserCreate(BaseModel):
    username: str
    password: str
    role: str


def _token(authorization: Optional[str] = Header(default=None)) -> Optional[str]:
    """Extract the bearer token; None (-> 401 from the service) if absent/malformed."""
    if not authorization:
        return None
    parts = authorization.split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return None


def _enum(enum_cls, value: Optional[str], name: str):
    if value is None:
        return None
    try:
        return enum_cls(value)
    except ValueError:
        raise ValidationError(f"Unknown {name} '{value}'")


def create_app(service: ApiService, enable_docs: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        yield
        service.shutdown()

    app = FastAPI(
        title="Vision Radar API", lifespan=lifespan,
        docs_url="/docs" if enable_docs else None,
        redoc_url="/redoc" if enable_docs else None,
        openapi_url="/openapi.json" if enable_docs else None,
    )

    # ---- error mapping: one handler, all logic lives in api/errors.py ----
    def _handle(request, exc):
        status = status_for(exc)
        headers = {"WWW-Authenticate": "Bearer"} if status == 401 else None
        return JSONResponse(status_code=status, content={"detail": public_message(exc)}, headers=headers)

    for exc_type in (ApiError, AuthenticationError, PermissionDenied, DataStorageUnavailable,
                     DataIntegrityError, ValueError, Exception):
        app.add_exception_handler(exc_type, _handle)

    def _bad_request(request, exc):
        # FastAPI's default 422 body echoes the submitted input (which can
        # include a password). Report only where and why, never the values.
        errors = [{"loc": [str(p) for p in e.get("loc", [])], "msg": e.get("msg", "invalid")}
                  for e in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": errors})

    app.add_exception_handler(RequestValidationError, _bad_request)

    # ---- auth ----
    @app.post("/auth/login")
    def login(body: LoginRequest):
        return service.login(body.username, body.password)

    # ---- status ----
    @app.get("/status")
    def status(token: Optional[str] = Depends(_token)):
        return service.get_status(token)

    # ---- natural-language search (Section 19) ----
    @app.get("/search")
    def search_events(token: Optional[str] = Depends(_token), q: str = Query(..., min_length=1)):
        return service.search_events(token, q)

    # ---- events / alerts / entities / sessions (read) ----
    @app.get("/events")
    def list_events(
        token: Optional[str] = Depends(_token),
        start_time_from: Optional[float] = None, start_time_to: Optional[float] = None,
        event_type: Optional[str] = None, entity_id: Optional[str] = None,
        zone_id: Optional[str] = None, min_confidence: Optional[float] = None,
        state: Optional[str] = None,
        limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0),
    ):
        query = EventQuery(
            start_time_from=start_time_from, start_time_to=start_time_to,
            event_type=_enum(EventType, event_type, "event_type"), entity_id=entity_id,
            zone_id=zone_id, min_confidence=min_confidence,
            state=_enum(EventState, state, "state"), limit=limit, offset=offset)
        return service.list_events(token, query)

    @app.get("/events/{event_id}")
    def get_event(event_id: str, token: Optional[str] = Depends(_token)):
        return service.get_event(token, event_id)

    @app.get("/events/{event_id}/snapshot")
    def get_snapshot(event_id: str, token: Optional[str] = Depends(_token)):
        return FileResponse(service.get_evidence_image_path(token, event_id), media_type="image/jpeg")

    @app.get("/sessions/{session_id}/frame")
    def get_latest_frame(session_id: str, token: Optional[str] = Depends(_token)):
        return Response(content=service.get_latest_frame(token, session_id), media_type="image/jpeg")

    @app.get("/alerts")
    def list_alerts(token: Optional[str] = Depends(_token),
                    limit: int = Query(100, ge=1, le=1000), since: Optional[float] = None):
        return service.list_alerts(token, limit=limit, since=since)

    @app.get("/entities")
    def list_entities(token: Optional[str] = Depends(_token), session_id: Optional[str] = None,
                      limit: int = Query(100, ge=1, le=1000)):
        return service.list_entities(token, session_id=session_id, limit=limit)

    @app.get("/sessions")
    def list_sessions(token: Optional[str] = Depends(_token), limit: int = Query(100, ge=1, le=1000)):
        return service.list_sessions(token, limit=limit)

    # ---- session control ----
    @app.post("/sessions/start")
    def start_session(body: SessionStart, token: Optional[str] = Depends(_token)):
        return service.start_session(token, body.camera_id)

    @app.post("/sessions/{session_id}/stop")
    def stop_session(session_id: str, token: Optional[str] = Depends(_token)):
        return service.stop_session(token, session_id)

    # ---- configuration ----
    @app.get("/cameras")
    def list_cameras(token: Optional[str] = Depends(_token)):
        return service.list_cameras(token)

    @app.post("/cameras")
    def create_camera(body: CameraCreate, token: Optional[str] = Depends(_token)):
        return service.create_camera(token, body.name, body.source_uri, body.frame_width,
                                     body.frame_height, body.orientation)

    @app.delete("/cameras/{camera_id}")
    def delete_camera(camera_id: str, token: Optional[str] = Depends(_token)):
        service.delete_camera(token, camera_id)
        return Response(status_code=204)

    @app.get("/zones")
    def list_zones(token: Optional[str] = Depends(_token), camera_id: Optional[str] = None):
        return service.list_zones(token, camera_id)

    @app.post("/zones")
    def create_zone(body: ZoneCreate, token: Optional[str] = Depends(_token)):
        return service.create_zone(token, body.name, body.polygon, body.camera_id)

    @app.delete("/zones/{zone_id}")
    def delete_zone(zone_id: str, token: Optional[str] = Depends(_token)):
        service.delete_zone(token, zone_id)
        return Response(status_code=204)

    @app.get("/rules")
    def list_rules(token: Optional[str] = Depends(_token)):
        return service.list_rules(token)

    @app.post("/rules")
    def create_rule(body: RuleCreate, token: Optional[str] = Depends(_token)):
        return service.create_rule(token, body.name, body.definition)

    @app.delete("/rules/{rule_id}")
    def delete_rule(rule_id: str, token: Optional[str] = Depends(_token)):
        service.delete_rule(token, rule_id)
        return Response(status_code=204)

    # ---- users / retention (administrator) ----
    @app.get("/users")
    def list_users(token: Optional[str] = Depends(_token)):
        return service.list_users(token)

    @app.post("/users")
    def create_user(body: UserCreate, token: Optional[str] = Depends(_token)):
        return service.create_user(token, body.username, body.password, body.role)

    @app.post("/retention/run")
    def run_retention(token: Optional[str] = Depends(_token)):
        return service.run_retention(token)

    return app


def create_app_from_env() -> FastAPI:
    """Factory for `uvicorn ... --factory`: builds the whole service from
    VISION_RADAR_* environment variables (see api/bootstrap.py)."""
    from vision_radar.api.bootstrap import build_service
    import os
    app = create_app(build_service(), enable_docs=os.environ.get("VISION_RADAR_ENABLE_DOCS", "1") != "0")
    # Optionally serve the dashboard from the same origin (no CORS needed).
    # Mounted last, so every API route above takes priority. The static "/" page
    # (no data, no API route) also serves as the Render health check / uptime-ping target,
    # which keeps the project's rule "login is the only unauthenticated API route".
    frontend_dir = os.environ.get("VISION_RADAR_FRONTEND_DIR")
    if frontend_dir and os.path.isdir(frontend_dir):
        from fastapi.staticfiles import StaticFiles
        app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
    return app
