"""
Vision Radar - API service
Every operation the backend offers, framework-independent: authentication,
authorization, validation, orchestration. The FastAPI adapter is a thin
routing layer over this class, so all behaviour that matters is testable
without a web framework.

Spec reference: PDF Section 24-25 (FastAPI: API, session control,
configuration, events, queries), Section 28 (roles), Section 27 (API
latency is measured for every call, including ones that fail).

Known scope limit: zones and rules are read when a session STARTS; editing
them affects sessions started afterwards, not ones already running.
"""

from __future__ import annotations

import functools
import os
import time
import uuid
from dataclasses import replace
from typing import Callable, Dict, List, Optional

from vision_radar.api import serializers as ser
from vision_radar.api.errors import Conflict, NotFound, ValidationError, CameraUnavailable
from vision_radar.data_layer.database import DataIntegrityError
from vision_radar.data_layer.repository import CameraRecord, EventQuery, Repository
from vision_radar.data_layer.retention import RetentionPolicy, apply_retention
from vision_radar.api.nl_search import parse_query
from vision_radar.event_engine.models import Event
from vision_radar.ingestion.exceptions import CameraConnectionError
from vision_radar.ingestion.models import VideoSourceConfig
from vision_radar.ingestion.session import IngestionSession
from vision_radar.ingestion.video_source import OpenCVVideoSource
from vision_radar.monitoring.health import HealthRegistry
from vision_radar.monitoring.performance import PerformanceMonitor
from vision_radar.pipeline.runner import PipelineRunner, build_pipeline_runner
from vision_radar.rules_engine.models import Rule
from vision_radar.security.auth import AuthService, Permission, Role, User
from vision_radar.world_model.zone import Zone

RunnerFactory = Callable[[CameraRecord, List[Zone], List[Rule]], PipelineRunner]

DEFAULT_TRACKED_CLASSES = ["person", "car", "motorcycle", "bus", "truck"]   # Section 5: person, vehicle


def _api(permission: Permission):
    """Authenticate, authorize, run, and time the call (latency is recorded
    even when the call is rejected or fails - Section 27)."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(self, token, *args, **kwargs):
            started = time.perf_counter()
            try:
                user = self.auth.authenticate(token)
                self.auth.require(user, permission)
                return fn(self, user, *args, **kwargs)
            finally:
                self.monitor.record_api_latency((time.perf_counter() - started) * 1000.0)
        return wrapper
    return decorator


class ApiService:
    def __init__(self, repo: Repository, auth: AuthService, health: HealthRegistry,
                 monitor: PerformanceMonitor, evidence_dir: str,
                 retention: Optional[RetentionPolicy] = None,
                 model_path: Optional[str] = None,
                 runner_factory: Optional[RunnerFactory] = None):
        self.repo, self.auth, self.health, self.monitor = repo, auth, health, monitor
        self.evidence_dir = evidence_dir
        self.retention = retention or RetentionPolicy()
        self.model_path = model_path
        self._runner_factory = runner_factory or self._default_runner_factory
        self._runners: Dict[str, PipelineRunner] = {}

    # ------------------------------------------------------------------ auth
    def login(self, username: str, password: str) -> dict:
        started = time.perf_counter()
        try:
            token = self.auth.login(username, password)
            user = self.auth.authenticate(token)
            return {"token": token, "token_type": "bearer", "username": user.username, "role": user.role.value}
        finally:
            self.monitor.record_api_latency((time.perf_counter() - started) * 1000.0)

    # ---------------------------------------------------------------- status
    @_api(Permission.READ)
    def get_status(self, user: User) -> dict:
        self._reap_finished_runners()
        return {
            "overall": self.health.overall().value,
            "components": {
                name: {"status": h.status.value, "message": h.message, "detail": h.detail,
                       "stale": h.stale, "updated_at": h.updated_at}
                for name, h in self.health.snapshot().items()
            },
            "performance": self.monitor.snapshot(),
            "active_sessions": sorted(self._runners.keys()),
        }

    # ---------------------------------------------------------------- search
    @_api(Permission.READ)
    def search_events(self, user: User, text: str) -> dict:
        """Section 19: natural-language search, answered strictly from
        stored events (parse_query never invents a filter; a fragment it
        could not map to anything is returned in `ignored`, not silently
        dropped)."""
        zones = self.repo.list_zones()
        parsed = parse_query(text or "", zones, now=time.time())

        if not parsed.entity_types:
            events = self.repo.query_events(parsed.query)
        else:
            merged: Dict[str, Event] = {}
            for entity_type in parsed.entity_types:
                for e in self.repo.query_events(replace(parsed.query, entity_type=entity_type)):
                    merged[e.event_id] = e
            events = sorted(merged.values(), key=lambda e: e.start_time, reverse=True)[:parsed.query.limit]

        return {
            "interpreted_as": {
                "event_type": parsed.query.event_type.value if parsed.query.event_type else None,
                "entity_types": parsed.entity_types or None,
                "zone_id": parsed.query.zone_id,
                "start_time_from": parsed.query.start_time_from,
                "start_time_to": parsed.query.start_time_to,
                "min_confidence": parsed.query.min_confidence,
            },
            "ignored": parsed.ignored_fragments,
            "results": [ser.event_to_dict(e) for e in events],
        }

    # ---------------------------------------------------------------- events
    @_api(Permission.READ)
    def list_events(self, user: User, query: Optional[EventQuery] = None) -> List[dict]:
        return [ser.event_to_dict(e) for e in self.repo.query_events(query or EventQuery())]

    @_api(Permission.READ)
    def get_event(self, user: User, event_id: str) -> dict:
        event = self.repo.get_event(event_id)
        if event is None:
            raise NotFound(f"Event '{event_id}' not found")
        return ser.event_to_dict(event, self.repo.get_evidence(event_id))

    @_api(Permission.READ)
    def get_latest_frame(self, user: User, session_id: str) -> bytes:
        """Section 17-B: the live camera view. Only the actual process
        running that session holds a frame - if it already finished, there
        is nothing live left to show (Section 32: stale data must never be
        presented as live, so this fails rather than returning an old one)."""
        runner = self._runners.get(session_id)
        if runner is None or not runner.is_running():
            raise NotFound(f"No running session '{session_id}'")
        frame = runner.latest_frame_jpeg()
        if frame is None:
            raise NotFound("No frame has been processed yet for this session")
        return frame

    @_api(Permission.READ)
    def get_evidence_image_path(self, user: User, event_id: str) -> str:
        """Server path of an event's evidence image, for the web layer to
        stream. Refuses anything that does not resolve inside the evidence
        directory, so a tampered database row cannot be used to read
        arbitrary files."""
        evidence = self.repo.get_evidence(event_id)
        if evidence is None or not evidence.frame_snapshot_path:
            raise NotFound("No evidence image for this event")
        root = os.path.realpath(self.evidence_dir)
        target = os.path.realpath(evidence.frame_snapshot_path)
        if not target.startswith(root + os.sep) or not os.path.isfile(target):
            raise NotFound("No evidence image for this event")
        return target

    @_api(Permission.READ)
    def list_alerts(self, user: User, limit: int = 100, since: Optional[float] = None) -> List[dict]:
        return [ser.alert_to_dict(a) for a in self.repo.list_alerts(limit=limit, since=since)]

    @_api(Permission.READ)
    def list_entities(self, user: User, session_id: Optional[str] = None, limit: int = 100) -> List[dict]:
        return self.repo.list_entities(session_id=session_id, limit=limit)

    @_api(Permission.READ)
    def list_sessions(self, user: User, limit: int = 100) -> List[dict]:
        return self.repo.list_sessions(limit=limit)

    # --------------------------------------------------------------- cameras
    @_api(Permission.READ)
    def list_cameras(self, user: User) -> List[dict]:
        return [ser.camera_to_dict(c) for c in self.repo.list_cameras()]

    @_api(Permission.MANAGE_CONFIG)
    def create_camera(self, user: User, name: str, source_uri: str, frame_width: Optional[int] = None,
                      frame_height: Optional[int] = None, orientation: str = "landscape") -> dict:
        if not (name or "").strip():
            raise ValidationError("Camera name must not be empty")
        if not (source_uri or "").strip():
            raise ValidationError("Camera source_uri must not be empty")
        if orientation not in ("landscape", "portrait"):
            raise ValidationError("orientation must be 'landscape' or 'portrait'")
        for label, v in (("frame_width", frame_width), ("frame_height", frame_height)):
            if v is not None and (not isinstance(v, int) or isinstance(v, bool) or v <= 0):
                raise ValidationError(f"{label} must be a positive integer")
        cam = CameraRecord(str(uuid.uuid4()), name.strip(), source_uri.strip(), frame_width, frame_height, orientation)
        self.repo.add_camera(cam)
        return ser.camera_to_dict(cam)

    @_api(Permission.MANAGE_CONFIG)
    def delete_camera(self, user: User, camera_id: str) -> None:
        if camera_id in self._active_camera_ids():
            raise Conflict("Camera has a running session; stop it first")
        if not self.repo.delete_camera(camera_id):
            raise NotFound(f"Camera '{camera_id}' not found")

    # ----------------------------------------------------------------- zones
    @_api(Permission.READ)
    def list_zones(self, user: User, camera_id: Optional[str] = None) -> List[dict]:
        return [ser.zone_to_dict(z) for z in self.repo.list_zones(camera_id)]

    @_api(Permission.MANAGE_CONFIG)
    def create_zone(self, user: User, name: str, polygon: list, camera_id: Optional[str] = None) -> dict:
        if not (name or "").strip():
            raise ValidationError("Zone name must not be empty")
        try:
            points = [(float(x), float(y)) for x, y in polygon]
            zone = Zone.from_polygon(name.strip(), points)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"Invalid zone polygon: {exc}")
        if camera_id is not None and self.repo.get_camera(camera_id) is None:
            raise ValidationError(f"Camera '{camera_id}' does not exist")
        self.repo.save_zone(zone, camera_id)
        return ser.zone_to_dict(zone)

    @_api(Permission.MANAGE_CONFIG)
    def delete_zone(self, user: User, zone_id: str) -> None:
        if not self.repo.delete_zone(zone_id):
            raise NotFound(f"Zone '{zone_id}' not found")

    # ----------------------------------------------------------------- rules
    @_api(Permission.READ)
    def list_rules(self, user: User) -> List[dict]:
        return [ser.rule_to_dict(r) for r in self.repo.list_rules()]

    @_api(Permission.MANAGE_CONFIG)
    def create_rule(self, user: User, name: str, definition: dict) -> dict:
        try:
            rule = Rule.from_definition(name, definition)
        except ValueError as exc:
            raise ValidationError(str(exc))
        zone_id = rule.when.zone_id
        if zone_id is not None and self.repo.get_zone(zone_id) is None:
            raise ValidationError(f"Rule refers to zone '{zone_id}', which does not exist")
        self.repo.save_rule(rule)
        return ser.rule_to_dict(rule)

    @_api(Permission.MANAGE_CONFIG)
    def delete_rule(self, user: User, rule_id: str) -> None:
        if not self.repo.delete_rule(rule_id):
            raise NotFound(f"Rule '{rule_id}' not found")

    # -------------------------------------------------------------- sessions
    @_api(Permission.CONTROL_SESSIONS)
    def start_session(self, user: User, camera_id: str) -> dict:
        camera = self.repo.get_camera(camera_id)
        if camera is None:
            raise NotFound(f"Camera '{camera_id}' not found")
        self._reap_finished_runners()
        if camera_id in self._active_camera_ids():
            raise Conflict("This camera already has a running session")
        runner = self._runner_factory(camera, self.repo.list_zones(camera_id), self.repo.list_rules())
        try:
            runner.start_background()
        except CameraConnectionError as exc:
            raise CameraUnavailable(f"CAMERA OFFLINE: {exc}")
        self._runners[runner.session_id] = runner
        return {"session_id": runner.session_id, "camera_id": camera_id, "state": "RUNNING"}

    @_api(Permission.CONTROL_SESSIONS)
    def stop_session(self, user: User, session_id: str) -> dict:
        runner = self._runners.get(session_id)
        if runner is None:
            raise NotFound(f"No running session '{session_id}'")
        runner.stop()
        self._runners.pop(session_id, None)
        return {"session_id": session_id, "state": "STOPPED"}

    def _reap_finished_runners(self) -> None:
        for sid in [s for s, r in self._runners.items() if not r.is_running()]:
            self._runners.pop(sid, None)

    def _active_camera_ids(self) -> set:
        return {r.camera_id for r in self._runners.values() if r.camera_id is not None}

    def shutdown(self) -> None:
        """Stop every running session (call on process exit)."""
        for runner in list(self._runners.values()):
            runner.stop()
        self._runners.clear()

    # ----------------------------------------------------------------- users
    @_api(Permission.MANAGE_USERS)
    def create_user(self, user: User, username: str, password: str, role: str) -> dict:
        try:
            created = self.auth.create_user(username, password, Role(role))
        except ValueError as exc:
            raise ValidationError(str(exc))
        return {"user_id": created.user_id, "username": created.username, "role": created.role.value}

    @_api(Permission.MANAGE_USERS)
    def list_users(self, user: User) -> List[dict]:
        return self.repo.list_users()

    # ------------------------------------------------------------- retention
    @_api(Permission.MANAGE_CONFIG)
    def run_retention(self, user: User) -> dict:
        report = apply_retention(self.repo, self.retention, self.evidence_dir)
        result = {"events_deleted": report.events_deleted,
                  "evidence_images_deleted": report.evidence_images_deleted,
                  "images_skipped_outside_evidence_dir": report.images_skipped_outside_evidence_dir}
        self.repo.set_state("last_retention_run", {"at": time.time(), **result})
        return result

    # -------------------------------------------------------- default runner
    def _default_runner_factory(self, camera: CameraRecord, zones: List[Zone], rules: List[Rule]) -> PipelineRunner:
        if not self.model_path:
            raise CameraUnavailable("No detection model configured (model_path); cannot start a session")
        from vision_radar.vision.onnx_yolo_detector import YoloOnnxDetector
        detector = YoloOnnxDetector(self.model_path, class_filter=DEFAULT_TRACKED_CLASSES, conf_threshold=0.4)
        source = OpenCVVideoSource(VideoSourceConfig(source_uri=camera.source_uri))
        session = IngestionSession(source, camera_uri=camera.source_uri)
        return build_pipeline_runner(
            session=session, detectors=[detector], repository=self.repo, monitor=self.monitor,
            health=self.health, evidence_dir=self.evidence_dir, zones=zones, rules=rules,
            camera_id=camera.camera_id)
