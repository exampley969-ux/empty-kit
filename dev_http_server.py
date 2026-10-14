"""
Vision Radar - DEV-ONLY HTTP adapter (stdlib only, no FastAPI)

This is NOT the production API. `api/fastapi_app.py` is the real deliverable
adapter (Section 24-25 asks for FastAPI specifically); this file exists only
because FastAPI cannot be installed in the sandbox this project was built in
(no network access - see PHASE9_README's "What was NOT executed"), and the
Phase 10 frontend genuinely needs a running backend to be tested against.

It maps the exact same routes as fastapi_app.py onto the exact same
ApiService, using only Python's standard library (http.server, json).
Nothing here re-implements business logic - every request is a straight
call into ApiService, same as the FastAPI adapter. When FastAPI is
available, replace this with fastapi_app.py; no frontend code should need
to change, since both expose the same JSON routes.
"""

from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from vision_radar.api.errors import public_message, status_for
from vision_radar.api.service import ApiService
from vision_radar.data_layer.repository import EventQuery
from vision_radar.event_engine.models import EventState, EventType

ROUTES = []  # (method, regex, handler_name)


def route(method, pattern):
    regex = re.compile("^" + pattern + "$")
    def deco(fn):
        ROUTES.append((method, regex, fn))
        return fn
    return deco


def _enum(cls, value):
    if value is None:
        return None
    try:
        return cls(value)
    except ValueError:
        from vision_radar.api.errors import ValidationError
        raise ValidationError(f"Unknown value '{value}' for {cls.__name__}")


class Handler(BaseHTTPRequestHandler):
    service: ApiService = None  # set by make_server
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass  # keep test/dev output quiet

    def _token(self):
        auth = self.headers.get("Authorization", "")
        parts = auth.split(None, 1)
        return parts[1].strip() if len(parts) == 2 and parts[0].lower() == "bearer" else None

    def _send(self, status, body):
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")     # dev-only CORS: any origin
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_file(self, path, content_type):
        with open(path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_bytes(self, data, content_type):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _dispatch(self, method):
        parsed = urlparse(self.path)
        for m, regex, fn in ROUTES:
            if m != method:
                continue
            match = regex.match(parsed.path)
            if match:
                query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
                body = {}
                length = int(self.headers.get("Content-Length", 0))
                if length:
                    body = json.loads(self.rfile.read(length) or b"{}")
                try:
                    fn(self, self._token(), match.groupdict(), query, body)
                except Exception as exc:                          # noqa: BLE001
                    self._send(status_for(exc), {"detail": public_message(exc)})
                return
        self._send(404, {"detail": "Not found"})

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.end_headers()

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_DELETE(self):
        self._dispatch("DELETE")


@route("POST", r"/auth/login")
def _login(h, token, params, query, body):
    h._send(200, h.service.login(body["username"], body["password"]))


@route("GET", r"/status")
def _status(h, token, params, query, body):
    h._send(200, h.service.get_status(token))


@route("GET", r"/search")
def _search(h, token, params, query, body):
    h._send(200, h.service.search_events(token, query.get("q", "")))


@route("GET", r"/events")
def _events(h, token, params, query, body):
    q = EventQuery(
        start_time_from=float(query["start_time_from"]) if "start_time_from" in query else None,
        start_time_to=float(query["start_time_to"]) if "start_time_to" in query else None,
        event_type=_enum(EventType, query.get("event_type")),
        entity_id=query.get("entity_id"), zone_id=query.get("zone_id"),
        min_confidence=float(query["min_confidence"]) if "min_confidence" in query else None,
        state=_enum(EventState, query.get("state")),
        limit=int(query.get("limit", 100)), offset=int(query.get("offset", 0)))
    h._send(200, h.service.list_events(token, q))


@route("GET", r"/events/(?P<event_id>[^/]+)")
def _event(h, token, params, query, body):
    h._send(200, h.service.get_event(token, params["event_id"]))


@route("GET", r"/events/(?P<event_id>[^/]+)/snapshot")
def _snapshot(h, token, params, query, body):
    path = h.service.get_evidence_image_path(token, params["event_id"])
    h._send_file(path, "image/jpeg")


@route("GET", r"/sessions/(?P<session_id>[^/]+)/frame")
def _latest_frame(h, token, params, query, body):
    h._send_bytes(h.service.get_latest_frame(token, params["session_id"]), "image/jpeg")


@route("GET", r"/alerts")
def _alerts(h, token, params, query, body):
    h._send(200, h.service.list_alerts(token, limit=int(query.get("limit", 100)),
                                       since=float(query["since"]) if "since" in query else None))


@route("GET", r"/entities")
def _entities(h, token, params, query, body):
    h._send(200, h.service.list_entities(token, session_id=query.get("session_id"),
                                         limit=int(query.get("limit", 100))))


@route("GET", r"/sessions")
def _sessions(h, token, params, query, body):
    h._send(200, h.service.list_sessions(token, limit=int(query.get("limit", 100))))


@route("POST", r"/sessions/start")
def _start(h, token, params, query, body):
    h._send(200, h.service.start_session(token, body["camera_id"]))


@route("POST", r"/sessions/(?P<session_id>[^/]+)/stop")
def _stop(h, token, params, query, body):
    h._send(200, h.service.stop_session(token, params["session_id"]))


@route("GET", r"/cameras")
def _cameras(h, token, params, query, body):
    h._send(200, h.service.list_cameras(token))


@route("POST", r"/cameras")
def _create_camera(h, token, params, query, body):
    h._send(200, h.service.create_camera(token, body["name"], body["source_uri"],
                                         body.get("frame_width"), body.get("frame_height"),
                                         body.get("orientation", "landscape")))


@route("DELETE", r"/cameras/(?P<camera_id>[^/]+)")
def _delete_camera(h, token, params, query, body):
    h.service.delete_camera(token, params["camera_id"])
    h._send(204, {})


@route("GET", r"/zones")
def _zones(h, token, params, query, body):
    h._send(200, h.service.list_zones(token, camera_id=query.get("camera_id")))


@route("POST", r"/zones")
def _create_zone(h, token, params, query, body):
    h._send(200, h.service.create_zone(token, body["name"], body["polygon"], body.get("camera_id")))


@route("DELETE", r"/zones/(?P<zone_id>[^/]+)")
def _delete_zone(h, token, params, query, body):
    h.service.delete_zone(token, params["zone_id"])
    h._send(204, {})


@route("GET", r"/rules")
def _rules(h, token, params, query, body):
    h._send(200, h.service.list_rules(token))


@route("POST", r"/rules")
def _create_rule(h, token, params, query, body):
    h._send(200, h.service.create_rule(token, body["name"], body["definition"]))


@route("DELETE", r"/rules/(?P<rule_id>[^/]+)")
def _delete_rule(h, token, params, query, body):
    h.service.delete_rule(token, params["rule_id"])
    h._send(204, {})


@route("GET", r"/users")
def _users(h, token, params, query, body):
    h._send(200, h.service.list_users(token))


@route("POST", r"/users")
def _create_user(h, token, params, query, body):
    h._send(200, h.service.create_user(token, body["username"], body["password"], body["role"]))


@route("POST", r"/retention/run")
def _retention(h, token, params, query, body):
    h._send(200, h.service.run_retention(token))


def make_server(service: ApiService, host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"service": service})
    server = ThreadingHTTPServer((host, port), handler)
    return server


def run_in_background(service: ApiService, host: str = "127.0.0.1", port: int = 0):
    server = make_server(service, host, port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread
