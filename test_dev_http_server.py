"""
Tests for the DEV-ONLY stdlib HTTP adapter, using REAL HTTP requests over a
REAL socket (not an in-process test client) - and a static check that its
route set matches fastapi_app.py's, so the two adapters cannot silently
drift apart while FastAPI remains uninstallable in this sandbox.
"""
import ast
import os
import re
import shutil
import tempfile
import time
import unittest

import requests

from vision_radar.api.dev_http_server import run_in_background
from vision_radar.tests.test_api_service import ApiCase, RECT


class TestDevServerRealHttp(ApiCase):
    def setUp(self):
        super().setUp()
        self.server, self.thread = run_in_background(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        super().tearDown()

    def _bearer(self, token):
        return {"Authorization": f"Bearer {token}"}

    def test_login_over_real_socket(self):
        r = requests.post(f"{self.base}/auth/login", json={"username": "admin", "password": "password-123"})
        self.assertEqual(r.status_code, 200)
        self.assertIn("token", r.json())

    def test_bad_login_is_401(self):
        r = requests.post(f"{self.base}/auth/login", json={"username": "admin", "password": "wrong"})
        self.assertEqual(r.status_code, 401)

    def test_missing_token_is_401_on_a_protected_route(self):
        self.assertEqual(requests.get(f"{self.base}/status").status_code, 401)

    def test_viewer_forbidden_from_creating_a_zone_admin_allowed(self):
        body = {"name": "Restricted", "polygon": RECT}
        self.assertEqual(requests.post(f"{self.base}/zones", json=body, headers=self._bearer(self.view)).status_code, 403)
        r = requests.post(f"{self.base}/zones", json=body, headers=self._bearer(self.admin))
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["name"], "Restricted")

    def test_cors_preflight_and_header_present(self):
        opt = requests.options(f"{self.base}/status")
        self.assertEqual(opt.status_code, 204)
        got = requests.get(f"{self.base}/status", headers=self._bearer(self.view))
        self.assertEqual(got.headers.get("Access-Control-Allow-Origin"), "*")

    def test_full_session_flow_over_real_http_including_binary_jpeg_download(self):
        cam = requests.post(f"{self.base}/cameras", json={"name": "Gate", "source_uri": "fake://0"},
                            headers=self._bearer(self.admin)).json()
        zone = requests.post(f"{self.base}/zones", json={"name": "R", "polygon": RECT, "camera_id": cam["camera_id"]},
                             headers=self._bearer(self.admin)).json()
        requests.post(f"{self.base}/rules", headers=self._bearer(self.admin), json={
            "name": "Entry", "definition": {"when": {"event_type": "ZONE_ENTRY", "entity_type": "person",
                                                      "zone_id": zone["zone_id"]}}})
        start = requests.post(f"{self.base}/sessions/start", json={"camera_id": cam["camera_id"]},
                              headers=self._bearer(self.oper))
        self.assertEqual(start.status_code, 200)
        self.wait_until_finished()

        events = requests.get(f"{self.base}/events?event_type=ZONE_ENTRY", headers=self._bearer(self.view)).json()
        self.assertEqual(len(events), 1)
        alerts = requests.get(f"{self.base}/alerts", headers=self._bearer(self.view)).json()
        self.assertEqual(len(alerts), 1)

        snap = requests.get(f"{self.base}/events/{events[0]['event_id']}/snapshot", headers=self._bearer(self.view))
        self.assertEqual(snap.status_code, 200)
        self.assertEqual(snap.headers["Content-Type"], "image/jpeg")
        self.assertEqual(snap.content[:2], b"\xff\xd8")

    def test_delete_returns_204(self):
        cam = requests.post(f"{self.base}/cameras", json={"name": "G", "source_uri": "0"},
                            headers=self._bearer(self.admin)).json()
        r = requests.delete(f"{self.base}/cameras/{cam['camera_id']}", headers=self._bearer(self.admin))
        self.assertEqual(r.status_code, 204)

    def test_not_found_route_is_404(self):
        self.assertEqual(requests.get(f"{self.base}/nonexistent").status_code, 404)


def _extract_route_paths(py_source: str, prefix_quote: str) -> set:
    """Grab each `@app.<method>("<path>")` / `@route("<METHOD>", r"<path>")`
    path string and normalise FastAPI's `{name}` params to a common form."""
    paths = set()
    for m in re.finditer(r'@(?:app\.(?:get|post|put|patch|delete)|route)\(\s*(?:"[A-Z]+"\s*,\s*)?'
                         r'r?["\']([^"\']+)["\']', py_source):
        norm = re.sub(r"\{[^}]+\}|\(\?P<[^>]+>[^)]+\)", "{}", m.group(1))
        paths.add(norm)
    return paths


class TestDevServerMatchesFastApiRoutes(unittest.TestCase):
    def test_same_route_paths_in_both_adapters(self):
        here = os.path.dirname(__file__)
        with open(os.path.join(here, "..", "api", "fastapi_app.py")) as f:
            fastapi_src = f.read()
        with open(os.path.join(here, "..", "api", "dev_http_server.py")) as f:
            dev_src = f.read()
        fastapi_routes = _extract_route_paths(fastapi_src, '"')
        dev_routes = _extract_route_paths(dev_src, 'r"')
        self.assertGreater(len(fastapi_routes), 15)
        self.assertEqual(fastapi_routes, dev_routes)


if __name__ == "__main__":
    unittest.main()
