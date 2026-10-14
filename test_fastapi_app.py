"""
Runtime tests for the FastAPI adapter, using FastAPI's TestClient.

THESE ARE SKIPPED WHEN fastapi/httpx ARE NOT INSTALLED - which was the case
in the sandbox this project was built in (no network, so no pip install).
They were written but NOT EXECUTED there. Run them where FastAPI exists:

    pip install fastapi httpx
    python -m unittest vision_radar.tests.test_fastapi_app -v

The static checks in test_fastapi_static.py, which do run everywhere, cover
the adapter's wiring (method names/arguments, no unauthenticated routes);
everything behavioural lives in ApiService and is tested in
test_api_service.py.
"""
import importlib.util
import unittest

from vision_radar.tests.test_api_service import ApiCase, RECT

HAVE_FASTAPI = all(importlib.util.find_spec(m) for m in ("fastapi", "httpx"))


@unittest.skipUnless(HAVE_FASTAPI, "fastapi + httpx not installed in this environment")
class TestFastApiApp(ApiCase):
    def setUp(self):
        super().setUp()
        from fastapi.testclient import TestClient
        from vision_radar.api.fastapi_app import create_app
        self.app = create_app(self.svc)
        self.client = TestClient(self.app, raise_server_exceptions=False)

    @staticmethod
    def auth(token):
        return {"Authorization": f"Bearer {token}"}

    # ---- authentication ----
    def test_login_ok_and_bad_credentials(self):
        ok = self.client.post("/auth/login", json={"username": "admin", "password": "password-123"})
        self.assertEqual(ok.status_code, 200)
        self.assertIn("token", ok.json())
        bad = self.client.post("/auth/login", json={"username": "admin", "password": "nope-nope-1"})
        self.assertEqual(bad.status_code, 401)
        self.assertEqual(bad.headers.get("www-authenticate"), "Bearer")

    def test_protected_routes_reject_missing_malformed_and_bad_tokens(self):
        for headers in ({}, {"Authorization": "Basic abc"}, {"Authorization": "Bearer"},
                        {"Authorization": "Bearer not.a.token"}):
            r = self.client.get("/status", headers=headers)
            self.assertEqual(r.status_code, 401, headers)

    def test_status_with_a_valid_token(self):
        r = self.client.get("/status", headers=self.auth(self.view))
        self.assertEqual(r.status_code, 200)
        self.assertIn("components", r.json())

    # ---- authorization ----
    def test_viewer_cannot_create_zone_admin_can(self):
        body = {"name": "Restricted", "polygon": RECT}
        self.assertEqual(self.client.post("/zones", json=body, headers=self.auth(self.view)).status_code, 403)
        self.assertEqual(self.client.post("/zones", json=body, headers=self.auth(self.oper)).status_code, 403)
        ok = self.client.post("/zones", json=body, headers=self.auth(self.admin))
        self.assertEqual(ok.status_code, 200)
        self.assertEqual(ok.json()["name"], "Restricted")

    def test_operator_can_start_sessions_viewer_cannot(self):
        cam = self.client.post("/cameras", json={"name": "Gate", "source_uri": "fake://0"},
                               headers=self.auth(self.admin)).json()
        start = {"camera_id": cam["camera_id"]}
        self.assertEqual(self.client.post("/sessions/start", json=start, headers=self.auth(self.view)).status_code, 403)
        self.assertEqual(self.client.post("/sessions/start", json=start, headers=self.auth(self.oper)).status_code, 200)

    # ---- validation and error mapping ----
    def test_unknown_enum_filter_is_422(self):
        r = self.client.get("/events?event_type=BOGUS", headers=self.auth(self.view))
        self.assertEqual(r.status_code, 422)

    def test_out_of_range_paging_is_422(self):
        self.assertEqual(self.client.get("/events?limit=0", headers=self.auth(self.view)).status_code, 422)
        self.assertEqual(self.client.get("/events?limit=100000", headers=self.auth(self.view)).status_code, 422)

    def test_malformed_body_is_422_and_never_echoes_submitted_values(self):
        r = self.client.post("/auth/login", json={"username": "admin", "passwd": "SUPER-SECRET-VALUE"})
        self.assertEqual(r.status_code, 422)
        self.assertNotIn("SUPER-SECRET-VALUE", r.text)

    def test_not_found_and_conflict_and_business_validation(self):
        self.assertEqual(self.client.get("/events/ghost", headers=self.auth(self.view)).status_code, 404)
        bad_zone = {"name": "Z", "polygon": [[0, 0], [1, 1]]}
        self.assertEqual(self.client.post("/zones", json=bad_zone, headers=self.auth(self.admin)).status_code, 422)

    def test_delete_returns_204_with_no_body(self):
        cam = self.client.post("/cameras", json={"name": "Gate", "source_uri": "0"},
                               headers=self.auth(self.admin)).json()
        r = self.client.delete(f"/cameras/{cam['camera_id']}", headers=self.auth(self.admin))
        self.assertEqual(r.status_code, 204)
        self.assertEqual(r.content, b"")

    def test_unexpected_errors_return_a_generic_500_without_internals(self):
        def boom(token):
            raise RuntimeError("secret internals /etc/shadow")
        self.svc.get_status = boom
        r = self.client.get("/status", headers=self.auth(self.view))
        self.assertEqual(r.status_code, 500)
        self.assertNotIn("secret", r.text)
        self.assertNotIn("shadow", r.text)

    # ---- a whole session through HTTP, including the evidence image ----
    def test_full_flow_over_http_including_snapshot_download(self):
        h = self.auth(self.admin)
        cam = self.client.post("/cameras", json={"name": "Gate", "source_uri": "fake://0"}, headers=h).json()
        zone = self.client.post("/zones", json={"name": "Restricted", "polygon": RECT,
                                                "camera_id": cam["camera_id"]}, headers=h).json()
        self.client.post("/rules", headers=h, json={"name": "Entry", "definition": {
            "when": {"event_type": "ZONE_ENTRY", "entity_type": "person", "zone_id": zone["zone_id"]}}})

        started = self.client.post("/sessions/start", json={"camera_id": cam["camera_id"]},
                                   headers=self.auth(self.oper))
        self.assertEqual(started.status_code, 200)
        self.wait_until_finished()

        events = self.client.get("/events?event_type=ZONE_ENTRY", headers=self.auth(self.view)).json()
        self.assertEqual(len(events), 1)
        alerts = self.client.get("/alerts", headers=self.auth(self.view)).json()
        self.assertEqual(len(alerts), 1)

        snap = self.client.get(f"/events/{events[0]['event_id']}/snapshot", headers=self.auth(self.view))
        self.assertEqual(snap.status_code, 200)
        self.assertEqual(snap.headers["content-type"], "image/jpeg")
        self.assertEqual(snap.content[:2], b"\xff\xd8")                       # JPEG magic bytes

    # ---- docs exposure is configurable ----
    def test_docs_can_be_disabled(self):
        from fastapi.testclient import TestClient
        from vision_radar.api.fastapi_app import create_app
        closed = TestClient(create_app(self.svc, enable_docs=False), raise_server_exceptions=False)
        for path in ("/docs", "/redoc", "/openapi.json"):
            self.assertEqual(closed.get(path).status_code, 404, path)
        self.assertEqual(self.client.get("/openapi.json").status_code, 200)


if __name__ == "__main__":
    unittest.main()
