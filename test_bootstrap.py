import os
import shutil
import tempfile
import unittest

from vision_radar.api.bootstrap import build_service
from vision_radar.security.auth import AuthenticationError

SECRET = "s" * 40


class TestBootstrap(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.env = {
            "VISION_RADAR_DB": os.path.join(self.tmp, "vr.sqlite"),
            "VISION_RADAR_EVIDENCE_DIR": os.path.join(self.tmp, "evidence"),
            "VISION_RADAR_JWT_SECRET": SECRET,
        }

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_builds_a_working_service_and_creates_the_evidence_directory(self):
        self.env.update(VISION_RADAR_ADMIN_USERNAME="root", VISION_RADAR_ADMIN_PASSWORD="bootstrap-pw-1")
        svc = build_service(self.env)
        self.assertTrue(os.path.isdir(self.env["VISION_RADAR_EVIDENCE_DIR"]))
        token = svc.login("root", "bootstrap-pw-1")["token"]
        self.assertEqual(svc.get_status(token)["components"]["database"]["status"], "OK")

    def test_initial_admin_is_created_once_only(self):
        self.env.update(VISION_RADAR_ADMIN_USERNAME="root", VISION_RADAR_ADMIN_PASSWORD="bootstrap-pw-1")
        build_service(self.env)
        self.env.update(VISION_RADAR_ADMIN_USERNAME="usurper", VISION_RADAR_ADMIN_PASSWORD="another-pw-22")
        svc = build_service(self.env)                        # same DB, second start
        self.assertEqual(svc.repo.user_count(), 1)
        with self.assertRaises(AuthenticationError):
            svc.login("usurper", "another-pw-22")

    def test_no_admin_variables_means_no_users_and_a_warning_not_a_default_login(self):
        with self.assertLogs("vision_radar.bootstrap", level="WARNING") as logs:
            svc = build_service(self.env)
        self.assertEqual(svc.repo.user_count(), 0)
        self.assertTrue(any("nobody will be able to log in" in m for m in logs.output))
        with self.assertRaises(AuthenticationError):
            svc.login("admin", "admin")

    def test_missing_jwt_secret_warns_and_short_secret_is_refused(self):
        del self.env["VISION_RADAR_JWT_SECRET"]
        with self.assertLogs("vision_radar.bootstrap", level="WARNING") as logs:
            build_service(self.env)
        self.assertTrue(any("VISION_RADAR_JWT_SECRET" in m for m in logs.output))
        self.env["VISION_RADAR_JWT_SECRET"] = "too-short"
        with self.assertRaises(ValueError):
            build_service(self.env)

    def test_bad_numeric_settings_are_rejected_with_the_variable_name(self):
        for key in ("VISION_RADAR_EVENT_RETENTION_DAYS", "VISION_RADAR_EVIDENCE_RETENTION_DAYS",
                    "VISION_RADAR_TOKEN_TTL_SECONDS"):
            for bad in ("abc", "0", "-3"):
                env = dict(self.env, **{key: bad})
                with self.assertRaises(ValueError, msg=f"{key}={bad}") as ctx:
                    build_service(env)
                self.assertIn(key, str(ctx.exception))

    def test_settings_are_applied(self):
        self.env.update(VISION_RADAR_EVENT_RETENTION_DAYS="12", VISION_RADAR_EVIDENCE_RETENTION_DAYS="2.5",
                        VISION_RADAR_MODEL_PATH="/models/yolov8n.onnx")
        svc = build_service(self.env)
        self.assertEqual((svc.retention.event_retention_days, svc.retention.evidence_image_retention_days), (12.0, 2.5))
        self.assertEqual(svc.model_path, "/models/yolov8n.onnx")

    def test_configuration_persists_across_a_restart(self):
        # Section 32 (Data): configurations persist.
        self.env.update(VISION_RADAR_ADMIN_USERNAME="root", VISION_RADAR_ADMIN_PASSWORD="bootstrap-pw-1")
        svc1 = build_service(self.env)
        tok = svc1.login("root", "bootstrap-pw-1")["token"]
        cam = svc1.create_camera(tok, "Gate", "rtsp://cam")
        svc1.create_zone(tok, "Restricted", [[0, 0], [10, 0], [10, 10]], camera_id=cam["camera_id"])
        svc1.repo.db.close()

        svc2 = build_service(self.env)
        tok2 = svc2.login("root", "bootstrap-pw-1")["token"]
        self.assertEqual([c["name"] for c in svc2.list_cameras(tok2)], ["Gate"])
        self.assertEqual([z["name"] for z in svc2.list_zones(tok2)], ["Restricted"])


if __name__ == "__main__":
    unittest.main()
