import time
import unittest

import jwt

from vision_radar.data_layer.database import Database
from vision_radar.data_layer.repository import Repository
from vision_radar.security.auth import (
    AuthenticationError, AuthService, Permission, PermissionDenied, Role, ROLE_PERMISSIONS,
    hash_password, verify_password,
)

FAST = 1000   # low PBKDF2 iterations so the test suite stays fast; production default is 600k


def make_auth(**kw):
    repo = Repository(Database(":memory:"))
    return AuthService(repo, secret_key="test-secret-key-not-for-production-use-1234", hash_iterations=FAST, **kw), repo


class TestPasswordHashing(unittest.TestCase):
    def test_correct_password_verifies_wrong_does_not(self):
        h = hash_password("correct horse", iterations=FAST)
        self.assertTrue(verify_password("correct horse", h))
        self.assertFalse(verify_password("wrong horse", h))

    def test_hash_is_salted_and_never_contains_the_password(self):
        h1, h2 = hash_password("same-password", FAST), hash_password("same-password", FAST)
        self.assertNotEqual(h1, h2)
        self.assertNotIn("same-password", h1)

    def test_malformed_stored_hash_fails_closed(self):
        for bad in ("", "garbage", "a$b$c$d", "pbkdf2_sha256$x$zz$zz", "md5$1$00$00"):
            self.assertFalse(verify_password("anything", bad), bad)

    def test_production_default_iteration_count_is_strong(self):
        self.assertGreaterEqual(hash_password("x").split("$")[1] and int(hash_password("x").split("$")[1]), 600_000)


class TestUserCreation(unittest.TestCase):
    def test_create_and_login(self):
        auth, repo = make_auth()
        user = auth.create_user("alice", "longenough1", Role.OPERATOR)
        self.assertEqual(user.role, Role.OPERATOR)
        stored = repo.get_user_by_username("alice")["password_hash"]
        self.assertNotIn("longenough1", stored)

    def test_short_password_rejected(self):
        auth, _ = make_auth()
        with self.assertRaises(ValueError):
            auth.create_user("bob", "short", Role.VIEWER)

    def test_blank_username_rejected(self):
        auth, _ = make_auth()
        with self.assertRaises(ValueError):
            auth.create_user("   ", "longenough1", Role.VIEWER)

    def test_duplicate_username_rejected(self):
        auth, _ = make_auth()
        auth.create_user("alice", "longenough1", Role.VIEWER)
        with self.assertRaises(ValueError):
            auth.create_user("alice", "another-long-pw", Role.ADMINISTRATOR)

    def test_invalid_role_rejected(self):
        auth, _ = make_auth()
        with self.assertRaises(ValueError):
            auth.create_user("carol", "longenough1", "superuser")

    def test_ensure_admin_only_bootstraps_an_empty_system(self):
        auth, _ = make_auth()
        first = auth.ensure_admin("root", "bootstrap-pw-1")
        self.assertEqual(first.role, Role.ADMINISTRATOR)
        self.assertIsNone(auth.ensure_admin("intruder", "bootstrap-pw-2"))

    def test_there_are_no_default_credentials(self):
        auth, _ = make_auth()
        for u, p in (("admin", "admin"), ("admin", "password"), ("root", "root")):
            with self.assertRaises(AuthenticationError):
                auth.login(u, p)


class TestLoginAndTokens(unittest.TestCase):
    def setUp(self):
        self.auth, self.repo = make_auth()
        self.user = self.auth.create_user("alice", "longenough1", Role.OPERATOR)

    def test_login_returns_token_that_authenticates_to_the_same_user(self):
        token = self.auth.login("alice", "longenough1")
        got = self.auth.authenticate(token)
        self.assertEqual((got.user_id, got.username, got.role), (self.user.user_id, "alice", Role.OPERATOR))

    def test_wrong_password_and_unknown_user_give_identical_errors(self):
        with self.assertRaises(AuthenticationError) as wrong_pw:
            self.auth.login("alice", "not-the-password")
        with self.assertRaises(AuthenticationError) as no_user:
            self.auth.login("nobody", "whatever-pw")
        self.assertEqual(str(wrong_pw.exception), str(no_user.exception))

    def test_missing_and_garbage_tokens_rejected(self):
        for bad in (None, "", "garbage", "a.b.c"):
            with self.assertRaises(AuthenticationError):
                self.auth.authenticate(bad)

    def test_expired_token_rejected(self):
        auth, _ = make_auth(token_ttl_seconds=-5)
        auth.create_user("alice", "longenough1", Role.VIEWER)
        token = auth.login("alice", "longenough1")
        with self.assertRaises(AuthenticationError) as ctx:
            auth.authenticate(token)
        self.assertIn("expired", str(ctx.exception).lower())

    def test_token_signed_with_a_different_key_rejected(self):
        forged = jwt.encode({"sub": self.user.user_id, "exp": int(time.time()) + 600},
                            "attacker-key", algorithm="HS256")
        with self.assertRaises(AuthenticationError):
            self.auth.authenticate(forged)

    def test_alg_none_token_rejected(self):
        unsigned = jwt.encode({"sub": self.user.user_id, "exp": int(time.time()) + 600}, key=None, algorithm="none")
        with self.assertRaises(AuthenticationError):
            self.auth.authenticate(unsigned)

    def test_token_without_expiry_rejected(self):
        no_exp = jwt.encode({"sub": self.user.user_id}, "test-secret-key-not-for-production-use-1234", algorithm="HS256")
        with self.assertRaises(AuthenticationError):
            self.auth.authenticate(no_exp)

    def test_deleted_user_loses_access_immediately(self):
        token = self.auth.login("alice", "longenough1")
        self.repo.db.execute("DELETE FROM users WHERE user_id = ?", (self.user.user_id,))
        with self.assertRaises(AuthenticationError):
            self.auth.authenticate(token)

    def test_role_change_takes_effect_on_existing_token(self):
        token = self.auth.login("alice", "longenough1")
        self.repo.db.execute("UPDATE users SET role = ? WHERE user_id = ?", ("viewer", self.user.user_id))
        self.assertEqual(self.auth.authenticate(token).role, Role.VIEWER)


class TestAuthorization(unittest.TestCase):
    def setUp(self):
        self.auth, _ = make_auth()
        self.viewer = self.auth.create_user("v", "longenough1", Role.VIEWER)
        self.operator = self.auth.create_user("o", "longenough1", Role.OPERATOR)
        self.admin = self.auth.create_user("a", "longenough1", Role.ADMINISTRATOR)

    def test_permission_matrix(self):
        expect = {
            Role.VIEWER: {Permission.READ},
            Role.OPERATOR: {Permission.READ, Permission.CONTROL_SESSIONS},
            Role.ADMINISTRATOR: set(Permission),
        }
        for role, perms in expect.items():
            self.assertEqual(set(ROLE_PERMISSIONS[role]), perms, role)

    def test_require_allows_and_denies(self):
        self.auth.require(self.viewer, Permission.READ)
        self.auth.require(self.operator, Permission.CONTROL_SESSIONS)
        self.auth.require(self.admin, Permission.MANAGE_USERS)
        with self.assertRaises(PermissionDenied):
            self.auth.require(self.viewer, Permission.CONTROL_SESSIONS)
        with self.assertRaises(PermissionDenied):
            self.auth.require(self.operator, Permission.MANAGE_CONFIG)
        with self.assertRaises(PermissionDenied):
            self.auth.require(self.operator, Permission.MANAGE_USERS)

    def test_only_admin_can_do_everything(self):
        for p in Permission:
            self.assertTrue(self.admin.can(p))
        self.assertFalse(self.viewer.can(Permission.MANAGE_CONFIG))


if __name__ == "__main__":
    unittest.main()
