"""
Vision Radar - Security: authentication & authorization

Spec reference: PDF Section 28 (Authentication: protect dashboard/API;
Authorization: separate administrator, operator, viewer).

Design choices:
  * Passwords are stored only as salted PBKDF2-HMAC-SHA256 hashes (standard
    library, 600k iterations) and compared in constant time.
  * Sessions are signed JWTs (HS256, algorithm pinned on decode so an
    "alg: none" token is rejected) with an expiry.
  * The token carries only the user id. On every request the user is
    re-loaded from the database, so deleting a user or changing their role
    takes effect immediately instead of when their token expires.
  * Login fails identically for "no such user" and "wrong password", and
    still burns a hash computation for unknown users, so neither the message
    nor (much of) the timing reveals which usernames exist.
  * There are NO default credentials. The first administrator is created
    explicitly via `ensure_admin()` from deployment configuration.
  * If no signing key is supplied one is generated at startup: safe, but all
    tokens become invalid on restart. Production should supply a stable key.

The role -> permission table is an interpretation: the spec names the three
roles but not what each may do.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import jwt

from vision_radar.data_layer.database import DataIntegrityError
from vision_radar.data_layer.repository import Repository

PBKDF2_ITERATIONS = 600_000
MIN_PASSWORD_LENGTH = 8
_ALGORITHM = "HS256"


class Role(str, Enum):
    VIEWER = "viewer"
    OPERATOR = "operator"
    ADMINISTRATOR = "administrator"


class Permission(str, Enum):
    READ = "read"                          # events, alerts, entities, zones, rules, status
    CONTROL_SESSIONS = "control_sessions"  # start / stop a camera session
    MANAGE_CONFIG = "manage_config"        # cameras, zones, rules, retention
    MANAGE_USERS = "manage_users"


ROLE_PERMISSIONS = {
    Role.VIEWER: frozenset({Permission.READ}),
    Role.OPERATOR: frozenset({Permission.READ, Permission.CONTROL_SESSIONS}),
    Role.ADMINISTRATOR: frozenset(Permission),
}


class AuthError(Exception):
    pass


class AuthenticationError(AuthError):
    """Who are you? - missing/invalid/expired credentials (HTTP 401)."""


class PermissionDenied(AuthError):
    """You're known, but not allowed (HTTP 403)."""


@dataclass(frozen=True)
class User:
    user_id: str
    username: str
    role: Role

    def can(self, permission: Permission) -> bool:
        return permission in ROLE_PERMISSIONS[self.role]


def hash_password(password: str, iterations: int = PBKDF2_ITERATIONS) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, iters, salt_hex, digest_hex = stored.split("$")
        if scheme != "pbkdf2_sha256":
            return False
        candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                                        bytes.fromhex(salt_hex), int(iters))
        return hmac.compare_digest(candidate, bytes.fromhex(digest_hex))
    except (ValueError, TypeError):
        return False


class AuthService:
    def __init__(self, repo: Repository, secret_key: Optional[str] = None,
                 token_ttl_seconds: float = 3600.0, hash_iterations: int = PBKDF2_ITERATIONS):
        self._repo = repo
        self._secret = secret_key or secrets.token_hex(32)
        self._ttl = token_ttl_seconds
        self._iterations = hash_iterations
        self._dummy_hash = hash_password("not-a-real-password", hash_iterations)

    # ---- user management -------------------------------------------------
    def create_user(self, username: str, password: str, role: Role) -> User:
        username = (username or "").strip()
        if not username:
            raise ValueError("Username must not be empty")
        if len(password or "") < MIN_PASSWORD_LENGTH:
            raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters")
        role = Role(role)
        user_id = str(uuid.uuid4())
        try:
            self._repo.add_user(user_id, username, hash_password(password, self._iterations), role.value)
        except DataIntegrityError:
            raise ValueError(f"Username '{username}' is already taken")
        return User(user_id, username, role)

    def ensure_admin(self, username: str, password: str) -> Optional[User]:
        """Create the first administrator only if no users exist yet.
        Returns the new user, or None if users already exist."""
        if self._repo.user_count() > 0:
            return None
        return self.create_user(username, password, Role.ADMINISTRATOR)

    # ---- login / token ---------------------------------------------------
    def login(self, username: str, password: str) -> str:
        row = self._repo.get_user_by_username((username or "").strip())
        stored = row["password_hash"] if row else self._dummy_hash
        ok = verify_password(password or "", stored)
        if not row or not ok:
            raise AuthenticationError("Invalid username or password")
        return self._issue_token(row["user_id"])

    def _issue_token(self, user_id: str) -> str:
        import time
        now = int(time.time())
        payload = {"sub": user_id, "iat": now, "exp": now + int(self._ttl)}
        return jwt.encode(payload, self._secret, algorithm=_ALGORITHM)

    def authenticate(self, token: Optional[str]) -> User:
        if not token:
            raise AuthenticationError("Missing credentials")
        try:
            payload = jwt.decode(token, self._secret, algorithms=[_ALGORITHM],
                                 options={"require": ["exp", "sub"]})
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token expired")
        except jwt.InvalidTokenError:
            raise AuthenticationError("Invalid token")
        row = self._repo.get_user(payload["sub"])
        if row is None:
            raise AuthenticationError("Unknown user")
        return User(row["user_id"], row["username"], Role(row["role"]))

    # ---- authorization ---------------------------------------------------
    @staticmethod
    def require(user: User, permission: Permission) -> None:
        if not user.can(permission):
            raise PermissionDenied(
                f"Role '{user.role.value}' is not allowed to perform '{permission.value}'")
