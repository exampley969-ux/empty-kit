"""
Vision Radar - service bootstrap
Builds a fully-wired ApiService from environment variables. No web framework
needed, so this is testable (and usable from scripts/CLIs) on its own.

  VISION_RADAR_DATABASE_URL            PostgreSQL / Supabase URL (Session pooler string). If set, it is
                                       used INSTEAD of sqlite.
  VISION_RADAR_DB                      sqlite file           (default: vision_radar.db; or ":memory:")
  VISION_RADAR_EVIDENCE_DIR            evidence image dir    (default: evidence)
  VISION_RADAR_MODEL_PATH              YOLOv8 .onnx model    (default: unset -> sessions can't start)
  VISION_RADAR_JWT_SECRET              token signing key     (default: random per start; tokens die on restart)
  VISION_RADAR_TOKEN_TTL_SECONDS       token lifetime        (default: 3600)
  VISION_RADAR_ADMIN_USERNAME / _PASSWORD
                                       creates the FIRST administrator, only if no users exist yet
  VISION_RADAR_EVENT_RETENTION_DAYS    finished-event retention  (default: 30)
  VISION_RADAR_EVIDENCE_RETENTION_DAYS evidence-image retention  (default: 7)

There are deliberately no default credentials: without the two ADMIN
variables on first start, nobody can log in until an administrator is
created some other way.
"""

from __future__ import annotations

import logging
import os
from typing import Mapping, Optional

from vision_radar.api.service import ApiService
from vision_radar.data_layer.database import Database
from vision_radar.data_layer.repository import Repository
from vision_radar.data_layer.retention import RetentionPolicy
from vision_radar.monitoring.health import HealthRegistry
from vision_radar.monitoring.performance import PerformanceMonitor
from vision_radar.security.auth import AuthService

logger = logging.getLogger("vision_radar.bootstrap")


def _number(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{key} must be a number, got {raw!r}")
    if value <= 0:
        raise ValueError(f"{key} must be greater than zero, got {raw!r}")
    return value


def build_service(env: Optional[Mapping[str, str]] = None) -> ApiService:
    env = os.environ if env is None else env

    retention = RetentionPolicy(
        event_retention_days=_number(env, "VISION_RADAR_EVENT_RETENTION_DAYS", 30.0),
        evidence_image_retention_days=_number(env, "VISION_RADAR_EVIDENCE_RETENTION_DAYS", 7.0))
    retention.validate()
    token_ttl = _number(env, "VISION_RADAR_TOKEN_TTL_SECONDS", 3600.0)

    evidence_dir = env.get("VISION_RADAR_EVIDENCE_DIR") or "evidence"
    os.makedirs(evidence_dir, exist_ok=True)

    health = HealthRegistry()
    database_url = env.get("VISION_RADAR_DATABASE_URL")
    if database_url:
        from vision_radar.data_layer.postgres import PostgresDatabase   # lazy: psycopg only needed here
        database = PostgresDatabase(database_url, health=health)
        logger.info("Using PostgreSQL (Supabase) database")
    else:
        database = Database(env.get("VISION_RADAR_DB") or "vision_radar.db", health=health)
        logger.info("Using SQLite database")
    repo = Repository(database)

    secret = env.get("VISION_RADAR_JWT_SECRET") or None
    if secret is None:
        logger.warning("VISION_RADAR_JWT_SECRET is not set: using a random signing key, "
                       "so every login token becomes invalid when the service restarts.")
    elif len(secret) < 32:
        raise ValueError("VISION_RADAR_JWT_SECRET must be at least 32 characters")
    auth = AuthService(repo, secret_key=secret, token_ttl_seconds=token_ttl)

    admin_user, admin_pw = env.get("VISION_RADAR_ADMIN_USERNAME"), env.get("VISION_RADAR_ADMIN_PASSWORD")
    if admin_user and admin_pw:
        if auth.ensure_admin(admin_user, admin_pw) is not None:
            logger.info("Created initial administrator '%s'", admin_user)
    elif repo.user_count() == 0:
        logger.warning("No users exist and VISION_RADAR_ADMIN_USERNAME/PASSWORD are not set: "
                       "nobody will be able to log in.")

    return ApiService(repo, auth, health, PerformanceMonitor(), evidence_dir=evidence_dir,
                      retention=retention, model_path=env.get("VISION_RADAR_MODEL_PATH") or None)
