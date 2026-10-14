"""
Vision Radar - Data Layer: PostgreSQL backend (Supabase)

Drop-in replacement for data_layer/database.py's `Database`: same public
interface (execute / query_all / query_one / transaction / close), same
failure semantics (Section 26: any storage failure flips the "database"
health component to ERROR -> "DATA STORAGE UNAVAILABLE"; a constraint
violation is a DataIntegrityError and does NOT flip health).

The Repository's SQL is already standard (ON CONFLICT upserts, no SQLite-only
functions), so it is reused unchanged. This module only adapts:
  * placeholders      '?'  -> '%s'
  * column type       REAL -> DOUBLE PRECISION (Postgres REAL is 4-byte: it
                      would corrupt epoch timestamps, so this matters)
  * schema            same tables, plus Row Level Security ENABLED on every
                      table (see below)
  * dropped links     one connection guarded by a lock; if the server or
                      pooler drops it, reconnect once and retry

Use the Supabase **Session pooler** connection string (IPv4+IPv6, port 5432).
The *direct* connection is IPv6-only unless you buy the IPv4 add-on.

Row Level Security: Supabase exposes every table in `public` through its
REST Data API using the public anon key. These tables hold password hashes
and camera URIs, so RLS is enabled with NO policies: the anon/authenticated
API roles can read nothing. This service connects as the `postgres` role,
which bypasses RLS, so it is unaffected.
"""

from __future__ import annotations

import logging
import re
import threading
from types import SimpleNamespace
from typing import Any, Iterable, List, Optional

from vision_radar.data_layer.database import SCHEMA, DataIntegrityError, DataStorageUnavailable
from vision_radar.monitoring.health import ComponentStatus, HealthRegistry

logger = logging.getLogger("vision_radar.postgres")

TABLES = ("cameras", "sessions", "entities", "zones", "events", "event_entities",
          "rules", "alerts", "evidence", "system_state", "users")

_REAL = re.compile(r"\bREAL\b")


def translate_sql(sql: str) -> str:
    """SQLite-style '?' placeholders -> psycopg '%s' (a literal '%' is escaped)."""
    return sql.replace("%", "%%").replace("?", "%s")


def postgres_schema_statements() -> List[str]:
    stmts = [s.strip() for s in _REAL.sub("DOUBLE PRECISION", SCHEMA).split(";") if s.strip()]
    stmts += [f"ALTER TABLE {t} ENABLE ROW LEVEL SECURITY" for t in TABLES]
    return stmts


def normalize_url(url: str) -> str:
    """Accept postgres:// or postgresql://; Supabase requires TLS, so default sslmode=require."""
    url = url.strip()
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if not url.startswith("postgresql://"):
        raise ValueError("VISION_RADAR_DATABASE_URL must start with postgresql:// or postgres://")
    if "sslmode=" not in url:
        url += ("&" if "?" in url else "?") + "sslmode=require"
    return url


class PostgresDatabase:
    def __init__(self, url: str, health: Optional[HealthRegistry] = None, connect_timeout: int = 10):
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:                       # pragma: no cover
            raise RuntimeError("PostgreSQL support needs: pip install 'psycopg[binary]>=3.1'") from exc
        self._pg = psycopg
        self._dict_row = dict_row
        self._url = normalize_url(url)
        self._timeout = connect_timeout
        self.path = "postgresql://<redacted>"            # never keep the password in a printable attribute
        self.health = health
        self._lock = threading.RLock()
        self._conn = None
        try:
            self._connect()
            self._init_schema()
        except self._pg.Error as exc:
            self._report_failure(exc)
            raise DataStorageUnavailable(str(exc)) from exc
        self._report_ok()

    # ---- connection management ----
    def _connect(self) -> None:
        # prepare_threshold=None: no server-side prepared statements, so this also
        # works through Supabase's transaction pooler (port 6543).
        self._conn = self._pg.connect(self._url, row_factory=self._dict_row, autocommit=False,
                                      connect_timeout=self._timeout, prepare_threshold=None)

    def _reconnect(self) -> None:
        logger.warning("Database connection lost; reconnecting")
        try:
            if self._conn is not None:
                self._conn.close()
        except Exception:
            pass
        self._connect()

    def _init_schema(self) -> None:
        with self._conn.cursor() as cur:
            for stmt in postgres_schema_statements():
                cur.execute(stmt)
        self._conn.commit()

    def _safe_rollback(self) -> None:
        try:
            self._conn.rollback()
        except Exception:
            pass

    def _with_retry(self, work):
        """Caller holds the lock. One reconnect+retry if the connection itself dropped."""
        try:
            return work()
        except (self._pg.OperationalError, self._pg.InterfaceError):
            self._reconnect()
            return work()

    def _report_ok(self) -> None:
        if self.health is not None:
            self.health.set("database", ComponentStatus.OK)

    def _report_failure(self, exc: Exception) -> None:
        # Full text (may name the host / project ref / DB user) goes to the server log only.
        # /status shows health detail to every logged-in user, so expose just the error class.
        logger.error("Database error: %s", exc)
        if self.health is not None:
            self.health.set("database", ComponentStatus.ERROR, detail=type(exc).__name__)

    # ---- public interface (same as Database) ----
    def execute(self, sql: str, params: Iterable[Any] = ()):
        sql, params = translate_sql(sql), tuple(params)

        def work():
            with self._conn.cursor() as cur:
                cur.execute(sql, params)
                n = cur.rowcount
            self._conn.commit()
            return n

        with self._lock:
            try:
                rowcount = self._with_retry(work)
            except self._pg.IntegrityError as exc:
                self._safe_rollback()
                raise DataIntegrityError(str(exc)) from exc
            except self._pg.Error as exc:
                self._safe_rollback()
                self._report_failure(exc)
                raise DataStorageUnavailable(str(exc)) from exc
        self._report_ok()
        return SimpleNamespace(rowcount=rowcount)

    def query_all(self, sql: str, params: Iterable[Any] = ()) -> List[dict]:
        sql, params = translate_sql(sql), tuple(params)

        def work():
            with self._conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
            self._conn.commit()          # end the read transaction (no idle-in-transaction on a pooler)
            return rows

        with self._lock:
            try:
                rows = self._with_retry(work)
            except self._pg.Error as exc:
                self._safe_rollback()
                self._report_failure(exc)
                raise DataStorageUnavailable(str(exc)) from exc
        self._report_ok()
        return rows

    def query_one(self, sql: str, params: Iterable[Any] = ()) -> Optional[dict]:
        rows = self.query_all(sql, params)
        return rows[0] if rows else None

    def transaction(self, statements: List[tuple]) -> None:
        """Several (sql, params) statements atomically - all or nothing."""
        prepared = [(translate_sql(s), tuple(p)) for s, p in statements]

        def work():
            with self._conn.cursor() as cur:
                for sql, params in prepared:
                    cur.execute(sql, params)
            self._conn.commit()

        with self._lock:
            try:
                self._with_retry(work)
            except self._pg.IntegrityError as exc:
                self._safe_rollback()
                raise DataIntegrityError(str(exc)) from exc
            except self._pg.Error as exc:
                self._safe_rollback()
                self._report_failure(exc)
                raise DataStorageUnavailable(str(exc)) from exc
        self._report_ok()

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
