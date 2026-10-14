"""
Tests for data_layer/postgres.py against a FAKE psycopg (no server, no network).
They verify the adaptation logic - placeholder translation, REAL->DOUBLE
PRECISION, RLS statements, error mapping, reconnect-and-retry, backend choice.
They do NOT prove behaviour against a live Supabase: run
`python -m vision_radar.data_layer.supabase_check` for that.
"""
import sys
import types
import unittest

from vision_radar.data_layer import postgres as pgmod
from vision_radar.data_layer.database import DataIntegrityError, DataStorageUnavailable
from vision_radar.monitoring.health import ComponentStatus, HealthRegistry


def make_fake_psycopg(script):
    """script: list of behaviours consumed per cursor.execute: None | Exception | rows."""
    mod = types.ModuleType("psycopg")

    class Error(Exception): ...
    class OperationalError(Error): ...
    class InterfaceError(Error): ...
    class IntegrityError(Error): ...
    mod.Error, mod.OperationalError, mod.InterfaceError, mod.IntegrityError = Error, OperationalError, InterfaceError, IntegrityError
    mod.executed, mod.connects, mod.commits, mod.rollbacks = [], [], [0], [0]

    class Cur:
        rowcount = 1
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def execute(self, sql, params=()):
            mod.executed.append((sql, params))
            if "COUNT(*)" in sql:
                self._rows = [{"n": 0}]
                return
            if sql.startswith(("CREATE", "ALTER")) or not script:
                return
            step = script.pop(0)
            if isinstance(step, Exception):
                raise step
            self._rows = step
        def fetchall(self): return getattr(self, "_rows", [])

    class Conn:
        def cursor(self): return Cur()
        def commit(self): mod.commits[0] += 1
        def rollback(self): mod.rollbacks[0] += 1
        def close(self): pass

    def connect(url, **kw):
        mod.connects.append((url, kw))
        return Conn()
    mod.connect = connect
    rows = types.ModuleType("psycopg.rows"); rows.dict_row = object()
    mod.rows = rows
    return mod, rows


class PostgresBackendTests(unittest.TestCase):
    def setUp(self):
        self._saved = {k: sys.modules.get(k) for k in ("psycopg", "psycopg.rows")}

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None: sys.modules.pop(k, None)
            else: sys.modules[k] = v

    def _db(self, script=None, health=None):
        mod, rows = make_fake_psycopg(list(script or []))
        sys.modules["psycopg"], sys.modules["psycopg.rows"] = mod, rows
        return pgmod.PostgresDatabase("postgres://u:secret@host:5432/postgres", health=health), mod

    def test_placeholders_translated_and_percent_escaped(self):
        self.assertEqual(pgmod.translate_sql("SELECT * FROM t WHERE a = ? AND b = ?"), "SELECT * FROM t WHERE a = %s AND b = %s")
        self.assertEqual(pgmod.translate_sql("x LIKE '50%' AND y = ?"), "x LIKE '50%%' AND y = %s")

    def test_schema_uses_double_precision_and_enables_rls_on_every_table(self):
        stmts = pgmod.postgres_schema_statements()
        joined = "\n".join(stmts)
        self.assertNotRegex(joined, r"\bREAL\b")
        self.assertIn("DOUBLE PRECISION", joined)
        for t in pgmod.TABLES:
            self.assertIn(f"ALTER TABLE {t} ENABLE ROW LEVEL SECURITY", stmts)

    def test_url_normalised_and_tls_required(self):
        u = pgmod.normalize_url("postgres://a:b@h:5432/postgres")
        self.assertTrue(u.startswith("postgresql://") and u.endswith("?sslmode=require"))
        self.assertEqual(pgmod.normalize_url("postgresql://a@h/db?sslmode=verify-full"), "postgresql://a@h/db?sslmode=verify-full")
        with self.assertRaises(ValueError):
            pgmod.normalize_url("mysql://x")

    def test_connect_disables_prepared_statements_and_hides_password(self):
        db, mod = self._db()
        self.assertIsNone(mod.connects[0][1]["prepare_threshold"])
        self.assertNotIn("secret", db.path)

    def test_execute_translates_sql_and_reports_ok(self):
        health = HealthRegistry()
        db, mod = self._db(health=health)
        res = db.execute("DELETE FROM cameras WHERE camera_id = ?", ("c1",))
        self.assertEqual(res.rowcount, 1)
        self.assertEqual(mod.executed[-1], ("DELETE FROM cameras WHERE camera_id = %s", ("c1",)))

    @staticmethod
    def _boom(exc):
        class Boom:
            def __enter__(s): return s
            def __exit__(s, *a): return False
            def execute(s, *a): raise exc
        return lambda: Boom()

    def test_integrity_error_mapped_and_does_not_flip_health(self):
        health = HealthRegistry()
        db, mod = self._db(health=health)
        db._conn.cursor = self._boom(mod.IntegrityError("duplicate key"))
        with self.assertRaises(DataIntegrityError):
            db.execute("INSERT INTO users VALUES (?)", ("x",))
        self.assertNotEqual(health.get("database").status, ComponentStatus.ERROR)

    def test_storage_error_flips_health_to_error(self):
        health = HealthRegistry()
        mod, rows = make_fake_psycopg([])
        sys.modules["psycopg"], sys.modules["psycopg.rows"] = mod, rows
        db = pgmod.PostgresDatabase("postgresql://u:p@h/db", health=health)
        class Boom:
            def __enter__(s): return s
            def __exit__(s, *a): return False
            def execute(s, *a): raise mod.Error("disk full")
        db._conn.cursor = lambda: Boom()
        with self.assertRaises(DataStorageUnavailable):
            db.query_all("SELECT 1")
        self.assertEqual(health.get("database").status, ComponentStatus.ERROR)

    def test_health_detail_never_contains_connection_details(self):
        health = HealthRegistry()
        db, mod = self._db(health=health)
        db._conn.cursor = self._boom(mod.Error(
            'connection to server at "aws-0-eu.pooler.supabase.com" failed: password authentication failed for user "postgres.abcref"'))
        with self.assertRaises(DataStorageUnavailable):
            db.query_all("SELECT 1")
        detail = health.get("database").detail
        self.assertNotIn("supabase", detail)
        self.assertNotIn("abcref", detail)

    def test_dropped_connection_reconnects_once_and_retries(self):
        mod, rows = make_fake_psycopg([])
        sys.modules["psycopg"], sys.modules["psycopg.rows"] = mod, rows
        db = pgmod.PostgresDatabase("postgresql://u:p@h/db")
        first_conn = db._conn
        calls = {"n": 0}
        class Flaky:
            def __enter__(s): return s
            def __exit__(s, *a): return False
            rowcount = 1
            def execute(s, *a):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise mod.OperationalError("server closed the connection")
            def fetchall(s): return [{"n": 3}]
        first_conn.cursor = lambda: Flaky()
        # after reconnect a fresh (healthy) connection is used
        rows_out = db.query_all("SELECT username FROM users")
        self.assertEqual(len(mod.connects), 2)            # initial + one reconnect
        self.assertIsNot(db._conn, first_conn)
        self.assertEqual(rows_out, [])                    # fake healthy cursor returns no rows

    def test_bootstrap_chooses_postgres_when_url_set_else_sqlite(self):
        from vision_radar.api.bootstrap import build_service
        mod, rows = make_fake_psycopg([])
        sys.modules["psycopg"], sys.modules["psycopg.rows"] = mod, rows
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            svc = build_service({"VISION_RADAR_DATABASE_URL": "postgresql://u:p@h/db",
                                 "VISION_RADAR_EVIDENCE_DIR": d})
            self.assertIsInstance(svc.repo.db, pgmod.PostgresDatabase)
            svc2 = build_service({"VISION_RADAR_DB": ":memory:", "VISION_RADAR_EVIDENCE_DIR": d})
            self.assertNotIsInstance(svc2.repo.db, pgmod.PostgresDatabase)
            self.assertEqual(len(mod.connects), 1)        # only the first call touched Postgres


if __name__ == "__main__":
    unittest.main()
