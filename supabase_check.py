"""
Real connectivity check for the Supabase/PostgreSQL backend.

    VISION_RADAR_DATABASE_URL='postgresql://...' python -m vision_radar.data_layer.supabase_check

Connects for real, creates the schema, does a write/read/delete round-trip
through the same Repository the service uses, times it, and confirms Row
Level Security is on for every table. Never prints the password.
"""
import os
import sys
import time
import uuid

from vision_radar.data_layer.postgres import TABLES, PostgresDatabase
from vision_radar.data_layer.repository import CameraRecord, Repository


def main() -> int:
    url = os.environ.get("VISION_RADAR_DATABASE_URL")
    if not url:
        print("FAIL: set VISION_RADAR_DATABASE_URL first")
        return 2
    t0 = time.time()
    try:
        db = PostgresDatabase(url)
    except Exception as exc:
        print(f"FAIL: could not connect/create schema: {type(exc).__name__}: {exc}")
        print("Hint: use the Supabase *Session pooler* string (port 5432) - the direct host is IPv6-only.")
        return 1
    print(f"OK   connected + schema ready in {time.time() - t0:.2f}s")

    repo = Repository(db)
    cam_id = "check-" + uuid.uuid4().hex[:8]
    t1 = time.time()
    repo.add_camera(CameraRecord(cam_id, "supabase-check", "file:///none", None, None, "landscape"))
    got = repo.get_camera(cam_id)
    repo.delete_camera(cam_id)
    ok = got is not None and got.name == "supabase-check" and repo.get_camera(cam_id) is None
    print(f"{'OK  ' if ok else 'FAIL'} write/read/delete round-trip in {(time.time() - t1) * 1000:.0f} ms")

    rows = db.query_all(
        "SELECT c.relname, c.relrowsecurity FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
        "WHERE n.nspname = 'public' AND c.relkind = 'r' AND c.relname = ANY(%s)", (list(TABLES),))
    open_tables = sorted(r["relname"] for r in rows if not r["relrowsecurity"])
    missing = sorted(set(TABLES) - {r["relname"] for r in rows})
    print("OK   Row Level Security enabled on all tables" if not open_tables and not missing
          else f"FAIL RLS off for {open_tables}, tables missing: {missing}")
    db.close()
    return 0 if ok and not open_tables and not missing else 1


if __name__ == "__main__":
    sys.exit(main())
