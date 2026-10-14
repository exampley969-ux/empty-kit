# Phase 9 — Data Layer & Backend API

Scope (per the 10-phase plan): persistence for every Section 23 entity,
the backend service and API, authentication/roles, retention/privacy,
error visibility, and performance monitoring. The React experience layer,
natural-language search and the final acceptance run are Phase 10.

**Test status: 315 tests; 302 run and pass, 13 are skipped here** (see
"What was NOT executed"). 165 of the 315 are new this phase.

## Structure
```
data_layer/   database.py (SQLite, health-reporting)  repository.py (all SQL)  retention.py
security/     auth.py            password hashing, JWT tokens, roles/permissions
monitoring/   health.py          Section 26 failure states, staleness
              performance.py     Section 27 metrics
pipeline/     runner.py          wires Phases 1-8 together + persists results
api/          service.py         ALL API behaviour, framework-independent
              errors.py  serializers.py  bootstrap.py (env -> wired service)
              fastapi_app.py     thin HTTP adapter (see limits below)
```

## Running it
```
pip install -r requirements.txt
export VISION_RADAR_JWT_SECRET=<32+ random chars>
export VISION_RADAR_ADMIN_USERNAME=... VISION_RADAR_ADMIN_PASSWORD=...   # first start only
export VISION_RADAR_MODEL_PATH=vision_radar/models/yolov8n.onnx
uvicorn vision_radar.api.fastapi_app:create_app_from_env --factory
```
All settings are listed in `api/bootstrap.py`. There are **no default
credentials**: without the two ADMIN variables on first start nobody can
log in. If `JWT_SECRET` is unset a random key is used and every token dies
on restart (a warning is logged).

## Six defects found and fixed (all in earlier phases)
Reviewing Phases 3-8 as a long-running service surfaced problems that the
short fixed test sequences of those phases could never show. Each has a
regression test.

1. **Entity leaves the scene while inside a zone -> its `ZONE_ENTRY` event
   stayed `ACTIVE` forever.** The change engine now emits `ZONE_EXITED`
   (reason `entity_disappeared`) before `DISAPPEARED`.
2. **Duplicate alerts.** The event engine legitimately reports one event
   several times (entered / left / archived); the rule engine alerted each
   time. A rule now fires at most once per event, yet a `duration_seconds`
   rule still fires the first time the event resolves.
3. **Unbounded memory.** Exited tracks and archived events were kept in RAM
   forever. `purge_exited()` / `purge_archived()` added; the runner calls
   them once results are persisted (and skips purging if persisting failed,
   so nothing unsaved is lost). Verified O(1): retained state after 3 and 30
   appear/disappear cycles is identical.
4. **Archived state never reached storage** (archival was silent). Newly
   archived events are now reported to the persister.
5. **A session ending with someone inside a zone left the event `ACTIVE`
   forever**, and retention never deletes `ACTIVE` events. Found by
   reproducing the scenario before the final run. Open events are now closed
   at session end with `closed_reason: session_ended`; no `exited_at` is
   written, because the entity did not leave - observation stopped.
6. **`requirements.txt` was missing `scipy`, `PyJWT`, `psutil`** (scipy since
   Phase 3): a fresh install would have crashed on import. Fixed, with a
   guard test. That guard's *first version passed while scanning nothing*;
   proving it by removing scipy exposed this, and it now has non-vacuous
   and negative tests.

Also caught by Phase 9's own tests: `rollback()` inside the DB error handler
crashed on a closed connection (leaking a raw sqlite error instead of
`DATA STORAGE UNAVAILABLE`), and one test was flaky because it depended on
real-clock speed (fixed with a deterministic clock and a stronger claim).
Small visual fix from looking at a real evidence image: the label was
clipped when the box touched the top edge.

## Design decisions worth knowing
- **SQLite** (standard library) - Section 24 asks for local, close-to-source
  storage. Only `repository.py` knows SQL; every value is a bound parameter
  (tested against injection strings). In this implementation an Entity and
  its Track are one object, so one `entities` table covers both.
- **Security** - PBKDF2-SHA256 (600k iterations), constant-time compare,
  JWT with pinned algorithm and required expiry (`alg:none`, wrong-key and
  no-expiry tokens are tested and rejected). The user is re-loaded from the
  DB on every request, so deleting a user or changing a role bites
  immediately. Login errors are identical for unknown user / wrong password.
- **Roles** (the spec names them but not their powers - this table is my
  interpretation): viewer = read; operator = read + start/stop sessions;
  administrator = everything incl. cameras, zones, rules, users, retention.
  The whole matrix is tested for all 14 operations x 3 roles.
- **Retention** (Sections 28-29): evidence *images* expire after 7 days,
  finished events after 30 (placeholder defaults - the spec requires
  "configurable and defined", not a number). Active events are never
  deleted; deletion only touches files inside the evidence directory
  (path-traversal and tampered-row tests).
- **Evidence images only for alert-triggering events**, not every event -
  deliberate (Section 29: no unnecessary retention of people's images).
- **Errors never leak internals**: storage failures surface as
  `DATA STORAGE UNAVAILABLE`, unknown errors as a generic 500; the evidence
  endpoint refuses any path outside the evidence directory.
- **Failure visibility** (Section 26): each pipeline stage reports to a
  health registry using the spec's exact strings; a failing frame is skipped
  and counted, not fatal; components that stop reporting are flagged
  stale and stop counting as OK (Section 32: no stale data shown as live).

## Verified for real (full stack, not mocks)
On-disk SQLite; a real `.mp4` made from the real motorcycle photo; the
default runner factory (`OpenCVVideoSource` -> real YOLOv8 model); driven
entirely through the API service as admin/operator/viewer:
- 13 frames processed, ~10 fps on CPU (101 ms/frame), ~200 MB, GPU reported
  honestly as `None`; all five components `OK`.
- Events: 1 `APPEARED`, 6 `MOVED`, 1 `ZONE_ENTRY` (closed at session end).
  **Exactly one alert**, with an annotated evidence image (viewed: box,
  label and zone boundary correct).
- **Restart**: brand-new service on the same file - users (bootstrap admin
  not duplicated), camera, zone, rule, events, alerts, session all present.
- **Retention** on that data deleted all 8 aged events and cascaded to
  alerts and evidence; fresh data was untouched.
- **Missing video source** -> `CAMERA OFFLINE`, camera health `ERROR`.

## What was NOT executed (read this)
**FastAPI is not installed in the sandbox this was built in (no network), so
`fastapi_app.py` has never run.** What is in place instead:
- The adapter is deliberately thin: one service call per route, all logic in
  `ApiService` (which *is* fully tested, 27 tests).
- 4 static tests read the adapter's source and verify every `service.x(...)`
  call against the real `ApiService` signatures and that **every route but
  login requires the auth dependency**. Mutation-checked: injected typos,
  missing arguments, wrong keywords and a removed auth dependency were each
  caught.
- 13 runtime tests (`test_fastapi_app.py`, `TestClient`) are written but
  **skipped here**. On a machine with FastAPI run
  `python -m unittest vision_radar.tests.test_fastapi_app -v` before
  trusting the HTTP layer; expect to fix small integration details.

## Known limitations (flagged, not hidden)
- Zones/rules are read when a session *starts*; editing them affects
  sessions started afterwards, not running ones.
- No login rate-limiting/lockout (not in the spec; add before exposing the
  API to an untrusted network). No CORS configured - set your React app's
  explicit origin at deployment.
- While the database is down, frames are still processed but their events
  are lost (counted in `persist_failures`, not retried).
- Sessions on the same camera are limited to one at a time.
- `cpu_percent` is machine-wide; its first reading after start is ~0.
- One session per runner thread; many simultaneous cameras are untested.

## Not in this phase (Phase 10)
React dashboard (live camera, radar view, timeline, entity panel,
analytics), natural-language search, zone editor UI, and the final
end-to-end acceptance run against Sections 31-32.
