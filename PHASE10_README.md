# Phase 10 — Experience Layer, Search, and Final Acceptance

Scope (per the 10-phase plan): React dashboard (reinterpreted - see below),
natural-language search, zone editor UI, and the final end-to-end
acceptance run against spec Sections 31-32. This is the last phase: what
follows is also the project's completion report.

## Natural-language search (Section 19)
`vision_radar/api/nl_search.py` — a small, deterministic set of regex
extractors (event type, zone name, entity type, time range, confidence),
never a free-form interpretation. A query is answered strictly from
`Repository.query_events` (Phase 9); anything the parser can't map to a
filter is returned verbatim in `ignored`, not silently dropped or guessed
at — matching Section 19's own requirement ("It should answer from stored
events, not hallucinate... this distinction is extremely important").
Wired into `ApiService.search_events` and exposed as `GET /search?q=`.

**27 unit tests**, plus **4 real bugs found and fixed**: "disappear" was
matching the `appear` pattern (substring); the zone-name regex was greedy
enough to swallow trailing words like "today" into the zone name; the
confidence regex only captured one digit (`"80"` read as `"8"`); and a
zone referenced without the literal word "zone" was wrongly reported as
not understood. Verified for real over an actual running backend
(`search_events` hit over a real socket) with genuine English questions,
including one Section 19 was explicit about: an unrecognised phrase
("did anybody loiter suspiciously... ") correctly returns today's real
events with the unrecognised words honestly flagged as ignored, rather
than fabricating a match for them.

## Live camera frame (added this phase, closing a Section 17 gap)
Building the dashboard surfaced a real gap: nothing exposed a live view of
a running session (Section 17-B). Added `PipelineRunner.latest_frame_jpeg()`
(thread-safe, JPEG-encoded, replaced not accumulated) and
`GET /sessions/{id}/frame`, available only while that session is actually
running — once it ends, the endpoint returns 404 rather than serving a
stale last frame (Section 32: "stale data isn't presented as live").

## Frontend dashboard (Section 17, 16, 18, 8, 21, 28)
Full detail in `frontend/README.md`. Summary: vanilla JS/ES-modules, not
React — disclosed, deliberate, for the same reason Phase 9's FastAPI
runtime tests were disclosed rather than skipped silently: no npm registry
access in this sandbox meant no real, buildable React dependency, and
shipping untested React source seemed worse than a real, fully-tested
plain-JS dashboard. Unlike the FastAPI situation, this one **could** be
fully tested for real, because Playwright's Chromium binary happened to
already be installed — so it was.

**22/22 real end-to-end checks**, run in an actual headless browser against
an actual running backend (real SQLite, real YOLOv8 model, a real video
file) — stable across 6 repeated runs. Covers: three-role login, every
dashboard panel, drawing a zone by clicking a canvas, writing a rule,
starting a real session, watching a real detection become a real alert,
opening its real evidence image, filtering event history, two real
natural-language queries, user creation, and confirming a viewer sees no
admin controls. **Three real bugs found and fixed**: tab switches not
refreshing data, a background poll wiping mid-filled form fields, and the
evidence `<img>` being silently blocked because images can't carry the
`Authorization` header the endpoint required (fixed with an authenticated
`fetch` → `blob:` URL instead of a bare `src`).

---

## Section 31 — the complete scenario, mapped to real evidence

> Camera running → Person enters scene → Detector identifies person →
> Tracker creates identity → World Model updates position → entity crosses
> Zone A boundary → Event Engine creates Zone Entry → Rule Engine evaluates
> condition → Alert generated → Event stored → Dashboard updates → User
> searches event → Evidence displayed

| Step | Real evidence |
|---|---|
| Camera running | `OpenCVVideoSource` on a real `.mp4`; Phase 9/10 full-stack runs |
| Entity enters scene / Detector identifies it | Real YOLOv8 detection of a real motorcycle photo (Phase 2), re-confirmed every phase since |
| Tracker creates identity | `IoUTracker`, same `track_id` held across the whole real sequence (Phase 3) |
| World Model updates position | `WorldModel.update()` reflecting it every frame (Phase 4) |
| Crosses a zone boundary | `ZoneManager.compute_occupancy` + `StateChangeEngine` → `ZONE_ENTERED` at the exact real frame (Phase 5/6) |
| Event Engine creates Zone Entry | `EventEngine` → one `ZONE_ENTRY` event, correct lifecycle (Phase 7) |
| Rule Engine evaluates / Alert generated | Real rule matched, real `Alert` produced, exactly once (Phase 8/9) |
| Event stored | Real SQLite row, survives a simulated restart (Phase 9) |
| Dashboard updates | Playwright: the real entity and real alert appear in the running dashboard (Phase 10) |
| User searches event | Playwright: a real NL query returns that real event (Phase 10) |
| Evidence displayed | Playwright: the real annotated evidence image actually decodes on screen (Phase 10) |

Every step of Section 31's own worked example has now been executed for
real, by an automated test, at least once — most steps many times, across
multiple phases, with multiple independently-constructed scenarios.

## Section 32 — acceptance criteria

**Input** — camera can be connected ✅ (Phase 1, 9); video session can
start/stop ✅ (Phase 1, 9, 10 UI); camera failure is detected ✅ (Phase 1
reconnect, Phase 9 mid-run disconnect + missing-file, both surfaced as
`CAMERA OFFLINE`).

**Perception** — defined entities are detected ✅ (real YOLOv8, Phase
2/3/9); detections contain confidence ✅ (threaded end-to-end into events,
Phase 7); tracking maintains identity under defined test conditions ✅
(Phase 3's real-photo sequence, one identity held throughout).

**World model** — entities have positions ✅; trajectories are maintained
✅ (Phase 4); zones can be defined ✅ (Phase 5 API, Phase 10 UI editor,
both exercised for real); current state is continuously updated ✅
(`WorldModel.update()` rebuilds every frame, Phase 4).

**Intelligence** — entry/exit events work ✅ (Phase 6/9, including the
session-end-closes-open-events fix); movement events work ✅ (Phase 6,
real `MOVED` events in the full run); dwell events work ✅ (Phase 6 unit
+ Phase 8's rule example); appearance/disappearance works ✅ (Phase 6,
real `APPEARED`/`DISAPPEARED` in the full run); relationship/state events
⚠️ **partially verified** — `STATE_CHANGED` (Phase 6) and the relationship
engine's `NEAR`/`MOVING_TOGETHER` (Phase 5) are both real, unit-tested
implementations, but neither fired in any of this project's real
end-to-end runs: the real test scenario has exactly one real entity that
never stops moving, so there was never a second entity to relate to, nor a
moving→stationary transition to trigger. Correct as built; **not
exercised against real multi-entity footage** — worth re-confirming once
real camera footage with two entities (or one that stops) is available.

**Event system** — events have lifecycle ✅ (all 5 states, Phase 7);
duplicate events are controlled ✅ (Phase 7 zone-entry dedup, Phase 9's
rule-dedup fix for the same event reported at multiple lifecycle stages);
confidence is preserved ✅ (real, varying, non-fabricated values
end-to-end); evidence is attached ✅ (real annotated images, Phase
8/9/10, visually confirmed multiple times).

**Experience** — live camera works ✅ (added this phase, Playwright-
confirmed with a real decoded frame — a *polled snapshot*, not a push
video stream; disclosed in `frontend/README.md`); radar view ✅
structurally (same `state.entities` array confirmed real for the Entity
Panel feeds the radar SVG's `<circle>` elements directly — not
separately pixel-asserted in the E2E run); event feed / history work ✅
(Playwright: real filtering by type, confirmed real rows); search works
✅ (27 unit tests + real Node run + real Playwright run); alerts work ✅
(real alert from a real rule, exactly once).

**Reliability** — failures are visible ✅ (Section 26's exact strings,
Phase 9, every pipeline stage); stale data isn't presented as live ✅
(health staleness flagging Phase 9; live-frame 404s once a session ends,
this phase); system recovers from defined interruptions ✅ (reconnect
Phase 1, mid-run camera loss + DB loss both keep processing and surface
the failure, Phase 9).

**Data** — sessions persist ✅; events persist ✅; configurations persist
✅ (all three confirmed surviving a simulated full-service restart on the
same database file, Phase 9).

**Security/privacy** — access control works ✅ (3-role matrix, 23 auth
tests, Phase 9; Playwright confirms a viewer genuinely sees no admin
controls); camera status is explicit ✅ (Section 26 strings surfaced
through to the dashboard); retention behaviour is defined ✅ (configurable
days for events/evidence images, path-traversal-safe deletion, Phase 9,
real aged-data run).

### Honest summary
Every Section 32 line item has real, automated evidence behind it except
one explicitly flagged partial: relationship and state-change events are
correctly implemented and unit-tested, but were never triggered by real
multi-entity or stop-and-go footage in any of this project's integration
runs, because the one real test subject available offline (a single
real motorcycle photo, panned to simulate motion) never stood still and
was always alone in frame. Re-confirm those two event types specifically
once real camera footage exists.

## What to do next (beyond this project's scope)
1. Run `frontend/e2e_test.mjs` and `tests/test_fastapi_app.py` on a machine
   with real network access, to confirm the real FastAPI adapter and a
   from-scratch `npm install` of this frontend's React port (if pursued)
   behave the same as the stdlib/vanilla-JS stand-ins verified here.
2. Re-run the full Section 31 scenario against real camera hardware and
   real multi-person/vehicle footage — closing the one flagged gap above,
   and replacing every "real photo, synthetic motion" evidence note in
   earlier phases' READMEs with genuine live-camera evidence.
3. Decide real retention/token-lifetime values, a production `JWT_SECRET`,
   and CORS origin for deployment (all currently placeholders, disclosed in
   Phase 9's README).
