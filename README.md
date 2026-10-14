# Vision Radar — Frontend

Implements the Experience layer (spec Section 17/16/18/19/8/21/28): status,
live camera, radar view, entity panel, event history + filters, natural-
language search, zone editor, alerts, and (administrator) user management.

## Why vanilla JS, not React
The spec's architecture diagram names "React / TypeScript" for the
frontend. This project was built in a sandbox with no npm registry access
— `react`/`react-dom` existed globally but with no bundler reachable, so a
real React build could never be produced or tested here. Rather than ship
untested React source, this is a real, working dashboard in plain
JS/ES-modules — same component boundaries the spec describes, verified
end-to-end in an actual browser (see below). Porting it to React later is a
view-layer rewrite, not a behaviour rewrite: all the logic (API client,
state, polling, zone geometry) is already correct and tested.

## Running it
Needs any static file server (ES modules can't load from `file://`) plus a
running backend (`vision_radar.api.bootstrap.build_service` +
`fastapi_app.py` or the dev-only `dev_http_server.py`):

```
python3 -m http.server 8766 --directory vision_radar/frontend
# open http://127.0.0.1:8766/index.html?api=http://YOUR-BACKEND-HOST:PORT
```

The `?api=` query param points the frontend at the backend; omit it if
both are served from the same origin.

## Real end-to-end test (`e2e_test.mjs`)
A genuine Playwright test: launches real headless Chromium, drives the real
dashboard against a real running backend (real SQLite, real YOLOv8 model,
a real video file) — not mocks. Needs Node + Playwright with its browser
binary installed:

```
node vision_radar/frontend/e2e_test.mjs
```

It logs in as each of the three roles, adds a camera, draws a zone by
clicking the canvas, writes a rule, starts a real session, watches a real
motorcycle appear and a real alert fire, opens the real evidence image,
filters event history, asks two natural-language questions (one the parser
understands, one it honestly can't), creates a user, and confirms a viewer
cannot see admin controls. 22 checks; stable at 22/22 across repeated runs
in this project's own sandbox (6/6 most recently).

## Three real bugs found and fixed while building this
1. **Switching tabs didn't fetch fresh data.** Data only refreshed via a
   background poll gated to the Overview tab; switching to another tab and
   back showed stale (often still-empty, from before a camera/session even
   existed) data. Fixed: every tab now eagerly refreshes its own data when
   opened (`switchView`), not just on an interval.
2. **A background poll could wipe mid-filled form fields.** Because the
   whole view is rebuilt from `state` on every render, a poll landing
   between two `fill()` calls reset whatever had already been typed into
   other fields in that form - not just a test artifact; a real person
   typing slower than the poll interval would hit the same thing. Fixed
   generically in `render()`: every input/textarea/select's current value
   is preserved across a rebuild, not just the focused one.
3. **The evidence image was ORB-blocked.** `<img src>` can't carry an
   `Authorization` header, but the snapshot endpoint requires one - the
   request came back 401 (JSON) where an image was expected, and
   Chromium's opaque-response-blocking silently killed it rather than
   showing a normal broken image. Fixed by fetching the image with
   `fetch()` + the auth header and handing the page a `blob:` URL instead.

## Known limitations (flagged, not hidden)
- No automated JS unit tests for pure logic (e.g. the zone-geometry click
  math) separate from the full Playwright run - the sandbox had no
  JS test runner available offline; the Playwright run exercises that code
  for real instead.
- The live-camera view (Section 17-B) is a polled "most recent frame"
  (~2s cadence), not a push/streaming video feed - simpler, and matches
  the already frame-based architecture, but is not literally live video.
- Zone editor assumes a single camera's resolution; multi-camera zone
  reuse across differently-sized cameras isn't handled.
