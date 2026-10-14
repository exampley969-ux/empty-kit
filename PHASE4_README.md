# Phase 4 — World Model: Entities & Spatial Model

Scope (per spec Section 6 - World Model, Section 8 - Spatial model), per
the 10-phase plan: **live entity state, coordinates, trajectory, velocity,
direction only** — zones and relationships are Phase 5, not here.

## Structure
```
vision_radar/world_model/
  models.py       # MotionState (MOVING/STATIONARY), EntitySnapshot
  world_model.py  # WorldModel: tracks -> queryable live entity snapshots
vision_radar/tests/
  test_world_model.py   # 11 tests
```

## Design decision — thin layer, not a second source of truth
`WorldModel` does not re-decide identity or lifecycle; it only reads
`Track` objects from Phase 3's tracker each frame and derives
presentation-ready fields (`motion_state`, normalized position). This keeps
one authority for "is this the same entity" (the tracker) and avoids two
components ever disagreeing about it.

- `EXITED` tracks are dropped from the World Model immediately — "what
  exists right now" (Section 6) never includes something that just left.
- `update()` is called once per frame and fully replaces the previous
  snapshot set — this satisfies the Phase 32 acceptance line "current
  state is continuously updated" directly: nothing here is cached longer
  than one frame.
- `MotionState` (MOVING/STATIONARY) is intentionally separate from the
  tracking lifecycle state (`DETECTED/TRACKED/...`, Section 7) — one
  describes physical movement, the other describes tracking confidence;
  the spec's own Section 6 example shows both ("State: Moving" alongside
  implicit tracking) so they should not be conflated into one enum.

## Verified (71/71 unit tests total, 11 new this phase)
Covers: snapshot creation per active track, snapshots reflecting the
*latest* tracker call (not stale data), entity lookup by ID (including the
correct `KeyError` for an unknown ID), exited tracks disappearing from the
model, moving vs stationary classification, per-type and per-motion-state
counts, and normalized-position scaling (plus the zero-frame-size guard).

## Real end-to-end validation (full chain, not just this layer)
Re-ran the Phase 3 real-photo pan sequence (13 frames, real motorcycle
photo, real YOLO detections, real tracker) through the World Model too.
Confirmed the live snapshot correctly reported: normalized position moving
consistently from `(0.68, 0.51)` to `(0.32, 0.46)` (left-to-right pan =>
entity appears to move left/west, matching Phase 3's direction finding),
direction holding `W` frame over frame, and `dwell_seconds` incrementing
correctly by exactly 1.0s per frame throughout.

## Known limitation (found and fixed during this phase's own test-writing, flagged rather than buried)
A first draft of the "moving entity" test used a synthetic motion step of
30px/frame with a 40px-wide box. That exceeded what IoU-based matching
(threshold 0.3) can bridge frame-to-frame — the "moving" entity kept losing
its identity and restarting as a new track, so the World Model correctly
never got enough history to report it as MOVING (a brand-new track's
velocity is genuinely unknown, not zero — see Phase 3's tracker code
comments). This was a test-authoring bug, not a World Model or tracker
bug — fixed by using a slower, IoU-compatible step (8px/frame) — but it
surfaces a real, general property worth remembering: **naive IoU tracking
requires enough frame-to-frame box overlap to bridge motion**; very fast
motion relative to box size and frame rate needs either a higher frame
rate or motion-prediction (e.g. a Kalman filter) to stay continuous. Not
implemented now — out of scope unless real footage shows this actually
matters — but recorded here so it isn't rediscovered as a "mystery bug"
later.

## Not in this phase (per the 10-phase plan)
Zones, camera calibration, relationship engine — Phase 5.
