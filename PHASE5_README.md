# Phase 5 — World Model: Zones & Relationships

Scope (per spec Section 8 - Zones, Section 15 - Camera calibration,
Section 10 - Relationship engine), per the 10-phase plan.

## Structure
```
vision_radar/world_model/
  zone.py          # point_in_polygon, Zone (rect/polygon), ZoneManager, ZoneOccupancy
  calibration.py   # CameraCalibration (resolution/orientation/scene boundary)
  relationship.py  # RelationshipEngine (NEAR, MOVING_TOGETHER)
vision_radar/tests/
  test_zone.py          # 11 tests
  test_calibration.py    # 5 tests
  test_relationship.py   # 9 tests
```

## Zones (Section 8)
`Zone.from_rect(...)` and `Zone.from_polygon(...)` cover both the common
rectangular case and arbitrary shapes, via one ray-casting
`point_in_polygon` test. `ZoneManager.compute_occupancy(entities)` returns
one `ZoneOccupancy` per defined zone (even empty ones, as `IDLE`) shaped
directly like the spec's own Section 6 example
(`Zone A: Occupancy: 2, Entities: #17,#04, State: Active`). Overlapping
zones are explicitly supported — an entity can count toward more than one
zone at once (Section 8 imposes no non-overlap rule).

## Camera calibration (Section 15)
`CameraCalibration` stores resolution/orientation/an optional scene
boundary polygon, and converts between pixel and normalized coordinates.
It deliberately provides **no** pixel-to-real-world-distance conversion —
`estimate_real_world_distance()` exists specifically to raise
`NotImplementedError` with a clear explanation, so a future caller who
reaches for real-world units gets an honest error instead of a
plausible-looking fabricated number. This enforces Section 15's rule in
code, not just in a comment.

## Relationship engine (Section 10)
`RelationshipEngine` detects, per frame, pairwise `NEAR` (proximity below a
distance threshold) and `MOVING_TOGETHER` (same direction, similar speed,
both actually moving) relationships, giving each a stable ID for as long as
it continuously holds.

**Explicit scope boundary:** this does not persist a relationship across a
gap (if it drops out for even one frame, it ends; restarting later gets a
brand-new ID). Full dedup/hysteresis lifecycle (Potential -> Active ->
Resolved) is Phase 7's Event Engine job, which will consume these
relationship observations as one input. Documented here so it isn't
mistaken for an oversight later.

## Verified (96/96 unit tests total, 25 new this phase)
Zone: point-in-polygon (rectangle + arbitrary triangle), rect/polygon
construction, empty-zone IDLE state, entity counted only when inside,
overlapping-zone double-counting, remove/get. Calibration: normalize/
denormalize round-trip, zero-size guard, scene boundary in/out, and the
`NotImplementedError` guard. Relationship: proximity creation, distant
pair producing nothing, stable ID while continuously near, ID *not* reused
after a gap (matches the documented scope boundary above), same-direction/
similar-speed co-movement, and each of the three ways co-movement
correctly does *not* fire (different direction, one stationary, very
different speed).

## Real end-to-end validation (found and fixed a real design gap)
Ran the Phase 3/4 real-photo pan sequence through a defined zone covering
the left third of the frame. First pass fed *all* World Model entities
(including single-frame `DETECTED` noise blips, the same background
clutter flagged in Phase 3) into `compute_occupancy` — and a noise entity
sitting inside the zone boundary made it report `ACTIVE` at a frame where
the real, reliably-tracked motorcycle was nowhere near it. This is not a
bug in `ZoneManager` (it correctly reported the entities it was given) —
it surfaces a real usage decision: **zone occupancy should be computed from
filtered, confirmed-or-better entities**, not every raw entity in the World
Model, or single-frame noise can register a false zone event once Phase 7
wires zone entry to alerts.

Re-ran filtering out `DETECTED`-only entities first: the zone then reported
`IDLE` for 12 frames and flipped to `ACTIVE` at exactly the frame the real
motorcycle's tracked position crossed into the zone boundary (x=203,
zone width=213) — clean, correct, no false trigger.

**Action taken:** documented this as the required calling convention (not
a code change to `ZoneManager` itself, which is correctly staying
state-agnostic/reusable) — Phase 6/7 (state-change and event engines) will
need to apply this same filter before treating a zone entry as a real
event.

## Not in this phase (per the 10-phase plan)
Change/state-transition events (appeared/entered-zone/dwelled etc.) and
their deduplicated lifecycle — Phase 6/7. This phase only defines zones
and computes instantaneous occupancy/relationships; it does not yet decide
what a *change* in either of those means as an event.
