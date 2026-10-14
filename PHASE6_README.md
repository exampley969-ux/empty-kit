# Phase 6 — State & Change Engine

Scope (per spec Section 9), per the 10-phase plan: detect
Appeared/Disappeared/Moved/Zone-entered/Zone-exited/Dwelled/State-changed
by diffing each frame's World Model + Zone occupancy against the previous
frame. This layer does **not** do event lifecycle, deduplication-across-
gaps, confidence, or alerts — that's Phase 7 (Event Engine), which
consumes the `Change` records this layer emits.

## Structure
```
vision_radar/state_engine/
  models.py   # ChangeType (7 types), StateChangeConfig, Change
  engine.py   # StateChangeEngine
vision_radar/tests/
  test_state_engine.py   # 13 tests
```

## Interpretation decision worth knowing — "Movement" is threshold-based
Section 9 lists "Movement: Position A -> Position B" as its own change
type, distinct from "State change: Moving <-> Stationary". Taken
literally, every single frame an entity's position differs at all would
be a "movement change" — which would flood the system with a change every
frame for any moving entity, defeating the whole point of a *change*
engine. `StateChangeEngine` instead fires `MOVED` only once cumulative
displacement since the last `MOVED` for that entity crosses
`movement_threshold_px` (default 30px), then resets its baseline from
there. This is a documented interpretation choice, not something the spec
spells out explicitly — flagged here rather than buried.

## Dwell and zone bookkeeping
- `ZONE_ENTERED`/`ZONE_EXITED` fire exactly once per actual crossing (diffed
  against the previous frame's zone membership, not re-fired every frame
  the entity stays put).
- `DWELL` fires once when time-inside-zone crosses `dwell_threshold_seconds`
  (default 5.0s), then stays silent until the entity exits and re-enters
  (a fresh dwell period), rather than firing every frame once past the
  threshold.
- On `DISAPPEARED`, all per-entity zone/dwell bookkeeping is cleaned up
  immediately — verified directly by inspecting the engine's internal
  state in `test_disappearing_inside_a_zone_does_not_leave_dangling_dwell_state`,
  not just by absence-of-crash.

## Caller contract (carried forward from Phase 5's own finding)
Feed this engine the same *filtered* (confirmed-or-better) entities and
zone occupancy that Phase 5 established — this engine trusts its inputs
and does not re-filter by lifecycle state itself, staying a single-purpose
diffing layer. Passing raw, unfiltered entities (including single-frame
`DETECTED` noise) would let a noise blip fire a real `APPEARED`/
`ZONE_ENTERED`/`DWELL` change.

## Verified (109/109 unit tests total, 13 new this phase)
Appeared fires once, not repeated; disappeared fires once, not repeated;
movement below threshold is silent, above threshold fires and rebases;
an entity's very first frame never immediately fires `MOVED`; motion-state
transitions fire `STATE_CHANGED` with correct from/to values, silent when
stable; zone entry/exit fire exactly on the transition, silent while
inside; dwell fires once after the threshold, stays silent while still
inside, and correctly fires again after an exit-then-reentry; disappearing
while inside a zone cleans up all dwell bookkeeping (checked directly, not
just "didn't crash").

## Real end-to-end validation (full 6-phase chain)
Re-ran the real-photo pan sequence (real detector, real tracker, real
World Model, real zones from Phase 5) through this engine with
`movement_threshold_px=25`, `dwell_threshold_seconds=4.0`:

```
frame  1  APPEARED
frame  3  MOVED   distance=53.0px
frame  4  MOVED   distance=31.8px
frame  6  MOVED   distance=59.0px
frame  7  MOVED   distance=28.5px
frame  9  MOVED   distance=41.4px
frame 11  MOVED   distance=39.5px
frame 12  ZONE_ENTERED
```

Exactly one `APPEARED`, no spurious repeats, `MOVED` firing only on
genuine threshold-crossing displacement (not every frame despite the
entity moving every frame), and `ZONE_ENTERED` at precisely the frame
Phase 5 independently identified as the real zone-crossing frame. No
`DWELL` fired — correctly, since the sequence ends only 1 second after
zone entry, below the 4-second threshold. No noise-triggered changes,
because the filtered-entities calling convention was followed.

## Known limitation (flagged, not hidden)
The movement-threshold interpretation above is a judgment call, not a
spec-mandated number — 30px is a placeholder default that will need
tuning once real camera resolution/distance/frame-rate is known; it has
no principled "correct" value in the abstract.

## Not in this phase (per the 10-phase plan)
Event schema, event lifecycle (Potential/Confirmed/Active/Resolved/
Archived), event-level deduplication across gaps, confidence propagation,
rules, alerts, evidence — all Phase 7.
