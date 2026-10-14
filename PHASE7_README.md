# Phase 7 — Event Engine

Scope (per spec Section 11 - Event schema, Section 12 - Event lifecycle,
Section 13 - Event deduplication, Section 14 - Confidence system), per
the 10-phase plan. Rule engine, alerts, and rich evidence attachment
(frame snapshots) are explicitly Phase 8 — not here.

## Structure
```
vision_radar/event_engine/
  models.py   # EventState (5 lifecycle states), EventType, EventConfig, Event
  engine.py   # EventEngine: Change -> Event, dedup, lifecycle, archival
vision_radar/tests/
  test_event_engine.py   # 14 tests
```

## Event schema (Section 11)
`Event` matches the spec's own field list exactly: event_id, event_type,
entity_ids, zone_id, start_time, end_time, confidence, evidence, status
(here `state`). `duration_seconds()` mirrors the worked example directly
(`Start: 14:32:08, End: 14:32:26, Duration: 18 sec`).

## Deduplication (Section 13) — the core design decision
A `ZONE_ENTERED` change opens exactly one `ZONE_ENTRY` event in `ACTIVE`
state; while the entity stays inside, Phase 6 emits no further zone
changes, so no further events are created; the matching `ZONE_EXITED`
resolves that *same* event (same `event_id`) with `end_time` and duration
set. This is Section 13's own example implemented literally: one event
per real occupancy span, not one per frame.

All other change types (`APPEARED`, `DISAPPEARED`, `MOVED`,
`STATE_CHANGED`, `DWELL`) have no natural "ongoing" phase — they resolve
immediately with `start_time == end_time` (zero duration by nature, not a
bug).

## Lifecycle (Section 12) — a documented speed decision, not a shortcut
All 5 states (`POTENTIAL/CONFIRMED/ACTIVE/RESOLVED/ARCHIVED`) are real,
used values. `POTENTIAL -> CONFIRMED` collapses immediately within the
same `update()` call rather than waiting across frames, because the
filtering that state exists for already happened upstream: Phase 3's
tracker requires multiple consecutive hits before an entity is even
`CONFIRMED`/`TRACKED`, and Phase 5/6's calling convention excludes
`DETECTED`-only entities entirely. Re-implementing a second confirmation
delay here would just add latency without adding reliability. This is
documented explicitly (in `engine.py`'s module docstring too) so it's not
mistaken for a spec-skipping shortcut.

`RESOLVED -> ARCHIVED` is time-based (`archive_after_seconds`, default
60s) and checked every `update()` call — verified to never touch an
`ACTIVE` event no matter how much time passes (an ongoing zone visit must
never silently archive out from under itself).

## Confidence (Section 14)
Every event's confidence is sourced from the involved entity's *current*
tracking confidence (passed in via `confidence_lookup`, typically built
from `EntitySnapshot.confidence`) — never fabricated. If a caller omits
the lookup, `config.default_confidence` (0.5) is used and is clearly
documented as a fallback, not a measurement.

## Verified (124/124 unit tests total, 14 new this phase)
Instantaneous event creation/confidence/evidence-passthrough; the full
zone-entry dedup story (one event across entry+exit, correct duration, no
event while merely staying inside, exactly one event across a full
entry-exit cycle, exit-with-no-open-entry ignored rather than crashing,
independent entities/zones tracked independently so one exit never
resolves another's event); auto-archival timing (including the "never
archives an ACTIVE event" guard); and query filtering by state/type/entity.

## Real end-to-end validation (full 7-phase chain)
Extended the real-photo pan sequence (real detector -> real tracker ->
World Model -> zones -> Phase 6's Change engine) into this Event Engine,
threading real per-frame detection confidence through as
`confidence_lookup`:

```
frame  1  APPEARED    RESOLVED  conf=0.79  dur=0.0s
frame  3  MOVED        RESOLVED  conf=0.71  dur=0.0s
frame  4  MOVED        RESOLVED  conf=0.71  dur=0.0s
frame  6  MOVED        RESOLVED  conf=0.83  dur=0.0s
frame  7  MOVED        RESOLVED  conf=0.87  dur=0.0s
frame  9  MOVED        RESOLVED  conf=0.82  dur=0.0s
frame 11  MOVED        RESOLVED  conf=0.83  dur=0.0s
frame 12  ZONE_ENTRY   ACTIVE    conf=0.85  ongoing
```

Confidence values are real, varying per-frame detector output (0.71-0.87),
not fabricated or fixed — carried all the way from Phase 2's YOLO model
through five intermediate layers into the final Event record. Exactly one
event per real semantic change (matching Phase 6's own finding
frame-for-frame), and `ZONE_ENTRY` correctly stays `ACTIVE`/ongoing since
the 13-frame sequence ends before any exit occurs — no premature
resolution.

## Known limitation (flagged, not hidden)
`archive_after_seconds` (default 60s) is a placeholder, not a
spec-mandated value — real retention policy depends on deployment/storage
requirements not yet defined. Also, this phase's `evidence` field is a
lightweight context dict (positions, timestamps); actual frame-image
evidence attachment is explicitly Phase 8's job (Section 20).

## Not in this phase (per the 10-phase plan)
Rule engine (WHEN/IF/THEN), alert generation, rich evidence attachment
(frame snapshots, zone geometry rendering) — Phase 8.
