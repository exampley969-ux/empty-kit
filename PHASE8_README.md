# Phase 8 — Rule Engine, Alerts & Evidence

Scope (per spec Section 22 - Rule engine, Section 21 - Alerts, Section 20 -
Evidence), per the 10-phase plan.

## Structure
```
vision_radar/rules_engine/
  models.py   # FieldCondition, RuleTrigger, Rule, ActionType, Alert
  engine.py   # RuleEngine
vision_radar/event_engine/
  evidence.py # Evidence, EvidenceBuilder (new this phase)
  models.py   # Event gained `entity_type` (see schema note below)
vision_radar/tests/
  test_rules_engine.py   # 14 tests
  test_evidence.py        # 7 tests
```

## Small, disclosed schema change to Phase 7's Event
Rule matching needs to check entity type (Section 22's own example:
"WHEN Person enters Zone A"), but Phase 7's `Event` only stored
`entity_ids`, not the type. Added `entity_type: str` to `Event`, populated
by `EventEngine` from the source `Change` (which already carried it).
This is a minimal, necessary schema completion discovered while wiring
Phase 8 to Phase 7 — not a rewrite of anything Phase 7 already did. All
124 of Phase 7's own tests still pass unchanged after the addition.

## Rule engine (Section 22) — WHEN/IF/THEN, literally
- **WHEN** (`RuleTrigger`): event_type (required) + optional entity_type +
  optional zone_id.
- **IF** (`FieldCondition`, AND'ed list): a deliberately small resolver
  supporting exactly the spec's own examples — `confidence`,
  `duration_seconds`, and `evidence.<key>` (e.g. `evidence.dwell_seconds`)
  — with standard comparison operators. Not a general expression
  language; unsupported operators raise rather than silently doing
  nothing.
- **THEN** (`ActionType`): exactly one action, `GENERATE_ALERT`. Section
  21's own example also lists "Create Zone Entry Event" and "Record
  event" as THEN-actions, but those already happen elsewhere (event
  creation is Phase 7; DB persistence is Phase 9) — re-triggering them
  here would duplicate work across layers. Only the action that's
  genuinely new at this layer is implemented; no other action types were
  invented ahead of a real need for them.

## Alerts (Section 21)
Generated *only* by `RuleEngine.evaluate()` matching a real `Event` against
a defined `Rule` — never independently of a rule, matching Section 21's
explicit requirement ("Alerts are generated only from defined events").

## Evidence (Section 20) — a real feature, not a stub
`EvidenceBuilder` assembles timestamp, entity location, zone geometry, and
confidence — and when a real frame is supplied, it actually draws the
entity's bounding box and the zone polygon onto a copy of that frame and
saves it as a real `.jpg` to disk, keyed by event_id. This was
specifically verified to be a real, correctly-annotated image (reloaded
and pixel-compared against the original, not just checked for file
existence) — see the "Real end-to-end validation" section below for the
actual saved image.

## Verified (147/147 unit tests total, 21 new this phase)
Rule engine: WHEN matching on each field independently, all supported IF
operators (including the spec's own two worked examples — confidence
threshold and dwell-duration threshold), an unsupported operator raising
rather than silently passing, `duration_seconds` correctly evaluating
False (not crashing) against a still-`ACTIVE` event, empty IF-list always
matching, multiple rules independently firing on the same event, and
rule list/remove/get. Evidence: correct fields with and without a frame,
timestamp correctly picked (end_time when resolved, start_time when still
active), zone geometry passthrough, and — for the real-frame case — the
saved file actually exists, is non-empty, reloads as a correctly-sized
image, and its pixels are verifiably different from the unannotated
original (proving the overlay was actually drawn, not skipped).

## Real end-to-end validation (full 8-phase chain, with a real alert and a real evidence photo)
Defined a real `Rule` ("Vehicle enters restricted zone": motorcycle +
this specific zone + confidence > 0.5) and ran it against the real-photo
pan sequence's real Events (Phase 7). It fired exactly once, at the same
frame Phases 5-7 already independently identified as the true zone-
crossing frame:

```
ALERT: Rule 'Vehicle enters restricted zone' triggered: motorcycle zone
entry in zone <id> (confidence 0.85)
```

`EvidenceBuilder` then built real evidence from that alert's event,
drawing the actual detected bounding box (red) and the actual zone
boundary (green) onto that real frame and saving it to disk. Opened the
resulting image directly: the red box correctly bounds the real
motorcycle, and the green line sits exactly at the zone boundary the
motorcycle is crossing — visually confirming, not just asserting, that
Section 20's goal ("the user can inspect why the system generated the
event") is genuinely met.

## Known limitation (flagged, not hidden)
Evidence images are written to a local directory (`output_dir` passed at
construction) with no retention/cleanup policy yet — that belongs to
Phase 9's Data Layer, which will decide real persistence and retention
for evidence alongside events, sessions, and configuration.

## Not in this phase (per the 10-phase plan)
Data layer schema/persistence (Camera/Session/Entity/Track/Zone/Event/
Rule/Alert/Evidence/SystemState tables), the backend API, auth, retention
enforcement, performance monitoring — all Phase 9.
