# Phase 3 — Vision Engine: Tracking & Position

Scope (per spec Section 5 - Tracking/Position, Section 6 - World Model
persistent context, Section 7 - Entity lifecycle, Section 8 - Spatial
model movement direction/trajectory):

- `IoUTracker` — assigns and maintains a persistent `track_id` for each
  entity across frames, matched per entity_type via the Hungarian algorithm
  (optimal assignment) on an IoU cost matrix — never via greedy nearest-match
- Full entity lifecycle exactly as specified:
  `DETECTED -> CONFIRMED -> TRACKED -> TEMPORARILY_LOST -> REACQUIRED -> EXITED`
- Position estimation: velocity (px/sec), 8-point compass direction label,
  bounded trajectory history (deque, default 30 points)

## Structure
```
vision_radar/tracking/
  models.py    # TrackState, TrackedPoint, TrackerConfig, Track
  tracker.py   # IoUTracker (Hungarian-matched, per-type, full lifecycle)
  position.py  # compute_velocity / compute_direction_label / compute_trajectory
vision_radar/tests/
  test_position.py   # 11 tests
  test_tracker.py     # 12 tests
```

## Design decisions worth knowing
- **Matching is per entity_type.** A "car" detection can never be assigned
  to a "person" track, even if their boxes overlap — this was unit-tested
  directly (`test_different_entity_types_never_share_a_track`).
- **Hungarian (optimal) assignment, not greedy**, since scipy was already
  available in this sandbox — minimizes identity swaps when two same-type
  entities cross paths, versus a simpler greedy nearest-IoU match.
- **An unconfirmed (`DETECTED`) track that misses even once exits
  immediately** rather than entering a recovery window. This is a
  deliberate choice: a single-frame detection with no follow-up is more
  likely sensor/model noise than a real entity, and the real-photo test
  below shows this filtering real noise correctly, not just in theory.
- **Position/motion stays in image-space pixels**, consistent with Section
  15's honesty rule from Phase 1 — no real-world units are claimed.

## Verified (60/60 unit tests total, 23 new this phase)
`test_tracker.py` covers: new-track creation, identity persistence under
small movement, two simultaneous entities never merging, cross-type
non-matching, far-apart detections *not* matching (correctly becoming a new
track instead), full lifecycle progression at exact hit-count thresholds,
temporarily-lost -> reacquired -> tracked recovery, exit after exceeding the
missed-frame limit, and immediate-exit-on-first-miss for unconfirmed
tracks.

`test_position.py` covers velocity math (including the zero-history and
zero-time-delta edge cases), all 8 compass directions plus the stationary
threshold, and trajectory extraction.

## Real end-to-end validation (not just unit tests)
Built a sequence of 13 frames by panning a fixed-size viewport across a
wider canvas containing the real motorcycle photo (from `skimage.data`,
offline) — so the *detections* each frame are genuinely produced by the
real Phase-2 YOLO vehicle detector on real image content, and only the
apparent camera motion is synthetic (there is no real moving-camera
pedestrian/traffic footage available offline in this sandbox - flagged
honestly, same as prior phases).

Result: a single track (`539aaf06...`) persisted across all 13 frames as
the motorcycle moved through frame, correctly progressing
`DETECTED -> CONFIRMED -> TRACKED` at the exact configured hit counts,
reporting direction `W` (westward/leftward on screen) consistently with
speed ~20-28 px/sec throughout. Three separate single-frame noise
detections (confidence 0.27-0.52, likely background clutter at the
canvas's gray padding edges) each started a track but were correctly never
promoted past `DETECTED` and dropped on their very next miss — the
lifecycle design filtering real noise, not hypothetical noise.

## Known limitation (flagged, not hidden)
No real footage of two same-type entities (e.g. two people) actually
crossing paths was available offline to validate identity-swap resistance
under occlusion/crossing — that's tested at the unit level with
synthetic detections (`test_two_simultaneous_entities_get_different_track_ids`)
but not against a real crossing-paths video clip. Worth re-validating once
real camera footage is available.

## Not in this phase (per the 10-phase plan)
Zones, camera calibration, relationship engine (Phase 5). Change/state
events (appeared/entered-zone/dwelled etc. - Phase 6). This phase only
establishes *that* an entity has a persistent identity and *where* it is/was
moving — not what that means yet.
