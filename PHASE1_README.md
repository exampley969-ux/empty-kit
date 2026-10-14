# Phase 1 — Video Ingestion Layer

Scope (per spec Section 4 only — nothing from later layers):
- `VideoSource` abstraction, source-independent (webcam index / file path / stream URL)
- Session start/stop
- Frame acquisition with timestamping
- Camera status (`DISCONNECTED / CONNECTING / CONNECTED / RECONNECTING / ERROR`)
- Automatic reconnect with backoff, on live-source drops only
- Multi-session tracking (`SessionManager`)

## Structure
```
vision_radar/
  ingestion/
    models.py       # enums + dataclasses (CameraStatus, SessionState, Frame, SessionInfo...)
    exceptions.py    # CameraConnectionError, FrameAcquisitionError, SessionError
    video_source.py  # VideoSource (abstract) + OpenCVVideoSource (webcam/file/stream)
    session.py       # IngestionSession + SessionManager
  tests/
    fakes.py             # FakeVideoSource test double (no hardware needed)
    test_video_source.py # 10 tests: connect/disconnect/reconnect/exhaustion
    test_session.py       # 8 tests: session lifecycle + multi-session manager
```

## Run tests
```
cd vision_radar_project
python3 -m unittest discover -s vision_radar/tests -p "test_*.py" -v
```
18/18 tests passing.

## Verified manually (not just unit tests)
- Real `OpenCVVideoSource` against a non-existent camera index correctly raises
  `CameraConnectionError` instead of hanging or crashing.
- Real `OpenCVVideoSource` against an actual 15-frame `.mp4` file reads exactly
  15 frames end-to-end through `IngestionSession`, then stops cleanly
  (no infinite loop on end-of-file — this was caught and fixed during
  evaluation: end-of-file was initially mistaken for a live disconnect,
  causing the file to reopen and restart forever).

## Known limitation (flagged, not silently ignored)
`fps_actual` is only meaningful for live sources. When reading a video file
back-to-back with no real-time pacing, it reports an inflated number (file
read speed, not playback speed). This does not affect correctness of frame
count, status, or reconnect logic — it will be addressed if/when frame-rate
pacing for file sources becomes a real requirement. Not fixed now to avoid
scope creep beyond Section 4.

## Not in this phase (comes later, per the 10-phase plan)
Detection, tracking, world model, zones, events, rules, API, DB persistence,
frontend — all untouched, as agreed.
