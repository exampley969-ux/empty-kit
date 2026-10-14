# Phase 2 — Vision Engine: Detection & Classification

Scope (per spec Section 5 only, plus Section 14's confidence requirement):
- Generic `Detector` interface (entity-type agnostic)
- `HOGPersonDetector` — real, working, offline (SVM weights bundled in OpenCV, no download)
- `HaarCascadeDetector` — generic, works with any bundled Haar cascade (used here for `face` as a second real example, proving the interface is genuinely pluggable, not hardcoded to "person")
- `DetectionPipeline` — runs multiple detectors, filters by confidence, de-duplicates overlapping same-type boxes (NMS) without letting different entity types suppress each other
- Confidence stored per detection

## Structure
```
vision_radar/vision/
  models.py     # BoundingBox (+ IoU, normalization), Detection, FrameDetections
  detector.py   # Detector (abstract), HOGPersonDetector, HaarCascadeDetector
  pipeline.py   # DetectionPipeline (aggregate + filter + per-type NMS)
vision_radar/tests/
  fake_detectors.py     # FakeDetector test double
  test_bounding_box.py   # 6 tests
  test_pipeline.py       # 8 tests
```

## Explicit scope decision — no fake vehicle detector
This sandbox has no internet access, so no YOLO/vehicle-trained weights could
be downloaded. OpenCV ships with a real, working person detector (HOG) and
Haar cascades for faces/bodies/plates, but nothing for generic vehicles.

Rather than fake a "vehicle detector" that doesn't actually work, Phase 2
ships a genuinely pluggable `Detector` interface plus one fully working
concrete detector (person) and one more (face) to prove the interface isn't
hardcoded. A real vehicle detector can be added later by implementing the
same interface — no other code changes.

## Honesty note on confidence (Section 14)
Neither HOG's SVM decision value nor a Haar cascade's stage-reject weight is
a calibrated probability. Both are passed through a sigmoid to land in
(0,1) for storage/comparison — this is a monotonic normalization, not a
statistical confidence. Documented in `detector.py` so it isn't
misrepresented later.

## Verified (31/31 unit tests + manual smoke tests)
- `test_bounding_box.py` — IoU, normalization, edge cases (all synthetic, deterministic)
- `test_pipeline.py` — multi-detector aggregation, confidence filtering, and
  critically: NMS correctly deduplicates overlapping same-type boxes but
  does **not** suppress across different entity types (a face detection
  inside a person's bounding box is kept, not merged away)
- Real `HOGPersonDetector` + `HaarCascadeDetector` run on synthetic
  blank/noisy images: no crash, correct return types, and — correctly —
  zero false-positive detections (there's no real person/face in random
  noise, so zero is the right answer, not a bug)
- Full chain smoke test: real `IngestionSession` (Phase 1) feeding a real
  `.mp4` file's frames directly into `DetectionPipeline` (Phase 2) end to
  end — 8/8 frames processed without error

## Known limitation (flagged, not hidden)
We cannot validate real-world *detection accuracy* (e.g. "does it actually
find a person in a real photo") inside this offline sandbox, since there's
no way to fetch a real pedestrian image without internet access. What's
verified here is that the plumbing (interfaces, filtering, NMS, integration
with Phase 1) is correct — accuracy validation against real footage should
happen once this runs in an environment with real camera input.

## Not in this phase (per the 10-phase plan)
Tracking / identity persistence across frames, position/trajectory
tracking — that's Phase 3.

---

## UPDATE — Vehicle detection gap closed

The "no fake vehicle detector" limitation above is resolved. The user
downloaded a real `yolov8n.onnx` (Ultralytics, COCO-trained, exported via
their own Colab session) and provided it directly — no internet was used
from inside this sandbox at any point.

### What was added
- `vision_radar/vision/onnx_yolo_detector.py`
  - `decode_yolov8_output(...)` — pure post-processing function (letterbox
    undo, confidence filter, class filter), unit-testable with hand-crafted
    arrays, no model required
  - `YoloOnnxDetector` — real `Detector` implementation backed by
    `onnxruntime` (CPU), loads class names straight from the model's own
    embedded metadata
  - `VehicleDetector(model_path)` — convenience factory restricted to
    `car / motorcycle / bus / truck`
- `vision_radar/models/yolov8n.onnx` — the model file itself, now part of
  the shipped project
- `onnxruntime` added to `requirements.txt`

### Verified with REAL accuracy validation (not just plumbing)
Unlike Phase 2's original person/face detectors (which could only be
smoke-tested on synthetic images), this was validated against **real
photographs** — found bundled offline inside the already-installed
`scikit-image` package (`skimage.data`), so no internet was needed for this
either:

| Real photo | Contains a vehicle? | Vehicle-only detector result |
|---|---|---|
| `motorcycle_left.png` | Yes (motorcycle) | ✅ 1 `motorcycle`, confidence 0.86, after NMS collapsed 10 raw overlapping boxes to 1 |
| `astronaut.png` | No | ✅ 0 vehicle detections (all-class run correctly found `person` instead) |
| `coffee.png` | No | ✅ 0 vehicle detections (all-class run correctly found `cup`, `dining table`) |
| `chelsea.png` (a cat) | No | ✅ 0 vehicle detections (all-class run correctly found `cat`) |

This confirms both true positives (finds a real vehicle when one is
present) and true negatives (no false alarms on real photos with no
vehicle) — a meaningfully stronger check than the earlier synthetic-image
smoke tests.

Plus: 6 new unit tests (`test_onnx_yolo_decode.py`) covering confidence
filtering, class filtering, and letterbox coordinate correction — all
passing (37/37 total across the whole project now).

### Still honestly flagged
- Only one real test image per category was available offline (no dataset
  of hundreds of photos to compute precision/recall). This is a real,
  correct spot-check, not a statistically rigorous accuracy benchmark —
  that would need a proper labeled test set once real camera/deployment
  footage exists.
- `person` detection now effectively has two independent detectors
  available (`HOGPersonDetector` from earlier, and YOLO's own `person`
  class if the filter is opened up) — no conflict, but worth deciding in a
  later phase whether to run both or standardize on one.

