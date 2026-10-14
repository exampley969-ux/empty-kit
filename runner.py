"""
Vision Radar - Pipeline runner
Wires every layer built in Phases 1-8 into one running system and persists
what it produces:

  frame -> detect -> track -> world model -> zones -> state changes ->
  events -> rules -> alerts + evidence -> data layer

This is also the single place that enforces the calling contract found in
Phases 5-7: only confirmed-or-better entities (not single-frame `DETECTED`
noise) are fed to zone occupancy, the change engine and the event engine.

Failure handling (Section 26 - nothing fails silently): each stage reports
to the HealthRegistry. A failing stage skips that frame and is counted, it
does not kill the run; the next good frame flips the component back to OK.

Long-running hygiene (found in the Phase 9 review): exited tracks and
archived events are dropped from memory once they have been persisted, so a
service left running for weeks does not grow without bound. If persisting a
frame failed, purging is skipped for that frame so nothing unsaved is lost.
"""

from __future__ import annotations

import logging
import threading
import time

import cv2
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from vision_radar.data_layer.database import DataIntegrityError, DataStorageUnavailable
from vision_radar.data_layer.repository import Repository
from vision_radar.event_engine.engine import EventEngine
from vision_radar.event_engine.evidence import EvidenceBuilder
from vision_radar.event_engine.models import Event, EventConfig
from vision_radar.ingestion.exceptions import CameraConnectionError
from vision_radar.ingestion.models import CameraStatus, Frame, SessionState
from vision_radar.ingestion.session import IngestionSession
from vision_radar.monitoring.health import ComponentStatus, HealthRegistry
from vision_radar.monitoring.performance import PerformanceMonitor
from vision_radar.rules_engine.engine import RuleEngine
from vision_radar.rules_engine.models import Alert, Rule
from vision_radar.state_engine.engine import StateChangeEngine
from vision_radar.state_engine.models import StateChangeConfig
from vision_radar.tracking.models import TrackerConfig, TrackState
from vision_radar.tracking.tracker import IoUTracker
from vision_radar.vision.detector import Detector
from vision_radar.vision.pipeline import DetectionPipeline
from vision_radar.world_model.world_model import WorldModel
from vision_radar.world_model.zone import Zone, ZoneManager

logger = logging.getLogger("vision_radar.pipeline")


@dataclass
class FrameResult:
    skipped: bool = False
    detections: int = 0
    tracked: int = 0
    events: List[Event] = field(default_factory=list)
    alerts: List[Alert] = field(default_factory=list)


class PipelineRunner:
    def __init__(self, *, session: IngestionSession, detection_pipeline: DetectionPipeline,
                 tracker: IoUTracker, world_model: WorldModel, zone_manager: ZoneManager,
                 state_engine: StateChangeEngine, event_engine: EventEngine,
                 rule_engine: RuleEngine, evidence_builder: EvidenceBuilder,
                 repository: Repository, monitor: PerformanceMonitor, health: HealthRegistry,
                 camera_id: Optional[str] = None, entity_persist_every: int = 10,
                 session_persist_every: int = 50):
        self.session = session
        self.detection_pipeline = detection_pipeline
        self.tracker = tracker
        self.world_model = world_model
        self.zone_manager = zone_manager
        self.state_engine = state_engine
        self.event_engine = event_engine
        self.rule_engine = rule_engine
        self.evidence_builder = evidence_builder
        self.repo = repository
        self.monitor = monitor
        self.health = health
        self.camera_id = camera_id
        self.entity_persist_every = max(1, entity_persist_every)
        self.session_persist_every = max(1, session_persist_every)

        self.frames_processed = 0
        self.frames_failed = 0
        self.persist_failures = 0
        self._last_conf: Dict[str, float] = {}
        self._last_entity: Dict[str, object] = {}
        self._persisted_entities: set = set()
        self._counted_events: set = set()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._finished = False
        self._last_ts: Optional[float] = None
        self._latest_frame_lock = threading.Lock()
        self._latest_frame_jpeg: Optional[bytes] = None

    @property
    def session_id(self) -> str:
        return self.session.info.session_id

    # ---------------- lifecycle ----------------
    def begin(self) -> None:
        """Connect the camera and record the session. Raises if the camera
        cannot be opened - after flagging it, so the failure is visible."""
        try:
            self.session.start()
        except CameraConnectionError as exc:
            self.health.set("camera", ComponentStatus.ERROR, detail=str(exc))
            self._save_session()
            raise
        self._sync_camera_health()
        self._save_session()

    def run(self, max_frames: Optional[int] = None) -> None:
        try:
            while not self._stop.is_set():
                if max_frames is not None and self.frames_processed + self.frames_failed >= max_frames:
                    break
                frame = self.session.next_frame()
                if frame is None:
                    self._sync_camera_health()
                    break
                self.process_frame(frame)
        finally:
            self._finish()

    def start_background(self) -> None:
        self.begin()
        self._thread = threading.Thread(target=self.run, name=f"pipeline-{self.session_id[:8]}", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
        else:
            self._finish()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        try:
            if self.session.info.state == SessionState.RUNNING:
                self.session.stop()
        except Exception:                                   # noqa: BLE001 - already finishing
            logger.exception("Error stopping session %s", self.session_id)
        self._close_open_events()
        self._save_session()

    def _close_open_events(self) -> None:
        """Observation is over: resolve events still open (see
        EventEngine.close_open_events) so none is left ACTIVE forever."""
        if self._last_ts is None:
            return
        try:
            closed = self.event_engine.close_open_events(self._last_ts, reason="session_ended")
            if not closed:
                return
            alerts = self.rule_engine.evaluate(closed)
            for event in closed:
                self.repo.save_event(event, self.session_id)
            self._persist_alerts(alerts, frame_image=None)
        except (DataStorageUnavailable, DataIntegrityError):
            self.persist_failures += 1
            logger.exception("Closing open events for session %s failed", self.session_id)

    # ---------------- one frame ----------------
    def process_frame(self, frame: Frame) -> FrameResult:
        started = time.perf_counter()
        ts = frame.captured_at
        self._last_ts = ts

        # -- vision
        try:
            detections = self.detection_pipeline.process(frame.image, frame.session_id, frame.frame_id)
            self.health.set("vision", ComponentStatus.OK)
        except Exception as exc:                            # noqa: BLE001
            logger.exception("Vision stage failed on frame %s", frame.frame_id)
            self.health.set("vision", ComponentStatus.ERROR, detail=str(exc))
            self.frames_failed += 1
            return FrameResult(skipped=True)

        # -- tracking + world model
        try:
            tracks = self.tracker.update(detections.detections, timestamp=ts)
            entities = self.world_model.update(tracks)
            # The Phase 5-7 contract, enforced in one place: single-frame
            # DETECTED noise never reaches zones/changes/events.
            reliable = [e for e in entities if e.lifecycle_state != TrackState.DETECTED]
            self.health.set("tracking", ComponentStatus.OK)
        except Exception as exc:                            # noqa: BLE001
            logger.exception("Tracking stage failed on frame %s", frame.frame_id)
            self.health.set("tracking", ComponentStatus.DEGRADED, detail=str(exc))
            self.frames_failed += 1
            return FrameResult(skipped=True)

        # -- zones, changes, events, rules
        try:
            occupancy = self.zone_manager.compute_occupancy(reliable)
            changes = self.state_engine.update(reliable, zone_occupancy=occupancy, timestamp=ts)
            for e in reliable:                              # last-known values, so events about an
                self._last_conf[e.entity_id] = e.confidence  # entity that just vanished still carry
                self._last_entity[e.entity_id] = e            # its real confidence/location
            touched = self.event_engine.update(changes, timestamp=ts, confidence_lookup=dict(self._last_conf))
            alerts = self.rule_engine.evaluate(touched)
            self.health.set("events", ComponentStatus.OK)
        except Exception as exc:                            # noqa: BLE001
            logger.exception("Event stage failed on frame %s", frame.frame_id)
            self.health.set("events", ComponentStatus.ERROR, detail=str(exc))
            self.frames_failed += 1
            return FrameResult(skipped=True)

        # -- persist
        persisted_ok = self._persist(frame, reliable, touched, alerts)

        # -- housekeeping
        current_ids = {e.entity_id for e in reliable}
        for gone in [i for i in self._last_conf if i not in current_ids]:
            self._last_conf.pop(gone, None)
            self._last_entity.pop(gone, None)
        if persisted_ok:
            self.tracker.purge_exited()
            self.event_engine.purge_archived()

        # -- metrics + health
        new_events = [e for e in touched if e.event_id not in self._counted_events]
        for e in new_events:
            self._counted_events.add(e.event_id)
        for e in touched:
            if e.state.value == "ARCHIVED":
                self._counted_events.discard(e.event_id)
        self.monitor.record_events(len(new_events))
        self.monitor.record_frame((time.perf_counter() - started) * 1000.0,
                                  detections=len(detections.detections), tracked=len(reliable))
        self._update_latest_frame(frame.image)
        self._sync_camera_health()
        self.frames_processed += 1
        if self.frames_processed % self.session_persist_every == 0:
            self._save_session()
        return FrameResult(detections=len(detections.detections), tracked=len(reliable),
                           events=touched, alerts=alerts)

    # ---------------- helpers ----------------
    def _persist(self, frame: Frame, reliable, touched: Sequence[Event], alerts: Sequence[Alert]) -> bool:
        session_id = self.session_id
        try:
            current_ids = {e.entity_id for e in reliable}
            to_write = {e.entity_id: e for e in reliable
                        if e.entity_id not in self._persisted_entities
                        or frame.frame_id % self.entity_persist_every == 0}
            for gone in list(self._persisted_entities - current_ids):    # final snapshot of a departed entity
                if gone in self._last_entity:
                    to_write[gone] = self._last_entity[gone]
                self._persisted_entities.discard(gone)
            for entity in to_write.values():
                self.repo.upsert_entity(entity, session_id)
            # remember which *currently alive* entities have been written at least once
            alive_written = {i for i in to_write if i in current_ids}
            self._persisted_entities = (self._persisted_entities | alive_written) & current_ids

            for event in touched:                                        # events before alerts/evidence (FK order)
                self.repo.save_event(event, session_id)
            self._persist_alerts(alerts, frame_image=frame.image)
            return True
        except (DataStorageUnavailable, DataIntegrityError):
            # Already reflected in the "database" health component; count it
            # and keep processing rather than dropping the whole camera feed.
            self.persist_failures += 1
            logger.exception("Persisting frame %s failed", frame.frame_id)
            return False

    def _persist_alerts(self, alerts: Sequence[Alert], frame_image) -> None:
        """Alert + its evidence (with an annotated image when a frame is available)."""
        for alert in alerts:
            event = self.event_engine.get_event(alert.event_id)
            entity = self._last_entity.get(event.entity_ids[0]) if event.entity_ids else None
            zone = None
            if event.zone_id:
                try:
                    zone = self.zone_manager.get_zone(event.zone_id)
                except KeyError:
                    zone = None
            evidence = self.evidence_builder.build(event, entity=entity, zone=zone, frame=frame_image)
            self.repo.save_evidence(evidence)
            self.repo.save_alert(alert)

    def _save_session(self) -> None:
        try:
            self.repo.save_session(self.session.info, camera_id=self.camera_id)
        except (DataStorageUnavailable, DataIntegrityError):
            self.persist_failures += 1
            logger.exception("Saving session %s failed", self.session_id)

    def latest_frame_jpeg(self) -> Optional[bytes]:
        """The most recent frame this session processed, JPEG-encoded - the
        "live camera" view (Section 17-B). Only available while the session
        is actually running; there is deliberately no persisted history of
        raw frames (Section 28/29 - no unnecessary raw-video retention)."""
        with self._latest_frame_lock:
            return self._latest_frame_jpeg

    def _update_latest_frame(self, image) -> None:
        try:
            ok, buf = cv2.imencode(".jpg", image)
            if ok:
                with self._latest_frame_lock:
                    self._latest_frame_jpeg = buf.tobytes()
        except Exception:                                    # noqa: BLE001
            logger.exception("Failed to encode latest-frame preview for session %s", self.session_id)

    def _sync_camera_health(self) -> None:
        status = self.session.status()["camera_status"]
        if status == CameraStatus.CONNECTED.value:
            self.health.set("camera", ComponentStatus.OK)
        elif status == CameraStatus.RECONNECTING.value:
            self.health.set("camera", ComponentStatus.DEGRADED, detail="reconnecting")
        elif status == CameraStatus.DISCONNECTED.value and self.session.info.state == SessionState.STOPPED:
            self.health.set("camera", ComponentStatus.OK, detail="session stopped")
        else:
            self.health.set("camera", ComponentStatus.ERROR,
                            detail=self.session.info.error_message or status)


def build_pipeline_runner(*, session: IngestionSession, detectors: List[Detector],
                          repository: Repository, monitor: PerformanceMonitor,
                          health: HealthRegistry, evidence_dir: str,
                          zones: Sequence[Zone] = (), rules: Sequence[Rule] = (),
                          camera_id: Optional[str] = None,
                          tracker_config: Optional[TrackerConfig] = None,
                          state_config: Optional[StateChangeConfig] = None,
                          event_config: Optional[EventConfig] = None,
                          min_confidence: float = 0.3) -> PipelineRunner:
    """Assemble a runner with default-configured engines. Used by the API's
    session start and by tests, so the real wiring is what gets tested."""
    zone_manager = ZoneManager()
    for z in zones:
        zone_manager.add_zone(z)
    rule_engine = RuleEngine()
    for r in rules:
        rule_engine.add_rule(r)
    return PipelineRunner(
        session=session,
        detection_pipeline=DetectionPipeline(detectors, min_confidence=min_confidence),
        tracker=IoUTracker(tracker_config or TrackerConfig()),
        world_model=WorldModel(),
        zone_manager=zone_manager,
        state_engine=StateChangeEngine(state_config or StateChangeConfig()),
        event_engine=EventEngine(event_config or EventConfig()),
        rule_engine=rule_engine,
        evidence_builder=EvidenceBuilder(evidence_dir),
        repository=repository, monitor=monitor, health=health, camera_id=camera_id,
    )
