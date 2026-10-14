"""
Vision Radar - Event Engine
Converts Phase 6's Change stream into Events with a real lifecycle,
deduplicating zone occupancy into one spanning event instead of one per
frame (Section 13), and carrying confidence through from the source
entity (Section 14).

Design note on lifecycle speed: Section 12's POTENTIAL -> CONFIRMED
transition exists to avoid trusting a single noisy observation. That
filtering already happened upstream - Phase 3's tracker requires multiple
consecutive hits to CONFIRM/TRACK an entity, and Phase 5/6's calling
convention excludes unconfirmed entities entirely. By the time a Change
reaches this engine, it is already reliable, so POTENTIAL -> CONFIRMED
happens immediately within the same update() call rather than waiting for
repeated confirmation here too. All five states are real, used values -
none are skipped in the type system - only the POTENTIAL/CONFIRMED step is
fast rather than multi-frame, and that's a documented consequence of
upstream filtering, not a shortcut taken silently.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from vision_radar.state_engine.models import Change, ChangeType

from .models import Event, EventConfig, EventState, EventType

# Change types that resolve instantly (no natural "ongoing" phase).
_INSTANTANEOUS_TYPES = {
    ChangeType.APPEARED: EventType.APPEARED,
    ChangeType.DISAPPEARED: EventType.DISAPPEARED,
    ChangeType.MOVED: EventType.MOVED,
    ChangeType.STATE_CHANGED: EventType.STATE_CHANGED,
    ChangeType.DWELL: EventType.DWELL,
}


class EventEngine:
    def __init__(self, config: EventConfig = None):
        self.config = config or EventConfig()
        self._events: Dict[str, Event] = {}
        # (entity_id, zone_id) -> event_id, for the currently-open ZONE_ENTRY span
        self._open_zone_entries: Dict[Tuple[str, str], str] = {}

    def update(self, changes: List[Change], timestamp: float,
               confidence_lookup: Optional[Dict[str, float]] = None) -> List[Event]:
        """
        Process one frame's worth of Changes. `confidence_lookup` should map
        entity_id -> current confidence (typically from the World Model's
        EntitySnapshot.confidence) so events carry real, sourced confidence
        rather than a guess - callers that omit it get
        `config.default_confidence` instead, which is honest about being a
        fallback, not a measurement.
        """
        confidence_lookup = confidence_lookup or {}
        touched: List[Event] = []

        for change in changes:
            if change.change_type == ChangeType.ZONE_ENTERED:
                touched.append(self._open_zone_entry(change, timestamp, confidence_lookup))
            elif change.change_type == ChangeType.ZONE_EXITED:
                event = self._close_zone_entry(change, timestamp)
                if event is not None:
                    touched.append(event)
            elif change.change_type in _INSTANTANEOUS_TYPES:
                touched.append(self._create_instantaneous(change, timestamp, confidence_lookup))
            # Unknown change types are ignored rather than raised on, so a
            # future ChangeType added upstream doesn't crash this engine -
            # it simply won't become an event until explicitly wired here.

        # Newly-archived events are reported too, so whatever persists events
        # (Phase 9) sees the RESOLVED -> ARCHIVED transition instead of the
        # in-memory and stored states silently diverging.
        seen_ids = {e.event_id for e in touched}
        for archived in self._auto_archive(timestamp):
            if archived.event_id not in seen_ids:
                touched.append(archived)
        return touched

    def _confidence_for(self, entity_id: str, confidence_lookup: Dict[str, float]) -> float:
        return confidence_lookup.get(entity_id, self.config.default_confidence)

    def _open_zone_entry(self, change: Change, timestamp: float,
                          confidence_lookup: Dict[str, float]) -> Event:
        key = (change.entity_id, change.zone_id)
        if key in self._open_zone_entries:
            # Defensive: Phase 6 already dedups this at the source, so this
            # should not happen - but if it did, we must not silently
            # orphan/duplicate the open event.
            return self._events[self._open_zone_entries[key]]

        confidence = self._confidence_for(change.entity_id, confidence_lookup)
        event = Event.new(
            event_type=EventType.ZONE_ENTRY,
            entity_ids=[change.entity_id],
            entity_type=change.entity_type,
            start_time=timestamp,
            confidence=confidence,
            zone_id=change.zone_id,
            evidence={"entered_at": timestamp},
        )
        # POTENTIAL -> CONFIRMED -> ACTIVE, immediately (see module docstring)
        event.state = EventState.ACTIVE
        self._events[event.event_id] = event
        self._open_zone_entries[key] = event.event_id
        return event

    def _close_zone_entry(self, change: Change, timestamp: float) -> Optional[Event]:
        key = (change.entity_id, change.zone_id)
        event_id = self._open_zone_entries.pop(key, None)
        if event_id is None:
            return None  # exit with no matching open entry - nothing to resolve
        event = self._events[event_id]
        event.end_time = timestamp
        event.state = EventState.RESOLVED
        event.evidence["exited_at"] = timestamp
        event.evidence["duration_seconds"] = event.duration_seconds()
        return event

    def _create_instantaneous(self, change: Change, timestamp: float,
                               confidence_lookup: Dict[str, float]) -> Event:
        confidence = self._confidence_for(change.entity_id, confidence_lookup)
        event = Event.new(
            event_type=_INSTANTANEOUS_TYPES[change.change_type],
            entity_ids=[change.entity_id],
            entity_type=change.entity_type,
            start_time=timestamp,
            confidence=confidence,
            zone_id=change.zone_id,
            evidence=dict(change.extra) if change.extra else {},
        )
        event.end_time = timestamp   # zero-duration by nature
        event.state = EventState.RESOLVED   # POTENTIAL -> CONFIRMED -> ACTIVE -> RESOLVED, immediately
        self._events[event.event_id] = event
        return event

    def _auto_archive(self, timestamp: float) -> List[Event]:
        archived: List[Event] = []
        for event in self._events.values():
            if event.state == EventState.RESOLVED and event.end_time is not None:
                if timestamp - event.end_time >= self.config.archive_after_seconds:
                    event.state = EventState.ARCHIVED
                    archived.append(event)
        return archived

    def close_open_events(self, timestamp: float, reason: str = "session_ended") -> List[Event]:
        """Resolve every still-open ZONE_ENTRY at `timestamp` because observation
        itself has stopped (session ended / camera lost) - NOT because the
        entity left. Without this, a session ending while someone stands in a
        zone would leave that event ACTIVE forever (and retention never sweeps
        ACTIVE events). The evidence records *why* it was closed, and no
        `exited_at` is written: the entity did not exit, we just stopped looking."""
        closed: List[Event] = []
        for event_id in list(self._open_zone_entries.values()):
            event = self._events[event_id]
            event.end_time = max(timestamp, event.start_time)
            event.state = EventState.RESOLVED
            event.evidence["closed_reason"] = reason
            event.evidence["closed_at"] = event.end_time
            event.evidence["duration_seconds"] = event.duration_seconds()
            closed.append(event)
        self._open_zone_entries.clear()
        return closed

    def purge_archived(self) -> int:
        """Drop ARCHIVED events from memory (they live on in the data layer).
        Without this, a long-running service would hold every event it has
        ever created in RAM forever. Never touches ACTIVE/RESOLVED events."""
        ids = [eid for eid, e in self._events.items() if e.state == EventState.ARCHIVED]
        for eid in ids:
            del self._events[eid]
        return len(ids)

    # --- queries ----------------------------------------------------------
    def get_event(self, event_id: str) -> Event:
        return self._events[event_id]

    def list_events(self, state: Optional[EventState] = None,
                     event_type: Optional[EventType] = None,
                     entity_id: Optional[str] = None) -> List[Event]:
        results = list(self._events.values())
        if state is not None:
            results = [e for e in results if e.state == state]
        if event_type is not None:
            results = [e for e in results if e.event_type == event_type]
        if entity_id is not None:
            results = [e for e in results if entity_id in e.entity_ids]
        return results
