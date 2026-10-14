"""
Vision Radar - Natural-language search
Converts a plain-English question into a structured EventQuery (Phase 9),
which is then answered from stored events - never from the video, never
from a language model.

Spec reference: PDF Section 19: "The system converts that into structured
queries. It should answer from stored events, not hallucinate from the
video. This distinction is extremely important."

This is a small, deterministic, pattern-based parser - not an LLM call.
That is a deliberate honesty choice, not a shortcut: an LLM could produce a
plausible-looking query that silently means something different from what
was asked, which is exactly the hallucination risk Section 19 warns against.
A parser either recognises a pattern or says so; it never guesses.

Supported patterns (deliberately small, matched to what Sections 18-19
actually ask for - not a general NL-to-SQL engine):
  - event type: "zone entry/entries", "appeared", "disappeared", "moved",
    "dwell(ed)", "state change(d)"
  - entity type: "person/people", "car(s)", "motorcycle(s)", "bus(es)", "truck(s)"
  - zone: "in/in zone <name>", "in the <name> zone" (matched by name against
    the zones the caller passes in - never invented)
  - confidence: "confidence > 0.8", "confidence above 90%"
  - time range: "between 2pm and 4pm", "today", "last hour", "last 30 minutes"
  - status/state: "active", "resolved", "still active", "ongoing"
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, time as dtime, timedelta, timezone
from typing import Callable, List, Optional

from vision_radar.data_layer.repository import EventQuery
from vision_radar.event_engine.models import EventState, EventType

# Phrase -> EventType. Longest/most-specific phrases first so "zone entry"
# doesn't get swallowed by a looser pattern.
_EVENT_TYPE_PHRASES = [
    (r"zone[\s-]?entr(?:y|ies)", EventType.ZONE_ENTRY),
    (r"appear(?:ed|ances?)?", EventType.APPEARED),
    (r"disappear(?:ed|ances?)?", EventType.DISAPPEARED),
    (r"dwell(?:ed|s|ing)?", EventType.DWELL),
    (r"state[\s-]?chang(?:e|ed|es)", EventType.STATE_CHANGED),
    (r"mov(?:ed|ement|es)", EventType.MOVED),
]

_ENTITY_TYPE_PHRASES = [
    (r"\bpeople\b|\bperson(?:s)?\b", "person"),
    (r"\bmotorcycles?\b", "motorcycle"),
    (r"\btrucks?\b", "truck"),
    (r"\bbus(?:es)?\b", "bus"),
    (r"\bcars?\b|\bvehicles?\b", "car"),
]

_STATE_PHRASES = [
    (r"still active|ongoing|currently active|\bactive\b", EventState.ACTIVE),
    (r"resolved|finished|completed|\bended\b", EventState.RESOLVED),
    (r"archived", EventState.ARCHIVED),
]

_CONF_RE = re.compile(r"confidence\s*(?:is\s*)?(above|over|below|under|>=?|<=?)\s*(\d+(?:\.\d+)?)\s*%?")
_BETWEEN_RE = re.compile(
    r"between\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\s+and\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)")
_ZONE_RE = re.compile(r"\bin\s+(?:the\s+)?zone\s+([a-z0-9][\w\- ]*?)(?=\s+(?:zone\b|between|today|"
                      r"confidence|last\s|that\s|which\s|with\s)|[.?!]|$)", re.IGNORECASE)
_ZONE_SUFFIX_RE = re.compile(r"\bin\s+(?:the\s+)?([a-z0-9][\w\- ]*?)\s+zone\b", re.IGNORECASE)
_LAST_N_RE = re.compile(r"last\s+(\d+)\s*(minute|hour|day)s?")


@dataclass
class ParsedQuery:
    query: EventQuery
    matched_event_type: bool
    matched_zone_name: Optional[str]
    unmatched_zone_name: Optional[str]   # a "zone X" phrase that named no real zone
    notes: List[str]


def _parse_clock(text: str, ref: datetime) -> Optional[datetime]:
    text = text.strip().lower().replace(" ", "")
    m = re.match(r"^(\d{1,2})(?::(\d{2}))?(am|pm)?$", text)
    if not m:
        return None
    hour, minute, ampm = int(m.group(1)), int(m.group(2) or 0), m.group(3)
    if ampm == "pm" and hour != 12:
        hour += 12
    if ampm == "am" and hour == 12:
        hour = 0
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return datetime.combine(ref.date(), dtime(hour, minute), tzinfo=ref.tzinfo)


def parse(text: str, *, known_zone_names: Optional[dict] = None,
          now: Optional[Callable[[], datetime]] = None) -> ParsedQuery:
    """
    known_zone_names: {zone_name (any case): zone_id} - the caller's real
    zones. A name mentioned in the question that isn't in here is reported
    as `unmatched_zone_name`, never guessed at or invented.
    """
    now_dt = (now or (lambda: datetime.now(timezone.utc)))()
    raw = text
    lowered = text.lower()
    notes: List[str] = []

    q = EventQuery()
    matched_event_type = False

    for pattern, etype in _EVENT_TYPE_PHRASES:
        if re.search(pattern, lowered):
            q.event_type = etype
            matched_event_type = True
            break

    for pattern, ent_type in _ENTITY_TYPE_PHRASES:
        if re.search(pattern, lowered):
            q.entity_id = None   # entity TYPE, not entity ID - see note below
            notes.append(f"entity_type:{ent_type}")   # surfaced for the caller; EventQuery has no entity_type filter yet
            break

    for pattern, state in _STATE_PHRASES:
        if re.search(pattern, lowered):
            q.state = state
            break

    m = _CONF_RE.search(lowered)
    if m:
        op, value = m.group(1), float(m.group(2))
        value = value / 100.0 if value > 1.0 else value
        if op in ("above", "over", ">", ">="):
            q.min_confidence = value
        else:
            notes.append(f"confidence_below:{value}")   # EventQuery only supports a floor; see README

    matched_zone, unmatched_zone = None, None
    zone_match = _ZONE_RE.search(lowered) or _ZONE_SUFFIX_RE.search(lowered)
    if zone_match:
        name = zone_match.group(1).strip()
        zones = {k.lower(): v for k, v in (known_zone_names or {}).items()}
        if name in zones:
            q.zone_id = zones[name]
            matched_zone = name
        else:
            unmatched_zone = name

    m = _BETWEEN_RE.search(lowered)
    if m:
        start, end = _parse_clock(m.group(1), now_dt), _parse_clock(m.group(2), now_dt)
        if start and end:
            q.start_time_from, q.start_time_to = start.timestamp(), end.timestamp()
        else:
            notes.append("unparsed_time_range")
    elif "today" in lowered:
        start = datetime.combine(now_dt.date(), dtime.min, tzinfo=now_dt.tzinfo)
        q.start_time_from, q.start_time_to = start.timestamp(), now_dt.timestamp()
    else:
        m = _LAST_N_RE.search(lowered)
        if m:
            n, unit = int(m.group(1)), m.group(2)
            seconds = n * {"minute": 60, "hour": 3600, "day": 86400}[unit]
            q.start_time_from = (now_dt - timedelta(seconds=seconds)).timestamp()
        elif "last hour" in lowered:
            q.start_time_from = (now_dt - timedelta(hours=1)).timestamp()

    return ParsedQuery(query=q, matched_event_type=matched_event_type,
                       matched_zone_name=matched_zone, unmatched_zone_name=unmatched_zone,
                       notes=notes)
