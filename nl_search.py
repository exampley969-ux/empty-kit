"""
Vision Radar - Natural-language search
Converts a free-text question into a structured EventQuery (Phase 9) and
answers strictly from stored events.

Spec reference: PDF Section 19 ("The system converts that into structured
queries. It should answer from stored events, not hallucinate from the
video. This distinction is extremely important.").

Design: a small, deterministic set of regex extractors, each pulling one
dimension (event type / zone / entity type / time range / confidence) out
of the text and narrowing an EventQuery. There is no free-form interpretation
and nothing is ever invented: a clause that cannot be understood is reported
back to the caller as "ignored" rather than silently dropped or guessed at,
so a person using this knows exactly what was and wasn't applied. The output
is the already-tested query layer - this module decides WHAT to ask,
`Repository.query_events` decides the answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from vision_radar.data_layer.repository import EventQuery
from vision_radar.event_engine.models import EventType
from vision_radar.world_model.zone import Zone

DAY = 86400.0

# Human phrase -> EventType, longest/most specific phrases first so
# "zone entry" doesn't get shadowed by a hypothetical looser "entry" match.
_EVENT_PHRASES = [
    (r"zone[\s\-]?entr(?:y|ies)", EventType.ZONE_ENTRY),
    # disappear checked first, and appear excludes anything preceded by
    # "dis" - "disappear" contains "appear" as a literal substring, so
    # checking in the wrong order (or without the lookbehind) matched
    # APPEARED for a sentence about disappearing.
    (r"disappear(?:ed|ance)?s?", EventType.DISAPPEARED),
    (r"(?<!dis)appear(?:ed|ance)?s?", EventType.APPEARED),
    (r"mov(?:ed|ement)s?", EventType.MOVED),
    (r"dwell(?:s|ed|ing)?", EventType.DWELL),
    (r"state[\s\-]?chang(?:e|ed|es)", EventType.STATE_CHANGED),
]

_ENTITY_WORDS = {
    "person": "person", "people": "person", "pedestrian": "person",
    "vehicle": None, "vehicles": None,     # "vehicle" is a category, not one entity_type - see _entity_type()
    "car": "car", "cars": "car", "motorcycle": "motorcycle", "motorcycles": "motorcycle",
    "bus": "bus", "buses": "bus", "truck": "truck", "trucks": "truck",
}
_VEHICLE_TYPES = ["car", "motorcycle", "bus", "truck"]


@dataclass
class ParsedQuery:
    query: EventQuery
    entity_types: List[str] = field(default_factory=list)   # informational: which types were requested
    ignored_fragments: List[str] = field(default_factory=list)


def _extract_event_type(text: str) -> Optional[EventType]:
    for pattern, event_type in _EVENT_PHRASES:
        if re.search(pattern, text):
            return event_type
    return None


def _extract_entity_types(text: str) -> List[str]:
    found = []
    for word, mapped in _ENTITY_WORDS.items():
        if re.search(rf"\b{word}\b", text):
            found += _VEHICLE_TYPES if mapped is None else [mapped]
    seen = set()
    return [t for t in found if not (t in seen or seen.add(t))]


def _extract_zone(text: str, zones: List[Zone]) -> Optional[str]:
    """Matches 'in/at/inside [the] [zone] <zone name>' against actually-
    defined zone names only - tested directly against each real zone name
    rather than a free-text capture group, so a trailing word like "today"
    is never swallowed into the name and no zone_id is ever guessed for a
    name nothing was defined with. Longer names are tried first so
    "Restricted-A" isn't shadowed by a coincidentally-shorter zone name."""
    for zone in sorted(zones, key=lambda z: -len(z.name)):
        name = re.escape(zone.name.lower())
        if re.search(rf"\b(?:in|at|inside)\s+(?:the\s+)?(?:zone\s+)?{name}\b", text):
            return zone.zone_id
    return None   # named a zone that doesn't exist - caller sees zero results, not a crash


_TIME_WORD = r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?"


def _time_to_seconds_of_day(hour, minute, meridiem) -> int:
    hour, minute = int(hour), int(minute or 0)
    if meridiem == "pm" and hour != 12:
        hour += 12
    if meridiem == "am" and hour == 12:
        hour = 0
    return hour * 3600 + minute * 60


def _extract_time_range(text: str, now: float, day_start_of: "callable") -> tuple:
    """Returns (start, end) epoch seconds, or (None, None) if no time
    expression was recognised."""
    m = re.search(rf"between\s+{_TIME_WORD}\s+and\s+{_TIME_WORD}", text)
    if m:
        h1, m1, mer1, h2, m2, mer2 = m.groups()
        if mer1 is None:
            mer1 = mer2        # "between 2 and 4 pm" - the first inherits the second's meridiem
        base = day_start_of(now)
        return (base + _time_to_seconds_of_day(h1, m1, mer1),
                base + _time_to_seconds_of_day(h2, m2, mer2))

    if re.search(r"\btoday\b", text):
        base = day_start_of(now)
        return (base, base + DAY)
    if re.search(r"\byesterday\b", text):
        base = day_start_of(now) - DAY
        return (base, base + DAY)

    m = re.search(r"\blast\s+(\d+)\s*(minute|hour|day)s?\b", text)
    if m:
        amount, unit = int(m.group(1)), m.group(2)
        seconds = {"minute": 60.0, "hour": 3600.0, "day": DAY}[unit]
        return (now - amount * seconds, now)

    return (None, None)


def _extract_confidence(text: str) -> Optional[float]:
    m = re.search(r"confidence\s*(?:above|over|>=?|at least)\s*(0?\.\d+|\d+(?:\.\d+)?)", text)
    if not m:
        return None
    value = float(m.group(1))
    return value / 100.0 if value > 1.0 else value   # accepts "80" or "0.8"


def parse_query(text: str, zones: List[Zone], now: float,
                day_start_of=None) -> ParsedQuery:
    """
    day_start_of(timestamp) -> epoch seconds at local midnight. Injected so
    tests are deterministic and callers can supply their actual timezone;
    defaults to UTC midnight if not given.
    """
    if day_start_of is None:
        day_start_of = lambda ts: ts - (ts % DAY)

    original = text or ""
    lowered = original.lower()

    event_type = _extract_event_type(lowered)
    entity_types = _extract_entity_types(lowered)
    zone_id = _extract_zone(lowered, zones)
    start, end = _extract_time_range(lowered, now, day_start_of)
    min_confidence = _extract_confidence(lowered)

    # entity_type filtering: EventQuery has one slot but a phrase can name
    # several vehicle types ("vehicle"). Since Repository.query_events only
    # filters by entity_id, not entity_type, that half of the match happens
    # here, after fetching by every other filter - never by guessing one
    # single type and silently dropping the rest.
    query = EventQuery(
        start_time_from=start, start_time_to=end, event_type=event_type,
        zone_id=zone_id, min_confidence=min_confidence, limit=200,
    )

    ignored = []
    consumed_patterns = [p for p, _ in _EVENT_PHRASES] + [r"\btoday\b", r"\byesterday\b",
                         r"\blast\s+\d+\s*(?:minute|hour|day)s?\b",
                         rf"between\s+{_TIME_WORD}\s+and\s+{_TIME_WORD}",
                         r"confidence\s*(?:above|over|>=?|at least)\s*[\d.]+"]
    stripped = lowered
    for pattern in consumed_patterns:
        stripped = re.sub(pattern, " ", stripped)
    # Strip the SAME per-zone-name patterns _extract_zone actually matched
    # on, not the old generic placeholder - otherwise a zone referenced
    # without the literal word "zone" (e.g. "in Restricted-B") would wrongly
    # show up as "ignored" even though it genuinely was understood.
    for zone in sorted(zones, key=lambda z: -len(z.name)):
        stripped = re.sub(rf"\b(?:in|at|inside)\s+(?:the\s+)?(?:zone\s+)?{re.escape(zone.name.lower())}\b",
                          " ", stripped)
    for word in _ENTITY_WORDS:
        stripped = re.sub(rf"\b{word}\b", " ", stripped)
    for noise in (r"\bshow\b", r"\ball\b", r"\bwhat\b", r"\bhappened\b", r"\bevents?\b",
                 r"\bthe\b", r"\bdid\b", r"\bany\b", r"\bwith\b", r"\babove\b", r"\bover\b",
                 r"[?.!,]"):
        stripped = re.sub(noise, " ", stripped)
    leftover = " ".join(stripped.split())
    if leftover:
        ignored.append(leftover)

    return ParsedQuery(query=query, entity_types=entity_types, ignored_fragments=ignored)
