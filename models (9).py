"""
Vision Radar - Rule Engine
Generic WHEN/IF/THEN rule structure and the Alert it produces.

Spec reference: PDF Section 22 (Rule engine - "WHEN [EVENT] / IF
[CONDITION] / THEN [ACTION]"), Section 21 (Alerts - generated only from
defined events, per a rule).

Scope note: THEN currently supports exactly one action -
`GENERATE_ALERT`. The spec's own Section 21 example also lists "Create
Zone Entry Event" and "Record event" as THEN-actions, but those already
happen elsewhere in this architecture (event creation is Phase 7; DB
persistence is Phase 9) - a rule re-triggering them here would duplicate
work across layers rather than compose with them. Only the action that is
genuinely new at this layer (deciding whether an already-created event
should become a visible alert) is implemented. No other action types are
invented ahead of a real requirement for them.
"""

from __future__ import annotations

import operator
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional

from vision_radar.event_engine.models import Event, EventType

_OPERATORS = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}


class ActionType(str, Enum):
    GENERATE_ALERT = "GENERATE_ALERT"


def _resolve_field(event: Event, field_path: str) -> Optional[Any]:
    """
    Resolves exactly the field patterns the spec's own examples need
    (Section 22: "confidence > threshold", "dwell > defined duration") -
    a deliberately small, explicit resolver, not a general expression
    language, per the "no unnecessary features" rule.

    Supported patterns:
      "confidence"              -> event.confidence
      "duration_seconds"        -> event.duration_seconds() (None if ongoing)
      "evidence.<key>"          -> event.evidence.get(<key>)
    """
    if field_path == "confidence":
        return event.confidence
    if field_path == "duration_seconds":
        return event.duration_seconds()
    if field_path.startswith("evidence."):
        key = field_path[len("evidence."):]
        return event.evidence.get(key)
    return None


@dataclass
class FieldCondition:
    """One IF clause term, e.g. FieldCondition("confidence", ">", 0.8)."""
    field: str
    operator: str
    value: float

    def validate(self) -> None:
        """Reject an unusable condition at definition time."""
        if self.operator not in _OPERATORS:
            raise ValueError(f"Unsupported operator '{self.operator}' "
                             f"(supported: {', '.join(_OPERATORS)})")
        if not (self.field in ("confidence", "duration_seconds")
                or (self.field.startswith("evidence.") and len(self.field) > len("evidence."))):
            raise ValueError(f"Unsupported field '{self.field}' (supported: confidence, "
                             f"duration_seconds, evidence.<key>)")
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise ValueError("Condition value must be a number")

    def evaluate(self, event: Event) -> bool:
        if self.operator not in _OPERATORS:
            raise ValueError(f"Unsupported operator: {self.operator}")
        actual = _resolve_field(event, self.field)
        if actual is None:
            return False   # e.g. duration_seconds() on a still-ACTIVE event
        return _OPERATORS[self.operator](actual, self.value)


@dataclass
class RuleTrigger:
    """The WHEN clause: which Events this rule even looks at."""
    event_type: EventType
    entity_type: Optional[str] = None   # e.g. "person" - None matches any
    zone_id: Optional[str] = None        # None matches any zone (or no zone)

    def matches(self, event: Event) -> bool:
        if event.event_type != self.event_type:
            return False
        if self.entity_type is not None and event.entity_type != self.entity_type:
            return False
        if self.zone_id is not None and event.zone_id != self.zone_id:
            return False
        return True


@dataclass
class Rule:
    rule_id: str
    name: str
    when: RuleTrigger
    if_conditions: List[FieldCondition] = field(default_factory=list)   # AND'ed; empty = always true
    action: ActionType = ActionType.GENERATE_ALERT

    @classmethod
    def new(cls, name: str, when: RuleTrigger, if_conditions: List[FieldCondition] = None,
            action: ActionType = ActionType.GENERATE_ALERT) -> "Rule":
        return cls(rule_id=str(uuid.uuid4()), name=name, when=when,
                   if_conditions=if_conditions or [], action=action)

    def matches(self, event: Event) -> bool:
        return self.when.matches(event) and all(c.evaluate(event) for c in self.if_conditions)

    # --- (de)serialisation, used by the data layer and the API -----------
    def to_definition(self) -> dict:
        return {
            "when": {
                "event_type": self.when.event_type.value,
                "entity_type": self.when.entity_type,
                "zone_id": self.when.zone_id,
            },
            "if": [{"field": c.field, "operator": c.operator, "value": c.value}
                   for c in self.if_conditions],
            "action": self.action.value,
        }

    @classmethod
    def from_definition(cls, name: str, definition: dict, rule_id: str = None) -> "Rule":
        """Build a Rule from plain data, validating everything up front so a
        malformed rule is rejected when it is created, not when it first
        silently fails to fire."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Rule name must be a non-empty string")
        if not isinstance(definition, dict):
            raise ValueError("Rule definition must be an object")
        when = definition.get("when")
        if not isinstance(when, dict) or "event_type" not in when:
            raise ValueError("Rule definition needs a 'when' object with an 'event_type'")
        try:
            event_type = EventType(when["event_type"])
        except ValueError:
            valid = ", ".join(t.value for t in EventType)
            raise ValueError(f"Unknown event_type '{when['event_type']}' (valid: {valid})")

        conditions = []
        for raw in definition.get("if", []) or []:
            if not isinstance(raw, dict) or not {"field", "operator", "value"} <= raw.keys():
                raise ValueError("Each 'if' condition needs 'field', 'operator' and 'value'")
            cond = FieldCondition(raw["field"], raw["operator"], raw["value"])
            cond.validate()
            conditions.append(cond)

        try:
            action = ActionType(definition.get("action", ActionType.GENERATE_ALERT.value))
        except ValueError:
            raise ValueError(f"Unsupported action '{definition.get('action')}'")

        return cls(
            rule_id=rule_id or str(uuid.uuid4()), name=name.strip(),
            when=RuleTrigger(event_type=event_type,
                             entity_type=when.get("entity_type"), zone_id=when.get("zone_id")),
            if_conditions=conditions, action=action,
        )


@dataclass
class Alert:
    """Produced when a Rule matches an Event (Section 21)."""
    alert_id: str
    rule_id: str
    rule_name: str
    event_id: str
    event_type: EventType
    entity_ids: List[str]
    zone_id: Optional[str]
    confidence: float
    message: str
    created_at: float = field(default_factory=time.time)

    @classmethod
    def from_match(cls, rule: Rule, event: Event) -> "Alert":
        zone_part = f" in zone {event.zone_id}" if event.zone_id else ""
        message = (
            f"Rule '{rule.name}' triggered: {event.entity_type} "
            f"{event.event_type.value.replace('_', ' ').lower()}{zone_part} "
            f"(confidence {event.confidence:.2f})"
        )
        return cls(
            alert_id=str(uuid.uuid4()),
            rule_id=rule.rule_id,
            rule_name=rule.name,
            event_id=event.event_id,
            event_type=event.event_type,
            entity_ids=list(event.entity_ids),
            zone_id=event.zone_id,
            confidence=event.confidence,
            message=message,
        )
