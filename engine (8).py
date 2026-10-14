"""
Vision Radar - Rule Engine
Evaluates the set of defined Rules against incoming Events, producing
Alerts only from rules that actually match (Section 21: "Alerts are
generated only from defined events").

Deduplication (added in Phase 9's review): the Event Engine legitimately
reports the same event several times as it moves through its lifecycle
(ACTIVE when a zone is entered, RESOLVED when it is left, ARCHIVED later).
Each report is passed here, so without dedup one zone visit would raise the
same rule's alert two or three times - exactly the duplicate-notification
problem Section 13 exists to prevent. A rule now fires at most once per
event, the first time its WHEN + IF conditions hold (dedup memory for an
event is released once its final ARCHIVED report has been evaluated). That still lets
lifecycle-dependent conditions work: a rule gated on `duration_seconds`
is false while the event is ACTIVE and fires once when it becomes RESOLVED.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

from vision_radar.event_engine.models import Event, EventState

from .models import Alert, Rule


class RuleEngine:
    def __init__(self):
        self._rules: Dict[str, Rule] = {}
        self._fired: Set[Tuple[str, str]] = set()   # (rule_id, event_id)

    def add_rule(self, rule: Rule) -> None:
        self._rules[rule.rule_id] = rule

    def remove_rule(self, rule_id: str) -> None:
        del self._rules[rule_id]
        self._fired = {k for k in self._fired if k[0] != rule_id}

    def get_rule(self, rule_id: str) -> Rule:
        return self._rules[rule_id]

    def list_rules(self) -> List[Rule]:
        return list(self._rules.values())

    def evaluate(self, events: List[Event]) -> List[Alert]:
        """Check every defined rule against every event this call, in the
        order rules were added. Multiple different rules can fire on the same
        event; each rule fires at most once per event."""
        alerts: List[Alert] = []
        for event in events:
            for rule in self._rules.values():
                key = (rule.rule_id, event.event_id)
                if key in self._fired:
                    continue
                if rule.matches(event):
                    self._fired.add(key)
                    alerts.append(Alert.from_match(rule, event))
            if event.state == EventState.ARCHIVED:
                # Final lifecycle report - evaluated like any other (an event
                # created and archived within one update must still be able to
                # alert), then forgotten so the fired-set cannot grow without
                # bound in a long-running service.
                self._fired = {k for k in self._fired if k[1] != event.event_id}
        return alerts
