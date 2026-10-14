import unittest

from vision_radar.event_engine.models import Event, EventType
from vision_radar.rules_engine.engine import RuleEngine
from vision_radar.rules_engine.models import ActionType, FieldCondition, Rule, RuleTrigger


def make_event(event_type=EventType.ZONE_ENTRY, entity_type="person", zone_id="z1",
                confidence=0.9, start_time=0.0, end_time=None, evidence=None):
    e = Event.new(event_type=event_type, entity_ids=["e1"], entity_type=entity_type,
                  start_time=start_time, confidence=confidence, zone_id=zone_id,
                  evidence=evidence or {})
    if end_time is not None:
        e.end_time = end_time
    return e


class TestRuleTrigger(unittest.TestCase):
    def test_matches_on_event_type_only(self):
        trigger = RuleTrigger(event_type=EventType.ZONE_ENTRY)
        self.assertTrue(trigger.matches(make_event(event_type=EventType.ZONE_ENTRY)))
        self.assertFalse(trigger.matches(make_event(event_type=EventType.APPEARED)))

    def test_entity_type_filter(self):
        trigger = RuleTrigger(event_type=EventType.ZONE_ENTRY, entity_type="person")
        self.assertTrue(trigger.matches(make_event(entity_type="person")))
        self.assertFalse(trigger.matches(make_event(entity_type="car")))

    def test_zone_filter(self):
        trigger = RuleTrigger(event_type=EventType.ZONE_ENTRY, zone_id="restricted-A")
        self.assertTrue(trigger.matches(make_event(zone_id="restricted-A")))
        self.assertFalse(trigger.matches(make_event(zone_id="other-zone")))

    def test_no_entity_or_zone_filter_matches_anything(self):
        trigger = RuleTrigger(event_type=EventType.ZONE_ENTRY)
        self.assertTrue(trigger.matches(make_event(entity_type="whatever", zone_id="whatever-zone")))


class TestFieldCondition(unittest.TestCase):
    def test_confidence_greater_than(self):
        cond = FieldCondition("confidence", ">", 0.8)
        self.assertTrue(cond.evaluate(make_event(confidence=0.9)))
        self.assertFalse(cond.evaluate(make_event(confidence=0.5)))

    def test_duration_seconds_condition_on_resolved_event(self):
        cond = FieldCondition("duration_seconds", ">", 10.0)
        event = make_event(start_time=0.0, end_time=18.0)
        self.assertTrue(cond.evaluate(event))

    def test_duration_seconds_condition_false_on_still_active_event(self):
        # duration_seconds() is None while the event has no end_time yet -
        # a threshold check against None must be False, not raise.
        cond = FieldCondition("duration_seconds", ">", 5.0)
        event = make_event(start_time=0.0)   # end_time stays None
        self.assertFalse(cond.evaluate(event))

    def test_evidence_field_lookup(self):
        cond = FieldCondition("evidence.dwell_seconds", ">=", 4.0)
        event = make_event(evidence={"dwell_seconds": 6.0})
        self.assertTrue(cond.evaluate(event))

    def test_unsupported_operator_raises(self):
        cond = FieldCondition("confidence", "~=", 0.5)
        with self.assertRaises(ValueError):
            cond.evaluate(make_event())


class TestRuleEngineEvaluate(unittest.TestCase):
    def test_matching_rule_produces_alert(self):
        engine = RuleEngine()
        rule = Rule.new(
            name="Person enters restricted zone",
            when=RuleTrigger(event_type=EventType.ZONE_ENTRY, entity_type="person", zone_id="restricted-A"),
            if_conditions=[FieldCondition("confidence", ">", 0.8)],
        )
        engine.add_rule(rule)

        event = make_event(entity_type="person", zone_id="restricted-A", confidence=0.94)
        alerts = engine.evaluate([event])

        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].rule_id, rule.rule_id)
        self.assertEqual(alerts[0].event_id, event.event_id)
        self.assertIn("restricted-A", alerts[0].message)

    def test_rule_does_not_fire_when_if_condition_fails(self):
        engine = RuleEngine()
        rule = Rule.new(
            name="High confidence only",
            when=RuleTrigger(event_type=EventType.ZONE_ENTRY),
            if_conditions=[FieldCondition("confidence", ">", 0.9)],
        )
        engine.add_rule(rule)

        event = make_event(confidence=0.5)
        alerts = engine.evaluate([event])
        self.assertEqual(alerts, [])

    def test_rule_does_not_fire_for_wrong_event_type(self):
        engine = RuleEngine()
        rule = Rule.new(name="Zone rule", when=RuleTrigger(event_type=EventType.ZONE_ENTRY))
        engine.add_rule(rule)

        event = make_event(event_type=EventType.APPEARED)
        alerts = engine.evaluate([event])
        self.assertEqual(alerts, [])

    def test_no_if_conditions_means_always_true(self):
        engine = RuleEngine()
        rule = Rule.new(name="Any zone entry", when=RuleTrigger(event_type=EventType.ZONE_ENTRY))
        engine.add_rule(rule)

        alerts = engine.evaluate([make_event()])
        self.assertEqual(len(alerts), 1)

    def test_multiple_rules_can_each_fire_on_the_same_event(self):
        engine = RuleEngine()
        engine.add_rule(Rule.new(name="Rule A", when=RuleTrigger(event_type=EventType.ZONE_ENTRY)))
        engine.add_rule(Rule.new(name="Rule B", when=RuleTrigger(event_type=EventType.ZONE_ENTRY)))

        alerts = engine.evaluate([make_event()])
        self.assertEqual(len(alerts), 2)

    def test_vehicle_dwell_rule_example_from_spec(self):
        # Mirrors Section 22's own second example: WHEN Vehicle enters Zone
        # B / IF dwell > defined duration / THEN generate event(alert).
        engine = RuleEngine()
        rule = Rule.new(
            name="Vehicle dwell alert",
            when=RuleTrigger(event_type=EventType.DWELL, entity_type="car"),
            if_conditions=[FieldCondition("evidence.dwell_seconds", ">", 30.0)],
        )
        engine.add_rule(rule)

        short_dwell = make_event(event_type=EventType.DWELL, entity_type="car",
                                  evidence={"dwell_seconds": 10.0})
        long_dwell = make_event(event_type=EventType.DWELL, entity_type="car",
                                 evidence={"dwell_seconds": 45.0})

        self.assertEqual(engine.evaluate([short_dwell]), [])
        self.assertEqual(len(engine.evaluate([long_dwell])), 1)

    def test_remove_and_list_rules(self):
        engine = RuleEngine()
        rule = Rule.new(name="R", when=RuleTrigger(event_type=EventType.ZONE_ENTRY))
        engine.add_rule(rule)
        self.assertEqual(len(engine.list_rules()), 1)
        engine.remove_rule(rule.rule_id)
        self.assertEqual(len(engine.list_rules()), 0)
        with self.assertRaises(KeyError):
            engine.get_rule(rule.rule_id)


if __name__ == "__main__":
    unittest.main()
