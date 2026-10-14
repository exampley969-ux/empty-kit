import time
import unittest

from vision_radar.api.nl_search import parse_query
from vision_radar.event_engine.models import EventType
from vision_radar.world_model.zone import Zone

DAY = 86400.0
NOON_TODAY = 1_700_000_000.0 - (1_700_000_000.0 % DAY) + 12 * 3600   # fixed, deterministic "now"


def day_start(ts):
    return ts - (ts % DAY)


def zones():
    return [Zone.from_rect("Zone A", 0, 0, 10, 10, zone_id="zoneA-id"),
            Zone.from_rect("Restricted-B", 0, 0, 10, 10, zone_id="zoneB-id")]


def parse(text, zs=None, now=NOON_TODAY):
    return parse_query(text, zs or zones(), now, day_start_of=day_start)


class TestEventTypeExtraction(unittest.TestCase):
    def test_zone_entry_phrase(self):
        self.assertEqual(parse("Show all zone-entry events today").query.event_type, EventType.ZONE_ENTRY)
        self.assertEqual(parse("zone entries in Zone A").query.event_type, EventType.ZONE_ENTRY)

    def test_each_event_type_phrase(self):
        cases = {
            "what appeared today": EventType.APPEARED,
            "anything disappear": EventType.DISAPPEARED,
            "movement today": EventType.MOVED,
            "any dwell events": EventType.DWELL,
            "state change events": EventType.STATE_CHANGED,
        }
        for text, expected in cases.items():
            self.assertEqual(parse(text).query.event_type, expected, text)

    def test_no_event_type_mentioned(self):
        self.assertIsNone(parse("what happened today").query.event_type)


class TestZoneExtraction(unittest.TestCase):
    def test_the_spec_example_what_happened_in_zone_a_today(self):
        parsed = parse("What happened in Zone A today?")
        self.assertEqual(parsed.query.zone_id, "zoneA-id")
        self.assertIsNotNone(parsed.query.start_time_from)

    def test_zone_name_with_punctuation_and_case_insensitive(self):
        self.assertEqual(parse("events in restricted-b").query.zone_id, "zoneB-id")
        self.assertEqual(parse("events IN ZONE A today").query.zone_id, "zoneA-id")

    def test_unknown_zone_name_resolves_to_none_not_a_guess(self):
        self.assertIsNone(parse("what happened in Zone Q today").query.zone_id)

    def test_no_zone_mentioned(self):
        self.assertIsNone(parse("show all events today").query.zone_id)


class TestTimeRangeExtraction(unittest.TestCase):
    def test_the_spec_example_between_two_times(self):
        # Section 18: "Show all zone-entry events between 2 PM and 4 PM."
        parsed = parse("Show all zone-entry events between 2 PM and 4 PM")
        base = day_start(NOON_TODAY)
        self.assertEqual(parsed.query.start_time_from, base + 14 * 3600)
        self.assertEqual(parsed.query.start_time_to, base + 16 * 3600)

    def test_between_times_sharing_one_meridiem(self):
        parsed = parse("between 2 and 4 pm today")
        base = day_start(NOON_TODAY)
        self.assertEqual(parsed.query.start_time_from, base + 14 * 3600)   # "2" inherits "pm"

    def test_between_times_with_minutes(self):
        parsed = parse("between 9:30 am and 11:15 am")
        base = day_start(NOON_TODAY)
        self.assertEqual(parsed.query.start_time_from, base + 9 * 3600 + 30 * 60)
        self.assertEqual(parsed.query.start_time_to, base + 11 * 3600 + 15 * 60)

    def test_today(self):
        base = day_start(NOON_TODAY)
        q = parse("what happened today").query
        self.assertEqual((q.start_time_from, q.start_time_to), (base, base + DAY))

    def test_yesterday(self):
        base = day_start(NOON_TODAY) - DAY
        q = parse("what happened yesterday").query
        self.assertEqual((q.start_time_from, q.start_time_to), (base, base + DAY))

    def test_last_n_minutes_hours_days(self):
        for text, seconds in (("last 10 minutes", 600), ("last 2 hours", 7200), ("last 3 days", 3 * DAY)):
            q = parse(text).query
            self.assertAlmostEqual(q.start_time_to - q.start_time_from, seconds)
            self.assertAlmostEqual(q.start_time_to, NOON_TODAY)

    def test_no_time_expression_leaves_range_open(self):
        q = parse("show all zone entries").query
        self.assertEqual((q.start_time_from, q.start_time_to), (None, None))


class TestEntityTypeExtraction(unittest.TestCase):
    def test_specific_entity_types(self):
        self.assertEqual(parse("any car events today").entity_types, ["car"])
        self.assertEqual(parse("person appeared today").entity_types, ["person"])

    def test_vehicle_expands_to_all_vehicle_types(self):
        self.assertEqual(sorted(parse("any vehicle events today").entity_types),
                         sorted(["car", "motorcycle", "bus", "truck"]))

    def test_multiple_entity_words_deduplicated_and_ordered(self):
        self.assertEqual(parse("cars and cars today").entity_types, ["car"])

    def test_no_entity_word_mentioned(self):
        self.assertEqual(parse("what happened today").entity_types, [])


class TestConfidenceExtraction(unittest.TestCase):
    def test_confidence_above_fraction(self):
        self.assertAlmostEqual(parse("zone entries with confidence above 0.8").query.min_confidence, 0.8)

    def test_confidence_as_percentage(self):
        self.assertAlmostEqual(parse("events with confidence over 80").query.min_confidence, 0.8)

    def test_no_confidence_mentioned(self):
        self.assertIsNone(parse("what happened today").query.min_confidence)


class TestIgnoredFragmentsAreReportedNotHallucinated(unittest.TestCase):
    def test_fully_understood_query_has_nothing_ignored(self):
        parsed = parse("What happened in Zone A today?")
        self.assertEqual(parsed.ignored_fragments, [])

    def test_unrecognised_words_are_reported_verbatim_not_dropped_silently(self):
        parsed = parse("did anybody loiter suspiciously near the gate today")
        self.assertTrue(parsed.ignored_fragments)
        self.assertIn("loiter", parsed.ignored_fragments[0])
        self.assertIn("suspiciously", parsed.ignored_fragments[0])

    def test_empty_query_is_handled(self):
        parsed = parse("")
        self.assertEqual(parsed.query.event_type, None)
        self.assertEqual(parsed.ignored_fragments, [])

    def test_none_query_is_handled(self):
        parsed = parse_query(None, zones(), NOON_TODAY, day_start_of=day_start)
        self.assertEqual(parsed.entity_types, [])


class TestCombinedRealisticQueries(unittest.TestCase):
    def test_person_zone_entries_in_a_named_zone_today_with_confidence(self):
        parsed = parse("Show person zone entries in Zone A today with confidence above 0.7")
        q = parsed.query
        self.assertEqual(q.event_type, EventType.ZONE_ENTRY)
        self.assertEqual(q.zone_id, "zoneA-id")
        self.assertEqual(parsed.entity_types, ["person"])
        self.assertAlmostEqual(q.min_confidence, 0.7)
        self.assertIsNotNone(q.start_time_from)


if __name__ == "__main__":
    unittest.main()


class TestIgnoredFragmentsMatchActualZoneMatching(unittest.TestCase):
    """The leftover-text calculation must use the same zone-matching logic
    as _extract_zone, or a zone referenced without the word 'zone' would be
    wrongly reported as not understood."""

    def test_zone_referenced_without_the_word_zone_is_not_reported_as_ignored(self):
        parsed = parse("what happened in restricted-b today")
        self.assertEqual(parsed.query.zone_id, "zoneB-id")
        self.assertEqual(parsed.ignored_fragments, [])


class TestNoiseWordsDontLeakIntoIgnored(unittest.TestCase):
    def test_confidence_query_has_nothing_ignored(self):
        parsed = parse("show zone entries with confidence above 0.5")
        self.assertEqual(parsed.ignored_fragments, [])
