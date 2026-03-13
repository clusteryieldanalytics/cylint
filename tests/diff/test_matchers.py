"""Tests for matchers.py — operation matching tiers."""

import unittest

from cylint.diff.matchers import (
    GENERIC_NAMES,
    match_by_source_table,
    match_by_variable_scope,
    match_operations,
)
from cylint.diff.models import TrackedOperation


class TestMatchBySourceTable(unittest.TestCase):
    def test_same_table_matches_high(self):
        base = [TrackedOperation(variable="df", source_table="orders", line=1)]
        pr = [TrackedOperation(variable="renamed_df", source_table="orders", line=1)]
        matches = match_by_source_table(base, pr)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].confidence, "high")
        self.assertEqual(matches[0].match_strategy, "source_table")

    def test_different_table_no_match(self):
        base = [TrackedOperation(variable="df", source_table="orders", line=1)]
        pr = [TrackedOperation(variable="df", source_table="customers", line=1)]
        matches = match_by_source_table(base, pr)
        self.assertEqual(len(matches), 0)

    def test_none_source_table_skipped(self):
        base = [TrackedOperation(variable="df", source_table=None, line=1)]
        pr = [TrackedOperation(variable="df", source_table=None, line=1)]
        matches = match_by_source_table(base, pr)
        self.assertEqual(len(matches), 0)

    def test_greedy_matching(self):
        """Each PR op can only be matched once."""
        base = [
            TrackedOperation(variable="a", source_table="orders", line=1),
            TrackedOperation(variable="b", source_table="orders", line=5),
        ]
        pr = [TrackedOperation(variable="c", source_table="orders", line=1)]
        matches = match_by_source_table(base, pr)
        self.assertEqual(len(matches), 1)


class TestMatchByVariableScope(unittest.TestCase):
    def test_same_name_matches_medium(self):
        base = [TrackedOperation(variable="order_data", line=1)]
        pr = [TrackedOperation(variable="order_data", line=1)]
        matched_b: set[int] = set()
        matched_p: set[int] = set()
        matches = match_by_variable_scope(base, pr, matched_b, matched_p)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].confidence, "medium")

    def test_generic_name_excluded(self):
        for name in ["df", "data", "result", "tmp"]:
            base = [TrackedOperation(variable=name, line=1)]
            pr = [TrackedOperation(variable=name, line=1)]
            matches = match_by_variable_scope(base, pr, set(), set())
            self.assertEqual(len(matches), 0, f"Generic name '{name}' should not match")

    def test_different_name_no_match(self):
        base = [TrackedOperation(variable="orders", line=1)]
        pr = [TrackedOperation(variable="customers", line=1)]
        matches = match_by_variable_scope(base, pr, set(), set())
        self.assertEqual(len(matches), 0)


class TestMatchOperations(unittest.TestCase):
    def test_tier1_takes_priority(self):
        """Source table match (tier 1) beats variable name match (tier 2)."""
        base = [TrackedOperation(variable="orders", source_table="orders", line=1)]
        pr = [TrackedOperation(variable="orders", source_table="orders", line=1)]
        matches, unmatched_b, unmatched_p = match_operations(base, pr)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].confidence, "high")

    def test_unmatched_base_returned(self):
        base = [TrackedOperation(variable="legacy", source_table="old_table", line=1)]
        pr: list[TrackedOperation] = []
        matches, unmatched_b, unmatched_p = match_operations(base, pr)
        self.assertEqual(len(matches), 0)
        self.assertEqual(len(unmatched_b), 1)

    def test_unmatched_pr_returned(self):
        base: list[TrackedOperation] = []
        pr = [TrackedOperation(variable="new_thing", line=1)]
        matches, unmatched_b, unmatched_p = match_operations(base, pr)
        self.assertEqual(len(unmatched_p), 1)

    def test_mixed_tiers(self):
        base = [
            TrackedOperation(variable="a", source_table="orders", line=1),
            TrackedOperation(variable="custom_name", line=5),
        ]
        pr = [
            TrackedOperation(variable="b", source_table="orders", line=1),
            TrackedOperation(variable="custom_name", line=5),
        ]
        matches, _, _ = match_operations(base, pr)
        self.assertEqual(len(matches), 2)
        strategies = {m.match_strategy for m in matches}
        self.assertIn("source_table", strategies)
        self.assertIn("variable_scope", strategies)


if __name__ == "__main__":
    unittest.main()
