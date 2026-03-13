"""Tests for detectors.py — core 7 change type detectors."""

import unittest

from cylint.diff.detectors import (
    classify_changes,
    detect_broadcast_hint_added,
    detect_broadcast_hint_removed,
    detect_filter_added,
    detect_filter_modified,
    detect_filter_removed,
    detect_projection_changed,
    detect_source_changed,
)
from cylint.diff.models import (
    FilterOp,
    OperationMatch,
    SelectOp,
    TrackedOperation,
)


def _match(base_op, pr_op, confidence="high"):
    return OperationMatch(base_op, pr_op, confidence=confidence, match_strategy="source_table")


class TestFilterRemoved(unittest.TestCase):
    def test_simple_removal(self):
        base = TrackedOperation(variable="orders", line=1,
                                filters=[FilterOp(line=2, expr_hash="aaa")])
        pr = TrackedOperation(variable="orders", line=1, filters=[])
        result = detect_filter_removed(_match(base, pr), "test.py")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].change_type, "filter_removed")

    def test_one_of_two_removed(self):
        base = TrackedOperation(variable="orders", line=1,
                                filters=[FilterOp(line=2, expr_hash="aaa"),
                                         FilterOp(line=3, expr_hash="bbb")])
        pr = TrackedOperation(variable="orders", line=1,
                              filters=[FilterOp(line=2, expr_hash="aaa")])
        result = detect_filter_removed(_match(base, pr), "test.py")
        self.assertEqual(len(result), 1)

    def test_no_removal(self):
        base = TrackedOperation(variable="orders", line=1,
                                filters=[FilterOp(line=2, expr_hash="aaa")])
        pr = TrackedOperation(variable="orders", line=1,
                              filters=[FilterOp(line=2, expr_hash="aaa")])
        result = detect_filter_removed(_match(base, pr), "test.py")
        self.assertEqual(len(result), 0)


class TestFilterModified(unittest.TestCase):
    def test_expression_changed(self):
        base = TrackedOperation(variable="orders", line=1,
                                filters=[FilterOp(line=2, expr_hash="aaa")])
        pr = TrackedOperation(variable="orders", line=1,
                              filters=[FilterOp(line=2, expr_hash="bbb")])
        result = detect_filter_modified(_match(base, pr), "test.py")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].change_type, "filter_modified")

    def test_same_expression_no_fire(self):
        base = TrackedOperation(variable="orders", line=1,
                                filters=[FilterOp(line=2, expr_hash="aaa")])
        pr = TrackedOperation(variable="orders", line=1,
                              filters=[FilterOp(line=5, expr_hash="aaa")])
        result = detect_filter_modified(_match(base, pr), "test.py")
        self.assertEqual(len(result), 0)

    def test_different_count_no_fire(self):
        """filter_modified only fires when filter counts match."""
        base = TrackedOperation(variable="orders", line=1,
                                filters=[FilterOp(line=2, expr_hash="aaa")])
        pr = TrackedOperation(variable="orders", line=1,
                              filters=[FilterOp(line=2, expr_hash="bbb"),
                                       FilterOp(line=3, expr_hash="ccc")])
        result = detect_filter_modified(_match(base, pr), "test.py")
        self.assertEqual(len(result), 0)


class TestFilterAdded(unittest.TestCase):
    def test_new_filter(self):
        base = TrackedOperation(variable="orders", line=1, filters=[])
        pr = TrackedOperation(variable="orders", line=1,
                              filters=[FilterOp(line=2, expr_hash="aaa")])
        result = detect_filter_added(_match(base, pr), "test.py")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].change_type, "filter_added")

    def test_existing_filter_not_added(self):
        base = TrackedOperation(variable="orders", line=1,
                                filters=[FilterOp(line=2, expr_hash="aaa")])
        pr = TrackedOperation(variable="orders", line=1,
                              filters=[FilterOp(line=2, expr_hash="aaa")])
        result = detect_filter_added(_match(base, pr), "test.py")
        self.assertEqual(len(result), 0)


class TestSourceChanged(unittest.TestCase):
    def test_table_swapped(self):
        base = TrackedOperation(variable="orders", source_table="orders_v2", line=1)
        pr = TrackedOperation(variable="orders", source_table="orders_raw", line=1)
        result = detect_source_changed(_match(base, pr), "test.py")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].change_type, "source_changed")
        self.assertEqual(result[0].metadata["oldTable"], "orders_v2")
        self.assertEqual(result[0].metadata["newTable"], "orders_raw")

    def test_same_table_no_fire(self):
        base = TrackedOperation(variable="orders", source_table="orders", line=1)
        pr = TrackedOperation(variable="orders", source_table="orders", line=1)
        result = detect_source_changed(_match(base, pr), "test.py")
        self.assertEqual(len(result), 0)

    def test_null_source_no_fire(self):
        base = TrackedOperation(variable="orders", source_table=None, line=1)
        pr = TrackedOperation(variable="orders", source_table="orders", line=1)
        result = detect_source_changed(_match(base, pr), "test.py")
        self.assertEqual(len(result), 0)


class TestBroadcastHintRemoved(unittest.TestCase):
    def test_broadcast_removed(self):
        base = TrackedOperation(variable="orders", line=1, broadcasts=[3])
        pr = TrackedOperation(variable="orders", line=1, broadcasts=[])
        result = detect_broadcast_hint_removed(_match(base, pr), "test.py")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].change_type, "broadcast_hint_removed")

    def test_no_change(self):
        base = TrackedOperation(variable="orders", line=1, broadcasts=[3])
        pr = TrackedOperation(variable="orders", line=1, broadcasts=[3])
        result = detect_broadcast_hint_removed(_match(base, pr), "test.py")
        self.assertEqual(len(result), 0)

    def test_broadcast_added_doesnt_fire(self):
        base = TrackedOperation(variable="orders", line=1, broadcasts=[])
        pr = TrackedOperation(variable="orders", line=1, broadcasts=[3])
        result = detect_broadcast_hint_removed(_match(base, pr), "test.py")
        self.assertEqual(len(result), 0)


class TestBroadcastHintAdded(unittest.TestCase):
    def test_broadcast_added(self):
        base = TrackedOperation(variable="orders", line=1, broadcasts=[])
        pr = TrackedOperation(variable="orders", line=1, broadcasts=[3])
        result = detect_broadcast_hint_added(_match(base, pr), "test.py")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].change_type, "broadcast_hint_added")

    def test_removed_doesnt_fire(self):
        base = TrackedOperation(variable="orders", line=1, broadcasts=[3])
        pr = TrackedOperation(variable="orders", line=1, broadcasts=[])
        result = detect_broadcast_hint_added(_match(base, pr), "test.py")
        self.assertEqual(len(result), 0)


class TestProjectionChanged(unittest.TestCase):
    def test_column_count_changed(self):
        base = TrackedOperation(variable="orders", line=1,
                                selects=[SelectOp(line=2, col_count=2)])
        pr = TrackedOperation(variable="orders", line=1,
                              selects=[SelectOp(line=2, col_count=5)])
        result = detect_projection_changed(_match(base, pr), "test.py")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].change_type, "projection_changed")
        self.assertEqual(result[0].metadata["oldColCount"], 2)
        self.assertEqual(result[0].metadata["newColCount"], 5)

    def test_same_columns_no_fire(self):
        base = TrackedOperation(variable="orders", line=1,
                                selects=[SelectOp(line=2, col_count=3)])
        pr = TrackedOperation(variable="orders", line=1,
                              selects=[SelectOp(line=2, col_count=3)])
        result = detect_projection_changed(_match(base, pr), "test.py")
        self.assertEqual(len(result), 0)

    def test_col_names_changed(self):
        base = TrackedOperation(variable="orders", line=1,
                                selects=[SelectOp(line=2, col_names=["id", "amt"])])
        pr = TrackedOperation(variable="orders", line=1,
                              selects=[SelectOp(line=2, col_names=["id", "amt", "status"])])
        result = detect_projection_changed(_match(base, pr), "test.py")
        self.assertEqual(len(result), 1)


class TestClassifyChanges(unittest.TestCase):
    def test_low_confidence_skipped(self):
        base = TrackedOperation(variable="orders", line=1,
                                filters=[FilterOp(line=2, expr_hash="aaa")])
        pr = TrackedOperation(variable="orders", line=1, filters=[])
        match = OperationMatch(base, pr, confidence="low", match_strategy="structural")
        result = classify_changes([match], [], "test.py")
        self.assertEqual(len(result), 0)

    def test_unmatched_base_produces_operation_removed(self):
        unmatched = [TrackedOperation(variable="legacy", source_table="old", line=10)]
        result = classify_changes([], unmatched, "test.py")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].change_type, "operation_removed")
        self.assertEqual(result[0].scope, "legacy (deleted)")

    def test_deduplication(self):
        base = TrackedOperation(variable="orders", line=1,
                                filters=[FilterOp(line=2, expr_hash="aaa"),
                                         FilterOp(line=2, expr_hash="bbb")])
        pr = TrackedOperation(variable="orders", line=1, filters=[])
        match = _match(base, pr)
        result = classify_changes([match], [], "test.py")
        # Two filters removed at different expr_hashes but same line — dedup to 1
        lines_and_types = [(c.line, c.change_type) for c in result]
        self.assertEqual(len(set(lines_and_types)), len(result))


if __name__ == "__main__":
    unittest.main()
