"""Tests for diff/models.py — serialization and data model basics."""

import unittest

from cylint.diff.models import (
    ChangeClassification,
    ChangedFile,
    FilterOp,
    OperationMatch,
    SelectOp,
    TrackedOperation,
)


class TestTrackedOperationToDict(unittest.TestCase):
    def test_empty_op(self):
        op = TrackedOperation(variable="df", line=1)
        d = op.to_dict()
        self.assertEqual(d["variable"], "df")
        self.assertEqual(d["line"], 1)
        self.assertIsNone(d["sourceTable"])
        self.assertEqual(d["filters"], [])

    def test_with_filters_and_selects(self):
        op = TrackedOperation(
            variable="orders",
            source_table="orders",
            line=5,
            filters=[FilterOp(line=6, expr_hash="abc123")],
            selects=[SelectOp(line=7, col_count=3, col_names=["id", "name", "amt"])],
        )
        d = op.to_dict()
        self.assertEqual(len(d["filters"]), 1)
        self.assertEqual(d["filters"][0]["exprHash"], "abc123")
        self.assertEqual(d["selects"][0]["colCount"], 3)


class TestChangeClassificationToDict(unittest.TestCase):
    def test_serialization(self):
        c = ChangeClassification(
            file="test.py",
            line=10,
            change_type="filter_removed",
            confidence="high",
            source_table="orders",
            scope=".filter()",
            metadata={},
        )
        d = c.to_dict()
        self.assertEqual(d["changeType"], "filter_removed")
        self.assertEqual(d["confidence"], "high")
        self.assertEqual(d["sourceTable"], "orders")

    def test_with_metadata(self):
        c = ChangeClassification(
            file="test.py", line=5, change_type="source_changed",
            confidence="high", source_table="new_table",
            metadata={"oldTable": "old_table", "newTable": "new_table"},
        )
        d = c.to_dict()
        self.assertEqual(d["metadata"]["oldTable"], "old_table")


class TestChangedFile(unittest.TestCase):
    def test_modified(self):
        cf = ChangedFile(status="M", path="src/foo.py")
        self.assertEqual(cf.status, "M")
        self.assertIsNone(cf.old_path)

    def test_rename(self):
        cf = ChangedFile(status="R", path="src/new.py", old_path="src/old.py")
        self.assertEqual(cf.old_path, "src/old.py")


if __name__ == "__main__":
    unittest.main()
