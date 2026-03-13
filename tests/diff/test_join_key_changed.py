"""Tests for join_key_changed detector."""

import unittest

from cylint.diff.classifier import extract_operations
from cylint.diff.detectors import detect_join_key_changed
from cylint.diff.models import JoinOp, OperationMatch, TrackedOperation


def _match(base_op, pr_op, confidence="high"):
    return OperationMatch(base_op, pr_op, confidence=confidence, match_strategy="source_table")


class TestJoinKeyChanged(unittest.TestCase):
    """Unit tests using constructed TrackedOperations."""

    def test_key_changed(self):
        base = TrackedOperation(variable="joined", line=3,
                                joins=[JoinOp(line=3, key_expr_hash="aaa")])
        pr = TrackedOperation(variable="joined", line=3,
                              joins=[JoinOp(line=3, key_expr_hash="bbb")])
        results = detect_join_key_changed(_match(base, pr), "test.py")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].change_type, "join_key_changed")

    def test_same_key_no_fire(self):
        base = TrackedOperation(variable="joined", line=3,
                                joins=[JoinOp(line=3, key_expr_hash="aaa")])
        pr = TrackedOperation(variable="joined", line=3,
                              joins=[JoinOp(line=3, key_expr_hash="aaa")])
        results = detect_join_key_changed(_match(base, pr), "test.py")
        self.assertEqual(len(results), 0)

    def test_new_join_added_no_fire(self):
        """Different join counts should not fire — it's structural, not key-changed."""
        base = TrackedOperation(variable="orders", line=1, joins=[])
        pr = TrackedOperation(variable="orders", line=1,
                              joins=[JoinOp(line=3, key_expr_hash="aaa")])
        results = detect_join_key_changed(_match(base, pr), "test.py")
        self.assertEqual(len(results), 0)

    def test_join_removed_no_fire(self):
        base = TrackedOperation(variable="orders", line=1,
                                joins=[JoinOp(line=3, key_expr_hash="aaa")])
        pr = TrackedOperation(variable="orders", line=1, joins=[])
        results = detect_join_key_changed(_match(base, pr), "test.py")
        self.assertEqual(len(results), 0)


class TestJoinKeyChangedE2E(unittest.TestCase):
    """End-to-end tests using extract_operations."""

    def test_join_key_column_changed(self):
        base_source = '''\
orders = spark.table("orders")
customers = spark.table("customers")
joined = orders.join(customers, "customer_id")
'''
        pr_source = '''\
orders = spark.table("orders")
customers = spark.table("customers")
joined = orders.join(customers, "account_id")
'''
        base_ops = extract_operations(base_source)
        pr_ops = extract_operations(pr_source)
        base_joined = [op for op in base_ops if op.variable == "joined"][0]
        pr_joined = [op for op in pr_ops if op.variable == "joined"][0]
        results = detect_join_key_changed(_match(base_joined, pr_joined), "test.py")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].change_type, "join_key_changed")

    def test_same_join_key_no_fire(self):
        source = '''\
orders = spark.table("orders")
customers = spark.table("customers")
joined = orders.join(customers, "customer_id")
'''
        base_ops = extract_operations(source)
        pr_ops = extract_operations(source)
        base_joined = [op for op in base_ops if op.variable == "joined"][0]
        pr_joined = [op for op in pr_ops if op.variable == "joined"][0]
        results = detect_join_key_changed(_match(base_joined, pr_joined), "test.py")
        self.assertEqual(len(results), 0)


if __name__ == "__main__":
    unittest.main()
