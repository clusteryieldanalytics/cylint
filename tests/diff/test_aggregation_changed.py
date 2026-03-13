"""Tests for aggregation_changed detector."""

import unittest

from cylint.diff.classifier import extract_operations
from cylint.diff.detectors import detect_aggregation_changed
from cylint.diff.models import GroupByOp, OperationMatch, TrackedOperation


def _match(base_op, pr_op, confidence="high"):
    return OperationMatch(base_op, pr_op, confidence=confidence, match_strategy="source_table")


class TestAggregationChanged(unittest.TestCase):
    """Unit tests using constructed TrackedOperations."""

    def test_key_changed(self):
        base = TrackedOperation(variable="summary", line=2,
                                groupbys=[GroupByOp(line=2, key_expr_hash="aaa")])
        pr = TrackedOperation(variable="summary", line=2,
                              groupbys=[GroupByOp(line=2, key_expr_hash="bbb")])
        results = detect_aggregation_changed(_match(base, pr), "test.py")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].change_type, "aggregation_changed")

    def test_same_key_no_fire(self):
        base = TrackedOperation(variable="summary", line=2,
                                groupbys=[GroupByOp(line=2, key_expr_hash="aaa")])
        pr = TrackedOperation(variable="summary", line=2,
                              groupbys=[GroupByOp(line=2, key_expr_hash="aaa")])
        results = detect_aggregation_changed(_match(base, pr), "test.py")
        self.assertEqual(len(results), 0)

    def test_groupby_added_no_fire(self):
        base = TrackedOperation(variable="orders", line=1, groupbys=[])
        pr = TrackedOperation(variable="orders", line=1,
                              groupbys=[GroupByOp(line=2, key_expr_hash="aaa")])
        results = detect_aggregation_changed(_match(base, pr), "test.py")
        self.assertEqual(len(results), 0)

    def test_groupby_removed_no_fire(self):
        base = TrackedOperation(variable="orders", line=1,
                                groupbys=[GroupByOp(line=2, key_expr_hash="aaa")])
        pr = TrackedOperation(variable="orders", line=1, groupbys=[])
        results = detect_aggregation_changed(_match(base, pr), "test.py")
        self.assertEqual(len(results), 0)


class TestAggregationChangedE2E(unittest.TestCase):
    """End-to-end tests using extract_operations."""

    def test_groupby_key_changed(self):
        """groupBy key changed — recorded on the root DataFrame."""
        base_source = '''\
orders = spark.table("orders")
summary = orders.groupBy("date").count()
'''
        pr_source = '''\
orders = spark.table("orders")
summary = orders.groupBy("region").count()
'''
        base_ops = extract_operations(base_source)
        pr_ops = extract_operations(pr_source)
        base_orders = [op for op in base_ops if op.variable == "orders"][0]
        pr_orders = [op for op in pr_ops if op.variable == "orders"][0]
        results = detect_aggregation_changed(_match(base_orders, pr_orders), "test.py")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].change_type, "aggregation_changed")

    def test_groupby_key_additional_column(self):
        base_source = '''\
orders = spark.table("orders")
summary = orders.groupBy("date").count()
'''
        pr_source = '''\
orders = spark.table("orders")
summary = orders.groupBy("date", "status").count()
'''
        base_ops = extract_operations(base_source)
        pr_ops = extract_operations(pr_source)
        base_orders = [op for op in base_ops if op.variable == "orders"][0]
        pr_orders = [op for op in pr_ops if op.variable == "orders"][0]
        results = detect_aggregation_changed(_match(base_orders, pr_orders), "test.py")
        self.assertEqual(len(results), 1)

    def test_same_groupby_no_fire(self):
        source = '''\
orders = spark.table("orders")
summary = orders.groupBy("date").count()
'''
        base_ops = extract_operations(source)
        pr_ops = extract_operations(source)
        base_orders = [op for op in base_ops if op.variable == "orders"][0]
        pr_orders = [op for op in pr_ops if op.variable == "orders"][0]
        results = detect_aggregation_changed(_match(base_orders, pr_orders), "test.py")
        self.assertEqual(len(results), 0)

    def test_keys_reordered_no_fire(self):
        """groupBy("date", "region") vs groupBy("region", "date") — same keys."""
        base_source = '''\
orders = spark.table("orders")
summary = orders.groupBy("date", "region").count()
'''
        pr_source = '''\
orders = spark.table("orders")
summary = orders.groupBy("region", "date").count()
'''
        base_ops = extract_operations(base_source)
        pr_ops = extract_operations(pr_source)
        base_orders = [op for op in base_ops if op.variable == "orders"][0]
        pr_orders = [op for op in pr_ops if op.variable == "orders"][0]
        results = detect_aggregation_changed(_match(base_orders, pr_orders), "test.py")
        self.assertEqual(len(results), 0)

    def test_agg_function_changed_keys_same_no_fire(self):
        """Changing sum to avg with same keys should not fire."""
        base_source = '''\
orders = spark.table("orders")
summary = orders.groupBy("date").agg(F.sum("amount"))
'''
        pr_source = '''\
orders = spark.table("orders")
summary = orders.groupBy("date").agg(F.avg("amount"))
'''
        base_ops = extract_operations(base_source)
        pr_ops = extract_operations(pr_source)
        base_orders = [op for op in base_ops if op.variable == "orders"][0]
        pr_orders = [op for op in pr_ops if op.variable == "orders"][0]
        results = detect_aggregation_changed(_match(base_orders, pr_orders), "test.py")
        self.assertEqual(len(results), 0)


if __name__ == "__main__":
    unittest.main()
