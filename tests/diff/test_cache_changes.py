"""Tests for cache_added and cache_removed detectors."""

import unittest

from cylint.diff.classifier import extract_operations
from cylint.diff.detectors import detect_cache_added, detect_cache_removed
from cylint.diff.models import OperationMatch, TrackedOperation


def _match(base_op, pr_op, confidence="high"):
    return OperationMatch(base_op, pr_op, confidence=confidence, match_strategy="source_table")


class TestCacheAdded(unittest.TestCase):

    def test_cache_added(self):
        """Cache added where none existed."""
        base = TrackedOperation(variable="orders", line=1, caches=[])
        pr = TrackedOperation(variable="orders", line=1, caches=[2])
        results = detect_cache_added(_match(base, pr), "test.py")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].change_type, "cache_added")
        self.assertEqual(results[0].line, 2)

    def test_cache_present_both_no_fire(self):
        """Cache in both branches — no change."""
        base = TrackedOperation(variable="orders", line=1, caches=[2])
        pr = TrackedOperation(variable="orders", line=1, caches=[2])
        results = detect_cache_added(_match(base, pr), "test.py")
        self.assertEqual(len(results), 0)

    def test_no_cache_either_no_fire(self):
        """No cache in either — no change."""
        base = TrackedOperation(variable="orders", line=1, caches=[])
        pr = TrackedOperation(variable="orders", line=1, caches=[])
        results = detect_cache_added(_match(base, pr), "test.py")
        self.assertEqual(len(results), 0)

    def test_cache_removed_no_fire_for_added(self):
        """Cache removed should not fire cache_added."""
        base = TrackedOperation(variable="orders", line=1, caches=[2])
        pr = TrackedOperation(variable="orders", line=1, caches=[])
        results = detect_cache_added(_match(base, pr), "test.py")
        self.assertEqual(len(results), 0)


class TestCacheRemoved(unittest.TestCase):

    def test_cache_removed(self):
        """Cache removed from base."""
        base = TrackedOperation(variable="orders", line=1, source_table="orders", caches=[2])
        pr = TrackedOperation(variable="orders", line=1, caches=[])
        results = detect_cache_removed(_match(base, pr), "test.py")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].change_type, "cache_removed")
        self.assertEqual(results[0].line, 2)

    def test_cache_present_both_no_fire(self):
        """Cache in both branches — no change."""
        base = TrackedOperation(variable="orders", line=1, caches=[2])
        pr = TrackedOperation(variable="orders", line=1, caches=[2])
        results = detect_cache_removed(_match(base, pr), "test.py")
        self.assertEqual(len(results), 0)

    def test_cache_added_no_fire_for_removed(self):
        """Cache added should not fire cache_removed."""
        base = TrackedOperation(variable="orders", line=1, caches=[])
        pr = TrackedOperation(variable="orders", line=1, caches=[2])
        results = detect_cache_removed(_match(base, pr), "test.py")
        self.assertEqual(len(results), 0)


class TestCacheChangesE2E(unittest.TestCase):
    """End-to-end tests using extract_operations."""

    def test_cache_added_standalone(self):
        """df.cache() added as standalone expression."""
        base_source = '''\
orders = spark.table("orders")
result = orders.filter("x > 1")
'''
        pr_source = '''\
orders = spark.table("orders")
orders.cache()
result = orders.filter("x > 1")
'''
        base_ops = extract_operations(base_source)
        pr_ops = extract_operations(pr_source)
        base_orders = [op for op in base_ops if op.variable == "orders"][0]
        pr_orders = [op for op in pr_ops if op.variable == "orders"][0]
        results = detect_cache_added(_match(base_orders, pr_orders), "test.py")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].change_type, "cache_added")

    def test_cache_removed_standalone(self):
        """df.cache() removed."""
        base_source = '''\
orders = spark.table("orders")
orders.cache()
result = orders.filter("x > 1")
'''
        pr_source = '''\
orders = spark.table("orders")
result = orders.filter("x > 1")
'''
        base_ops = extract_operations(base_source)
        pr_ops = extract_operations(pr_source)
        base_orders = [op for op in base_ops if op.variable == "orders"][0]
        pr_orders = [op for op in pr_ops if op.variable == "orders"][0]
        results = detect_cache_removed(_match(base_orders, pr_orders), "test.py")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].change_type, "cache_removed")

    def test_cache_added_via_assignment(self):
        """cached = df.cache() added."""
        base_source = '''\
orders = spark.table("orders")
result = orders.filter("x > 1")
'''
        pr_source = '''\
orders = spark.table("orders")
cached = orders.cache()
result = cached.filter("x > 1")
'''
        base_ops = extract_operations(base_source)
        pr_ops = extract_operations(pr_source)
        # In PR, 'cached' has caches; in base, 'orders' doesn't
        pr_cached = [op for op in pr_ops if op.variable == "cached"][0]
        base_orders = [op for op in base_ops if op.variable == "orders"][0]
        results = detect_cache_added(_match(base_orders, pr_cached), "test.py")
        self.assertEqual(len(results), 1)

    def test_no_change_both_cached(self):
        """Cache in both branches."""
        source = '''\
orders = spark.table("orders")
orders.cache()
result = orders.filter("x > 1")
'''
        base_ops = extract_operations(source)
        pr_ops = extract_operations(source)
        base_orders = [op for op in base_ops if op.variable == "orders"][0]
        pr_orders = [op for op in pr_ops if op.variable == "orders"][0]
        results_added = detect_cache_added(_match(base_orders, pr_orders), "test.py")
        results_removed = detect_cache_removed(_match(base_orders, pr_orders), "test.py")
        self.assertEqual(len(results_added), 0)
        self.assertEqual(len(results_removed), 0)


if __name__ == "__main__":
    unittest.main()
