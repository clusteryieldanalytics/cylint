"""Tests for udf_added and udf_removed detectors."""

import unittest

from cylint.diff.classifier import extract_operations
from cylint.diff.detectors import detect_udf_added, detect_udf_removed
from cylint.diff.models import OperationMatch, TrackedOperation, UdfOp


def _match(base_op, pr_op, confidence="high"):
    return OperationMatch(base_op, pr_op, confidence=confidence, match_strategy="source_table")


class TestUdfAdded(unittest.TestCase):
    """Unit tests using constructed TrackedOperations."""

    def test_udf_added_filter(self):
        base = TrackedOperation(variable="orders", line=1, udfs=[])
        pr = TrackedOperation(variable="orders", line=1,
                              udfs=[UdfOp(line=3, context="filter", name="my_filter")])
        results = detect_udf_added(_match(base, pr), "test.py")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].change_type, "udf_added")
        self.assertEqual(results[0].metadata["udfContext"], "filter")

    def test_udf_added_withcolumn(self):
        base = TrackedOperation(variable="orders", line=1, udfs=[])
        pr = TrackedOperation(variable="orders", line=1,
                              udfs=[UdfOp(line=3, context="withColumn", name="clean_name")])
        results = detect_udf_added(_match(base, pr), "test.py")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].metadata["udfContext"], "withColumn")

    def test_same_udf_both_branches_no_fire(self):
        udf = UdfOp(line=3, context="filter", name="my_filter")
        base = TrackedOperation(variable="orders", line=1, udfs=[udf])
        pr = TrackedOperation(variable="orders", line=1, udfs=[udf])
        results = detect_udf_added(_match(base, pr), "test.py")
        self.assertEqual(len(results), 0)

    def test_no_udfs_either_no_fire(self):
        base = TrackedOperation(variable="orders", line=1, udfs=[])
        pr = TrackedOperation(variable="orders", line=1, udfs=[])
        results = detect_udf_added(_match(base, pr), "test.py")
        self.assertEqual(len(results), 0)


class TestUdfRemoved(unittest.TestCase):
    """Unit tests using constructed TrackedOperations."""

    def test_udf_removed_filter(self):
        base = TrackedOperation(variable="orders", line=1, source_table="orders",
                                udfs=[UdfOp(line=3, context="filter", name="my_filter")])
        pr = TrackedOperation(variable="orders", line=1, udfs=[])
        results = detect_udf_removed(_match(base, pr), "test.py")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].change_type, "udf_removed")
        self.assertEqual(results[0].metadata["udfContext"], "filter")

    def test_same_udf_both_branches_no_fire(self):
        udf = UdfOp(line=3, context="filter", name="my_filter")
        base = TrackedOperation(variable="orders", line=1, udfs=[udf])
        pr = TrackedOperation(variable="orders", line=1, udfs=[udf])
        results = detect_udf_removed(_match(base, pr), "test.py")
        self.assertEqual(len(results), 0)


class TestUdfAddedE2E(unittest.TestCase):
    """End-to-end tests using extract_operations."""

    def test_udf_added_to_filter(self):
        base_source = '''\
orders = spark.table("orders")
filtered = orders.filter(F.col("amount") > 100)
'''
        pr_source = '''\
from pyspark.sql.functions import udf
my_filter = udf(lambda x: x > 100)
orders = spark.table("orders")
filtered = orders.filter(my_filter(F.col("amount")))
'''
        base_ops = extract_operations(base_source)
        pr_ops = extract_operations(pr_source)
        base_filtered = [op for op in base_ops if op.variable == "filtered"][0]
        pr_filtered = [op for op in pr_ops if op.variable == "filtered"][0]
        results = detect_udf_added(_match(base_filtered, pr_filtered), "test.py")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].change_type, "udf_added")
        self.assertEqual(results[0].metadata["udfContext"], "filter")

    def test_udf_added_to_withcolumn(self):
        base_source = '''\
orders = spark.table("orders")
result = orders.withColumn("clean", F.trim(F.col("name")))
'''
        pr_source = '''\
from pyspark.sql.functions import udf
clean_name = udf(lambda x: x.strip() if x else None)
orders = spark.table("orders")
result = orders.withColumn("clean", clean_name(F.col("name")))
'''
        base_ops = extract_operations(base_source)
        pr_ops = extract_operations(pr_source)
        base_result = [op for op in base_ops if op.variable == "result"][0]
        pr_result = [op for op in pr_ops if op.variable == "result"][0]
        results = detect_udf_added(_match(base_result, pr_result), "test.py")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].metadata["udfContext"], "withColumn")

    def test_no_udf_either_no_fire(self):
        source = '''\
orders = spark.table("orders")
filtered = orders.filter(F.col("amount") > 100)
'''
        base_ops = extract_operations(source)
        pr_ops = extract_operations(source)
        base_filtered = [op for op in base_ops if op.variable == "filtered"][0]
        pr_filtered = [op for op in pr_ops if op.variable == "filtered"][0]
        results = detect_udf_added(_match(base_filtered, pr_filtered), "test.py")
        self.assertEqual(len(results), 0)

    def test_same_udf_both_no_fire(self):
        source = '''\
from pyspark.sql.functions import udf
my_filter = udf(lambda x: x > 100)
orders = spark.table("orders")
filtered = orders.filter(my_filter(F.col("amount")))
'''
        base_ops = extract_operations(source)
        pr_ops = extract_operations(source)
        base_filtered = [op for op in base_ops if op.variable == "filtered"][0]
        pr_filtered = [op for op in pr_ops if op.variable == "filtered"][0]
        results = detect_udf_added(_match(base_filtered, pr_filtered), "test.py")
        self.assertEqual(len(results), 0)


class TestUdfRemovedE2E(unittest.TestCase):
    """End-to-end tests using extract_operations."""

    def test_udf_removed_from_filter(self):
        base_source = '''\
from pyspark.sql.functions import udf
my_filter = udf(lambda x: x > 100)
orders = spark.table("orders")
filtered = orders.filter(my_filter(F.col("amount")))
'''
        pr_source = '''\
orders = spark.table("orders")
filtered = orders.filter(F.col("amount") > 100)
'''
        base_ops = extract_operations(base_source)
        pr_ops = extract_operations(pr_source)
        base_filtered = [op for op in base_ops if op.variable == "filtered"][0]
        pr_filtered = [op for op in pr_ops if op.variable == "filtered"][0]
        results = detect_udf_removed(_match(base_filtered, pr_filtered), "test.py")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].change_type, "udf_removed")
        self.assertEqual(results[0].metadata["udfContext"], "filter")

    def test_decorator_udf_removed(self):
        base_source = '''\
@udf("string")
def lower_name(x):
    return x.lower() if x else None
orders = spark.table("orders")
result = orders.withColumn("name_lower", lower_name(F.col("name")))
'''
        pr_source = '''\
orders = spark.table("orders")
result = orders.withColumn("name_lower", F.lower(F.col("name")))
'''
        base_ops = extract_operations(base_source)
        pr_ops = extract_operations(pr_source)
        base_result = [op for op in base_ops if op.variable == "result"][0]
        pr_result = [op for op in pr_ops if op.variable == "result"][0]
        results = detect_udf_removed(_match(base_result, pr_result), "test.py")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].metadata["udfContext"], "withColumn")


if __name__ == "__main__":
    unittest.main()
