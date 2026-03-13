"""Tests for classifier.py — operation extraction and end-to-end classification."""

import unittest
from unittest.mock import patch

from cylint.diff.classifier import DiffClassifier, extract_operations


class TestExtractOperations(unittest.TestCase):
    """Test the operation extractor on PySpark source snippets."""

    def test_simple_spark_table(self):
        source = 'orders = spark.table("orders")\n'
        ops = extract_operations(source)
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0].variable, "orders")
        self.assertEqual(ops[0].source_table, "orders")

    def test_filter_recorded(self):
        source = '''\
orders = spark.table("orders")
recent = orders.filter("date > '2024-01-01'")
'''
        ops = extract_operations(source)
        # Should have ops for both 'orders' and 'recent'
        recent_ops = [op for op in ops if op.variable == "recent"]
        self.assertEqual(len(recent_ops), 1)
        self.assertGreater(len(recent_ops[0].filters), 0)
        self.assertEqual(recent_ops[0].source_table, "orders")

    def test_select_recorded(self):
        source = '''\
orders = spark.table("orders")
slim = orders.select("id", "amount")
'''
        ops = extract_operations(source)
        slim_ops = [op for op in ops if op.variable == "slim"]
        self.assertEqual(len(slim_ops), 1)
        self.assertGreater(len(slim_ops[0].selects), 0)
        self.assertEqual(slim_ops[0].selects[-1].col_count, 2)
        self.assertEqual(slim_ops[0].selects[-1].col_names, ["id", "amount"])

    def test_chained_filter(self):
        source = '''\
orders = spark.table("orders")
filtered = orders.filter("date > '2024'").filter("status = 'active'")
'''
        ops = extract_operations(source)
        filtered_ops = [op for op in ops if op.variable == "filtered"]
        self.assertEqual(len(filtered_ops), 1)
        # Should have 2 filters from the chain
        self.assertGreaterEqual(len(filtered_ops[0].filters), 2)

    def test_cache_recorded(self):
        source = '''\
orders = spark.table("orders")
cached = orders.cache()
'''
        ops = extract_operations(source)
        cached_ops = [op for op in ops if op.variable == "cached"]
        self.assertEqual(len(cached_ops), 1)
        self.assertGreater(len(cached_ops[0].caches), 0)

    def test_spark_read_parquet(self):
        source = 'data = spark.read.parquet("/data/events")\n'
        ops = extract_operations(source)
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0].source_table, "/data/events")

    def test_groupby_recorded(self):
        source = '''\
orders = spark.table("orders")
grouped = orders.groupBy("region").agg({"amount": "sum"})
'''
        ops = extract_operations(source)
        grouped_ops = [op for op in ops if op.variable == "grouped"]
        self.assertEqual(len(grouped_ops), 1)
        self.assertGreater(len(grouped_ops[0].groupbys), 0)

    def test_no_dataframes(self):
        source = "x = 1\ny = x + 2\n"
        ops = extract_operations(source)
        self.assertEqual(len(ops), 0)

    def test_where_alias(self):
        """.where() should be recorded as a filter."""
        source = '''\
orders = spark.table("orders")
active = orders.where("status = 'active'")
'''
        ops = extract_operations(source)
        active_ops = [op for op in ops if op.variable == "active"]
        self.assertEqual(len(active_ops), 1)
        self.assertGreater(len(active_ops[0].filters), 0)


class TestExtractOperationsFilterHashes(unittest.TestCase):
    """Test that filter expr_hashes detect structural changes."""

    def test_same_filter_same_hash(self):
        src = '''\
orders = spark.table("orders")
a = orders.filter("date > '2024'")
'''
        ops1 = extract_operations(src)
        ops2 = extract_operations(src)
        f1 = [op for op in ops1 if op.variable == "a"][0].filters[0]
        f2 = [op for op in ops2 if op.variable == "a"][0].filters[0]
        self.assertEqual(f1.expr_hash, f2.expr_hash)

    def test_different_filter_different_hash(self):
        src1 = '''\
orders = spark.table("orders")
a = orders.filter("date > '2024'")
'''
        src2 = '''\
orders = spark.table("orders")
a = orders.filter("date > '2023'")
'''
        ops1 = extract_operations(src1)
        ops2 = extract_operations(src2)
        f1 = [op for op in ops1 if op.variable == "a"][0].filters[0]
        f2 = [op for op in ops2 if op.variable == "a"][0].filters[0]
        self.assertNotEqual(f1.expr_hash, f2.expr_hash)


class TestDiffClassifierClassifyFile(unittest.TestCase):
    """Test DiffClassifier.classify_file with mocked git."""

    @patch("cylint.diff.classifier.get_base_source")
    def test_filter_removed(self, mock_git):
        base_source = '''\
orders = spark.table("orders")
recent = orders.filter("order_date > '2024-01-01'")
recent.write.saveAsTable("output")
'''
        pr_source = '''\
orders = spark.table("orders")
recent = orders
recent.write.saveAsTable("output")
'''
        mock_git.return_value = base_source

        import tempfile, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(pr_source)
            f.flush()
            try:
                classifier = DiffClassifier(base_ref="origin/main")
                results = classifier.classify_file(f.name)
                types = [r.change_type for r in results]
                self.assertIn("filter_removed", types)
            finally:
                os.unlink(f.name)

    @patch("cylint.diff.classifier.get_base_source")
    def test_filter_modified(self, mock_git):
        base_source = '''\
orders = spark.table("orders")
recent = orders.filter("order_date > '2024-01-01'")
'''
        pr_source = '''\
orders = spark.table("orders")
recent = orders.filter("order_date > '2023-06-01'")
'''
        mock_git.return_value = base_source

        import tempfile, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(pr_source)
            f.flush()
            try:
                classifier = DiffClassifier(base_ref="origin/main")
                results = classifier.classify_file(f.name)
                types = [r.change_type for r in results]
                self.assertIn("filter_modified", types)
            finally:
                os.unlink(f.name)

    @patch("cylint.diff.classifier.get_base_source")
    def test_source_changed(self, mock_git):
        base_source = 'orders = spark.table("orders_v2")\n'
        pr_source = 'orders = spark.table("orders_raw")\n'
        mock_git.return_value = base_source

        import tempfile, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(pr_source)
            f.flush()
            try:
                classifier = DiffClassifier(base_ref="origin/main")
                results = classifier.classify_file(f.name)
                types = [r.change_type for r in results]
                self.assertIn("source_changed", types)
                sc = [r for r in results if r.change_type == "source_changed"][0]
                self.assertEqual(sc.metadata["oldTable"], "orders_v2")
                self.assertEqual(sc.metadata["newTable"], "orders_raw")
            finally:
                os.unlink(f.name)

    @patch("cylint.diff.classifier.get_base_source")
    def test_new_file_no_classification(self, mock_git):
        mock_git.return_value = None  # no base version
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write('orders = spark.table("orders")\n')
            f.flush()
            try:
                classifier = DiffClassifier(base_ref="origin/main")
                results = classifier.classify_file(f.name)
                self.assertEqual(len(results), 0)
            finally:
                os.unlink(f.name)

    @patch("cylint.diff.classifier.get_base_source")
    def test_deleted_file_produces_operation_removed(self, mock_git):
        mock_git.return_value = 'orders = spark.table("orders")\n'
        classifier = DiffClassifier(base_ref="origin/main")
        # File doesn't exist — simulates deletion
        results = classifier.classify_file("/tmp/nonexistent_file_xyz.py")
        self.assertTrue(any(r.change_type == "operation_removed" for r in results))

    @patch("cylint.diff.classifier.get_base_source")
    def test_projection_changed(self, mock_git):
        base_source = '''\
orders = spark.table("orders")
slim = orders.select("id", "amount")
'''
        pr_source = '''\
orders = spark.table("orders")
slim = orders.select("id", "amount", "status", "date", "region")
'''
        mock_git.return_value = base_source

        import tempfile, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(pr_source)
            f.flush()
            try:
                classifier = DiffClassifier(base_ref="origin/main")
                results = classifier.classify_file(f.name)
                types = [r.change_type for r in results]
                self.assertIn("projection_changed", types)
                pc = [r for r in results if r.change_type == "projection_changed"][0]
                self.assertEqual(pc.metadata["oldColCount"], 2)
                self.assertEqual(pc.metadata["newColCount"], 5)
            finally:
                os.unlink(f.name)

    @patch("cylint.diff.classifier.get_base_source")
    def test_no_changes_no_classifications(self, mock_git):
        source = '''\
orders = spark.table("orders")
recent = orders.filter("date > '2024'")
'''
        mock_git.return_value = source

        import tempfile, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(source)
            f.flush()
            try:
                classifier = DiffClassifier(base_ref="origin/main")
                results = classifier.classify_file(f.name)
                self.assertEqual(len(results), 0)
            finally:
                os.unlink(f.name)

    @patch("cylint.diff.classifier.get_base_source")
    def test_filter_added(self, mock_git):
        base_source = '''\
orders = spark.table("orders")
result = orders
'''
        pr_source = '''\
orders = spark.table("orders")
result = orders.filter("status = 'active'")
'''
        mock_git.return_value = base_source

        import tempfile, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(pr_source)
            f.flush()
            try:
                classifier = DiffClassifier(base_ref="origin/main")
                results = classifier.classify_file(f.name)
                types = [r.change_type for r in results]
                self.assertIn("filter_added", types)
            finally:
                os.unlink(f.name)


class TestDiffClassifierMatchesPaths(unittest.TestCase):
    def test_matches_directory(self):
        classifier = DiffClassifier(base_ref="origin/main")
        self.assertTrue(classifier._matches_paths("src/foo.py", ["src/"]))
        self.assertFalse(classifier._matches_paths("lib/foo.py", ["src/"]))

    def test_matches_exact(self):
        classifier = DiffClassifier(base_ref="origin/main")
        self.assertTrue(classifier._matches_paths("src/foo.py", ["src/foo.py"]))


if __name__ == "__main__":
    unittest.main()
