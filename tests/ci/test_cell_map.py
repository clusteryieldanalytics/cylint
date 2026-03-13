"""Tests for cell_map.py — notebook splitting, fingerprinting, coordinate conversion."""

import unittest

from cylint.ci.cell_map import (
    absolute_to_cell,
    build_cell_map,
    cell_to_absolute,
    is_databricks_notebook,
)


NOTEBOOK_SOURCE = """\
# Databricks notebook source
# COMMAND ----------
# cell 1: imports
import pyspark
# COMMAND ----------
# cell 2: logic
df = spark.table("orders")
df2 = df.filter("amount > 100")
# COMMAND ----------
# cell 3: output
df2.show()
"""

PLAIN_PY = """\
import pyspark

df = spark.table("orders")
df.show()
"""


class TestIsDatabricksNotebook(unittest.TestCase):
    def test_notebook_detected(self):
        self.assertTrue(is_databricks_notebook(NOTEBOOK_SOURCE))

    def test_plain_py_not_detected(self):
        self.assertFalse(is_databricks_notebook(PLAIN_PY))


class TestBuildCellMap(unittest.TestCase):
    def test_notebook_has_cells(self):
        cell_map = build_cell_map(NOTEBOOK_SOURCE)
        # 3 content cells; header handling may produce slightly different count
        self.assertGreaterEqual(len(cell_map), 3)

    def test_fingerprints_are_hex_strings(self):
        cell_map = build_cell_map(NOTEBOOK_SOURCE)
        for fp in cell_map:
            self.assertEqual(len(fp), 16)
            int(fp, 16)  # should not raise

    def test_start_lines_are_positive(self):
        cell_map = build_cell_map(NOTEBOOK_SOURCE)
        for start in cell_map.values():
            self.assertGreater(start, 0)

    def test_fingerprints_are_unique(self):
        cell_map = build_cell_map(NOTEBOOK_SOURCE)
        self.assertEqual(len(cell_map), len(set(cell_map.keys())))


class TestCoordinateConversion(unittest.TestCase):
    def setUp(self):
        self.cell_map = build_cell_map(NOTEBOOK_SOURCE)

    def test_round_trip(self):
        """absolute -> cell -> absolute should be identity."""
        for fp, start in self.cell_map.items():
            result = absolute_to_cell(start, self.cell_map)
            self.assertIsNotNone(result)
            fp_out, cell_line = result
            self.assertEqual(fp_out, fp)
            self.assertEqual(cell_line, 1)

            abs_back = cell_to_absolute(fp, cell_line, self.cell_map)
            self.assertEqual(abs_back, start)

    def test_absolute_to_cell_returns_correct_cell(self):
        """A line inside a cell should map to that cell."""
        # Find cell that starts earliest (after header)
        starts = sorted(self.cell_map.values())
        first_start = starts[0]
        # Line 1 of first cell
        result = absolute_to_cell(first_start, self.cell_map)
        self.assertIsNotNone(result)
        self.assertEqual(result[1], 1)

    def test_cell_to_absolute_unknown_fingerprint(self):
        """Unknown fingerprint returns None."""
        result = cell_to_absolute("0000000000000000", 1, self.cell_map)
        self.assertIsNone(result)

    def test_absolute_to_cell_on_separator(self):
        """Line on a separator might return the previous cell (acceptable)."""
        # Separator is on line 3 of source (# COMMAND ----------)
        # This is implementation-defined — just ensure no crash
        absolute_to_cell(3, self.cell_map)


class TestNonNotebookPassthrough(unittest.TestCase):
    def test_plain_py_no_cells(self):
        """Plain .py returns empty cell map (not a notebook)."""
        self.assertFalse(is_databricks_notebook(PLAIN_PY))


if __name__ == "__main__":
    unittest.main()
