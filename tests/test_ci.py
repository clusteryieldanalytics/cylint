"""Tests for CI Action provenance modules: cell_map, enrich, comment."""

import unittest

from cylint.ci.cell_map import (
    SEPARATOR,
    absolute_to_cell,
    build_cell_map,
    cell_to_absolute,
    is_databricks_notebook,
)
from cylint.ci.enrich import (
    build_enrich_request,
    convert_changed_lines,
    convert_finding,
    resolve_provenance,
)
from cylint.ci.comment import (
    format_linter_finding,
    format_plan_finding,
    format_pr_comment,
)
from cylint.models import Finding, Severity


# ---------------------------------------------------------------------------
# Sample notebook source for tests
# ---------------------------------------------------------------------------

# 3 cells separated by COMMAND lines
NOTEBOOK_SOURCE = (
    "# cell 1 line 1\n"
    "# cell 1 line 2\n"
    f"{SEPARATOR}\n"
    "# cell 2 line 1\n"
    "# cell 2 line 2\n"
    "# cell 2 line 3\n"
    f"{SEPARATOR}\n"
    "# cell 3 line 1\n"
)

NOTEBOOK_WITH_HEADER = (
    "# Databricks notebook source\n"
    "# cell 1 line 1\n"
    "# cell 1 line 2\n"
    f"{SEPARATOR}\n"
    "# cell 2 line 1\n"
)

PLAIN_SOURCE = (
    "import pyspark\n"
    "df = spark.table('orders')\n"
    "df.show()\n"
)


# ---------------------------------------------------------------------------
# Cell Map Tests
# ---------------------------------------------------------------------------

class TestIsDatabricksNotebook(unittest.TestCase):
    def test_notebook_detected(self):
        self.assertTrue(is_databricks_notebook(NOTEBOOK_SOURCE))

    def test_plain_script_not_detected(self):
        self.assertFalse(is_databricks_notebook(PLAIN_SOURCE))


class TestBuildCellMap(unittest.TestCase):
    def setUp(self):
        self.cell_map = build_cell_map(NOTEBOOK_SOURCE)

    def test_three_cells(self):
        self.assertEqual(len(self.cell_map), 3)

    def test_first_cell_starts_at_1(self):
        starts = sorted(self.cell_map.values())
        self.assertEqual(starts[0], 1)

    def test_second_cell_starts_at_4(self):
        # cell1 has 2 lines + separator = line 3, so cell2 starts at 4
        starts = sorted(self.cell_map.values())
        self.assertEqual(starts[1], 4)

    def test_third_cell_starts_at_8(self):
        # cell2 has 3 lines + separator = line 7, so cell3 starts at 8
        starts = sorted(self.cell_map.values())
        self.assertEqual(starts[2], 8)

    def test_fingerprints_are_16_hex_chars(self):
        import re
        for fp in self.cell_map:
            self.assertRegex(fp, r"^[0-9a-f]{16}$")

    def test_duplicate_cells_last_wins(self):
        """Identical cells produce the same fingerprint; last occurrence stored."""
        src = (
            "same content\n"
            f"{SEPARATOR}\n"
            "same content\n"
        )
        cell_map = build_cell_map(src)
        # Both cells have identical content, so only one fingerprint
        self.assertEqual(len(cell_map), 1)
        # The stored start should be the second cell's start
        fp = list(cell_map.keys())[0]
        self.assertEqual(cell_map[fp], 3)

    def test_file_starting_with_separator(self):
        """Leading separator creates empty first segment, should be skipped."""
        src = (
            f"{SEPARATOR}\n"
            "cell content\n"
        )
        cell_map = build_cell_map(src)
        self.assertEqual(len(cell_map), 1)
        # Algorithm skips empty segment without advancing line counter,
        # so the cell is recorded at line 1
        start = list(cell_map.values())[0]
        self.assertEqual(start, 1)

    def test_header_line_stripped(self):
        """'# Databricks notebook source' header should be excluded from first cell."""
        cell_map = build_cell_map(NOTEBOOK_WITH_HEADER)
        self.assertEqual(len(cell_map), 2)
        starts = sorted(cell_map.values())
        # Header is line 1, so first cell content starts at line 2
        self.assertEqual(starts[0], 2)

    def test_header_not_in_fingerprint(self):
        """Fingerprint of first cell should match content without header."""
        import hashlib
        cell_map = build_cell_map(NOTEBOOK_WITH_HEADER)
        starts = sorted(cell_map.values())
        # Find the fp for the cell starting at line 2
        fp_with_header = [fp for fp, s in cell_map.items() if s == starts[0]][0]
        # Manually compute what the fingerprint should be (cell content only)
        from cylint.ci.cell_map import normalize_cell_source
        cell_content = "# cell 1 line 1\n# cell 1 line 2\n"
        expected_fp = hashlib.sha256(
            normalize_cell_source(cell_content).encode("utf-8")
        ).hexdigest()[:16]
        self.assertEqual(fp_with_header, expected_fp)

    def test_no_header_unchanged(self):
        """Notebook without header should work the same as before."""
        cell_map = build_cell_map(NOTEBOOK_SOURCE)
        starts = sorted(cell_map.values())
        self.assertEqual(starts[0], 1)


class TestAbsoluteToCell(unittest.TestCase):
    def setUp(self):
        self.cell_map = build_cell_map(NOTEBOOK_SOURCE)

    def test_line_in_first_cell(self):
        result = absolute_to_cell(2, self.cell_map)
        self.assertIsNotNone(result)
        fp, cell_line = result
        self.assertEqual(cell_line, 2)

    def test_line_in_second_cell(self):
        result = absolute_to_cell(5, self.cell_map)
        self.assertIsNotNone(result)
        fp, cell_line = result
        # Line 5 is second line of cell2 (cell2 starts at 4)
        self.assertEqual(cell_line, 2)

    def test_line_in_third_cell(self):
        result = absolute_to_cell(8, self.cell_map)
        self.assertIsNotNone(result)
        fp, cell_line = result
        self.assertEqual(cell_line, 1)

    def test_first_line_of_cell_is_1(self):
        result = absolute_to_cell(4, self.cell_map)
        self.assertIsNotNone(result)
        _, cell_line = result
        self.assertEqual(cell_line, 1)


class TestCellToAbsolute(unittest.TestCase):
    def setUp(self):
        self.cell_map = build_cell_map(NOTEBOOK_SOURCE)
        self.fps = sorted(self.cell_map.keys(), key=lambda fp: self.cell_map[fp])

    def test_roundtrip_first_cell(self):
        fp = self.fps[0]
        abs_line = cell_to_absolute(fp, 2, self.cell_map)
        self.assertEqual(abs_line, 2)

    def test_roundtrip_second_cell(self):
        fp = self.fps[1]
        abs_line = cell_to_absolute(fp, 1, self.cell_map)
        self.assertEqual(abs_line, 4)

    def test_unknown_fingerprint(self):
        result = cell_to_absolute("0000000000000000", 1, self.cell_map)
        self.assertIsNone(result)

    def test_roundtrip_consistency(self):
        """absolute_to_cell then cell_to_absolute should return the original."""
        for abs_line in [1, 2, 4, 5, 6, 8]:
            result = absolute_to_cell(abs_line, self.cell_map)
            if result:
                fp, cl = result
                back = cell_to_absolute(fp, cl, self.cell_map)
                self.assertEqual(back, abs_line, f"Roundtrip failed for line {abs_line}")


# ---------------------------------------------------------------------------
# Enrich Request Tests
# ---------------------------------------------------------------------------

class TestConvertFinding(unittest.TestCase):
    def _make_finding(self, line=10):
        return Finding(
            rule_id="CY009",
            severity=Severity.WARNING,
            message="UDF in .filter() blocks pushdown",
            filepath="pipelines/order_report.py",
            line=line,
        )

    def test_notebook_finding_has_cell_coords(self):
        cell_map = build_cell_map(NOTEBOOK_SOURCE)
        result = convert_finding(self._make_finding(line=5), cell_map)
        self.assertIn("cellFingerprint", result)
        self.assertIn("cellLine", result)
        self.assertIn("absoluteLine", result)
        self.assertEqual(result["absoluteLine"], 5)
        self.assertNotIn("line", result)

    def test_plain_finding_has_line(self):
        result = convert_finding(self._make_finding(line=10), None)
        self.assertIn("line", result)
        self.assertEqual(result["line"], 10)
        self.assertNotIn("cellFingerprint", result)
        self.assertNotIn("absoluteLine", result)

    def test_finding_has_rule_and_message(self):
        result = convert_finding(self._make_finding(), None)
        self.assertEqual(result["rule"], "CY009")
        self.assertEqual(result["file"], "pipelines/order_report.py")
        self.assertIn("message", result)


class TestConvertChangedLines(unittest.TestCase):
    def test_notebook_changed_lines(self):
        cell_map = build_cell_map(NOTEBOOK_SOURCE)
        result = convert_changed_lines([4, 5], "test.py", cell_map)
        self.assertEqual(len(result), 2)
        for entry in result:
            self.assertIn("cellFingerprint", entry)
            self.assertIn("cellLine", entry)

    def test_plain_changed_lines(self):
        result = convert_changed_lines([7, 8], "test.py", None)
        self.assertEqual(len(result), 2)
        for entry in result:
            self.assertEqual(entry["file"], "test.py")
            self.assertIn("line", entry)


class TestBuildEnrichRequest(unittest.TestCase):
    def test_notebook_request_shape(self):
        finding = Finding(
            rule_id="CY009", severity=Severity.WARNING,
            message="test", filepath="test.py", line=5,
        )
        req = build_enrich_request("test.py", NOTEBOOK_SOURCE, [finding], [4, 5])
        self.assertIn("files", req)
        self.assertIn("linterFindings", req)
        self.assertIn("changedLines", req)
        self.assertEqual(req["files"][0]["path"], "test.py")
        self.assertIn("cellFingerprint", req["linterFindings"][0])

    def test_plain_request_shape(self):
        finding = Finding(
            rule_id="CY009", severity=Severity.WARNING,
            message="test", filepath="test.py", line=2,
        )
        req = build_enrich_request("test.py", PLAIN_SOURCE, [finding], [2])
        self.assertIn("line", req["linterFindings"][0])
        self.assertNotIn("cellFingerprint", req["linterFindings"][0])

    def test_empty_findings_and_lines(self):
        req = build_enrich_request("test.py", PLAIN_SOURCE, [], [])
        self.assertEqual(req["linterFindings"], [])
        self.assertEqual(req["changedLines"], [])
        self.assertEqual(len(req["files"]), 1)


class TestResolveProvenance(unittest.TestCase):
    def setUp(self):
        self.cell_map = build_cell_map(NOTEBOOK_SOURCE)
        self.fps = sorted(self.cell_map.keys(), key=lambda fp: self.cell_map[fp])

    def test_trigger_line_resolved(self):
        finding = {
            "triggerCellFingerprint": self.fps[2],
            "triggerLine": 1,
            "constructionLines": [],
        }
        result = resolve_provenance(finding, self.cell_map)
        self.assertEqual(result["triggerLineAbsolute"], 8)

    def test_construction_lines_resolved(self):
        finding = {
            "triggerCellFingerprint": self.fps[2],
            "triggerLine": 1,
            "constructionLines": [
                {"cellFingerprint": self.fps[0], "lines": [1, 2]},
                {"cellFingerprint": self.fps[1], "lines": [1, 2, 3]},
            ],
        }
        result = resolve_provenance(finding, self.cell_map)
        self.assertEqual(result["constructionLinesAbsolute"], [1, 2, 4, 5, 6])
        self.assertEqual(result["constructionSpanStart"], 1)
        self.assertEqual(result["constructionSpanEnd"], 6)

    def test_unknown_fingerprint_skipped(self):
        finding = {
            "triggerCellFingerprint": "0000000000000000",
            "triggerLine": 1,
            "constructionLines": [
                {"cellFingerprint": "0000000000000000", "lines": [1]},
            ],
        }
        result = resolve_provenance(finding, self.cell_map)
        self.assertIsNone(result["triggerLineAbsolute"])
        self.assertEqual(result["constructionLinesAbsolute"], [])
        self.assertIsNone(result["constructionSpanStart"])

    def test_empty_construction_lines(self):
        finding = {
            "triggerCellFingerprint": self.fps[0],
            "triggerLine": 1,
            "constructionLines": [],
        }
        result = resolve_provenance(finding, self.cell_map)
        self.assertEqual(result["constructionLinesAbsolute"], [])
        self.assertIsNone(result["constructionSpanStart"])
        self.assertIsNone(result["constructionSpanEnd"])


# ---------------------------------------------------------------------------
# Comment Formatting Tests
# ---------------------------------------------------------------------------

class TestFormatLinterFinding(unittest.TestCase):
    def test_basic_format(self):
        finding = {
            "file": "order_report.py",
            "absoluteLine": 10,
            "rule": "CY009",
            "message": "UDF in .filter() blocks pushdown",
        }
        text = format_linter_finding(finding)
        self.assertIn("order_report.py:10", text)
        self.assertIn("CY009", text)

    def test_savings_included(self):
        finding = {
            "file": "order_report.py",
            "line": 10,
            "rule": "CY009",
            "message": "UDF in .filter()",
            "savings": 3360,
        }
        text = format_linter_finding(finding)
        self.assertIn("$3,360/month", text)


class TestFormatPlanFinding(unittest.TestCase):
    def test_basic_format(self):
        finding = {
            "file": "order_report.py",
            "triggerLineAbsolute": 15,
            "constructionSpanStart": 1,
            "constructionSpanEnd": 13,
            "detectorId": "broadcast-miss",
            "message": "Broadcast miss on customers join",
            "savings": 3900,
        }
        text = format_plan_finding(finding)
        self.assertIn("order_report.py:15", text)
        self.assertIn("lines 1-13", text)
        self.assertIn("$3,900/month", text)


class TestFormatPrComment(unittest.TestCase):
    def test_empty_findings(self):
        text = format_pr_comment([], [])
        self.assertIn("No findings detected.", text)

    def test_linter_only(self):
        findings = [{
            "file": "test.py", "line": 1,
            "rule": "CY001", "message": "test",
        }]
        text = format_pr_comment(findings)
        self.assertIn("Linter Findings", text)
        self.assertNotIn("Plan Detector", text)

    def test_both_sections(self):
        linter = [{"file": "t.py", "line": 1, "rule": "CY001", "message": "m"}]
        plan = [{"file": "t.py", "triggerLineAbsolute": 5,
                 "detectorId": "x", "message": "m", "savings": 100}]
        text = format_pr_comment(linter, plan)
        self.assertIn("Linter Findings", text)
        self.assertIn("Plan Detector", text)


if __name__ == "__main__":
    unittest.main()
