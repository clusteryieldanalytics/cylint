"""Tests for comment.py — PR comment formatting and annotations."""

import unittest

from cylint.ci.comment import (
    FormattedOutput,
    format_change_classification,
    format_change_finding,
    format_linter_finding,
    format_output,
    format_plan_finding,
    format_pr_comment,
)


class TestFormatLinterFinding(unittest.TestCase):
    def test_basic_finding(self):
        f = {"file": "test.py", "line": 10, "rule": "CY001", "message": "collect without filter"}
        result = format_linter_finding(f)
        self.assertIn("test.py:10", result)
        self.assertIn("CY001", result)

    def test_enriched_with_savings(self):
        f = {"file": "test.py", "line": 10, "rule": "CY001", "message": "test", "savings": 5000}
        result = format_linter_finding(f)
        self.assertIn("$5,000/month", result)

    def test_notebook_uses_absolute_line(self):
        f = {"file": "nb.py", "absoluteLine": 42, "rule": "CY001", "message": "test"}
        result = format_linter_finding(f)
        self.assertIn("nb.py:42", result)


class TestFormatPlanFinding(unittest.TestCase):
    def test_with_span(self):
        f = {
            "file": "test.py",
            "triggerLineAbsolute": 20,
            "constructionSpanStart": 10,
            "constructionSpanEnd": 25,
            "message": "full table scan",
            "savings": 10000,
        }
        result = format_plan_finding(f)
        self.assertIn("test.py:20", result)
        self.assertIn("lines 10-25", result)
        self.assertIn("$10,000/month", result)

    def test_without_span(self):
        f = {"file": "test.py", "triggerLineAbsolute": 5, "message": "scan"}
        result = format_plan_finding(f)
        self.assertIn("test.py:5", result)
        self.assertNotIn("spans", result)


class TestFormatChangeFinding(unittest.TestCase):
    def test_precise_tier(self):
        f = {"type": "filter_removed", "message": "Filter removed on orders"}
        result = format_change_finding(f, enriched=False)
        self.assertIn("High confidence", result)

    def test_bounded_tier(self):
        f = {"type": "filter_modified", "message": "Filter modified"}
        result = format_change_finding(f, enriched=False)
        self.assertIn("Range estimate", result)

    def test_enriched_with_savings(self):
        f = {"type": "filter_removed", "message": "test", "savings": 20000}
        result = format_change_finding(f, enriched=True)
        self.assertIn("$20,000/month", result)


class TestFormatChangeClassification(unittest.TestCase):
    def test_with_table(self):
        c = {"type": "filter_removed", "sourceTable": "orders", "scope": "line 42"}
        result = format_change_classification(c)
        self.assertIn("filter_removed", result)
        self.assertIn("`orders`", result)


class TestFormatOutput(unittest.TestCase):
    def test_no_findings(self):
        result = format_output([], [], [], [], None, False)
        self.assertIn("No findings", result.markdown)
        self.assertEqual(len(result.annotations), 0)

    def test_linter_findings_only(self):
        findings = [
            {"file": "test.py", "line": 10, "rule": "CY001", "message": "collect"},
        ]
        result = format_output(findings, [], [], [], None, False)
        self.assertIn("Cluster Yield", result.markdown)
        self.assertIn("1 finding", result.markdown)
        self.assertEqual(len(result.annotations), 1)
        self.assertEqual(result.annotations[0]["level"], "warning")

    def test_enriched_with_savings(self):
        findings = [
            {"file": "a.py", "line": 1, "rule": "CY001", "message": "x", "savings": 5000},
        ]
        result = format_output(findings, [], [], [], None, True)
        self.assertIn("$5,000/month", result.markdown)

    def test_plan_findings_produce_error_annotations(self):
        plan = [
            {"file": "test.py", "absoluteTriggerLine": 20, "message": "full scan"},
        ]
        result = format_output([], [], plan, [], None, False)
        self.assertEqual(len(result.annotations), 1)
        self.assertEqual(result.annotations[0]["level"], "error")

    def test_change_classifications_shown_when_unenriched(self):
        classifications = [
            {"type": "filter_removed", "sourceTable": "orders"},
        ]
        result = format_output([], [], [], classifications, None, False)
        self.assertIn("Detected Changes", result.markdown)

    def test_match_stats_warning(self):
        stats = {"fingerprintMatchRate": 0.5}
        findings = [{"file": "a.py", "line": 1, "rule": "CY001", "message": "x"}]
        result = format_output(findings, [], [], [], stats, True)
        self.assertIn("50%", result.markdown)

    def test_to_dict(self):
        result = format_output([], [], [], [], None, False)
        d = result.to_dict()
        self.assertIn("markdown", d)
        self.assertIn("annotations", d)
        self.assertIn("stats", d)


class TestFormatPrCommentBackwardsCompat(unittest.TestCase):
    def test_basic(self):
        result = format_pr_comment([{"file": "a.py", "line": 1, "rule": "CY001", "message": "x"}])
        self.assertIn("Cluster Yield", result)
        self.assertIn("CY001", result)

    def test_empty(self):
        result = format_pr_comment([])
        self.assertIn("No findings", result)


if __name__ == "__main__":
    unittest.main()
