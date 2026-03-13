"""Tests for inline # cy:ignore suppression."""

import unittest
from cylint.engine import LintEngine


class TestCyIgnore(unittest.TestCase):
    """Inline # cy:ignore comment suppression."""

    def setUp(self):
        self.engine = LintEngine()

    def lint(self, source: str) -> list:
        return self.engine.lint_source(source, filepath="test.py")

    def test_suppress_specific_rule(self):
        """# cy:ignore CY001 suppresses only CY001 on that line."""
        src = '''\
df = spark.table("orders")
df.collect()  # cy:ignore CY001
'''
        findings = self.lint(src)
        cy001 = [f for f in findings if f.rule_id == "CY001"]
        self.assertEqual(len(cy001), 0)

    def test_suppress_all_rules(self):
        """Bare # cy:ignore suppresses all rules on that line."""
        src = '''\
df = spark.table("orders")
df.collect()  # cy:ignore
'''
        findings = self.lint(src)
        cy001 = [f for f in findings if f.rule_id == "CY001"]
        self.assertEqual(len(cy001), 0)

    def test_suppress_multiple_rules(self):
        """# cy:ignore CY001,CY012 suppresses both rules."""
        src = '''\
df = spark.table("orders")
df.show()  # cy:ignore CY001,CY012
'''
        findings = self.lint(src)
        suppressed = [f for f in findings if f.rule_id in ("CY001", "CY012")]
        self.assertEqual(len(suppressed), 0)

    def test_wrong_rule_id_not_suppressed(self):
        """# cy:ignore CY999 does not suppress CY001."""
        src = '''\
df = spark.table("orders")
df.collect()  # cy:ignore CY999
'''
        findings = self.lint(src)
        cy001 = [f for f in findings if f.rule_id == "CY001"]
        self.assertGreater(len(cy001), 0)

    def test_no_comment_not_suppressed(self):
        """Lines without # cy:ignore still produce findings."""
        src = '''\
df = spark.table("orders")
df.collect()
'''
        findings = self.lint(src)
        cy001 = [f for f in findings if f.rule_id == "CY001"]
        self.assertGreater(len(cy001), 0)

    def test_only_affects_same_line(self):
        """# cy:ignore on one line does not suppress findings on other lines."""
        src = '''\
df = spark.table("orders")
df.collect()  # cy:ignore CY001
df.collect()
'''
        findings = self.lint(src)
        cy001 = [f for f in findings if f.rule_id == "CY001"]
        self.assertEqual(len(cy001), 1)
        self.assertEqual(cy001[0].line, 3)

    def test_spaces_in_rule_list(self):
        """Spaces around commas in rule list are tolerated."""
        src = '''\
df = spark.table("orders")
df.collect()  # cy:ignore CY001 , CY012
'''
        findings = self.lint(src)
        cy001 = [f for f in findings if f.rule_id == "CY001"]
        self.assertEqual(len(cy001), 0)


if __name__ == "__main__":
    unittest.main()
