"""CY037: from pyspark.sql import functions — missing 'as F' alias."""

import ast

from cylint.models import Finding, RuleMeta, Severity
from cylint.rules import BaseRule, register_rule
from cylint.tracker import DataFrameTracker


@register_rule
class FunctionsAliasRule(BaseRule):
    META = RuleMeta(
        rule_id="CY037",
        name="functions-no-alias",
        description="pyspark.sql.functions imported without 'as F' alias",
        default_severity=Severity.INFO,
    )

    def check(self, tree: ast.Module, tracker: DataFrameTracker, filepath: str) -> list[Finding]:
        findings = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            # Match: from pyspark.sql import ...
            if node.module != "pyspark.sql":
                continue
            for alias in node.names:
                if alias.name == "functions" and alias.asname != "F":
                    findings.append(self._make_finding(
                        filepath=filepath,
                        line=node.lineno,
                        col=node.col_offset,
                        message=(
                            "pyspark.sql.functions imported without the standard 'F' alias. "
                            "The PySpark community convention is "
                            "'from pyspark.sql import functions as F', "
                            "which makes function calls self-documenting (F.col, F.lit, etc.) "
                            "and avoids shadowing built-in names."
                        ),
                        suggestion="Add 'as F': from pyspark.sql import functions as F",
                    ))
        return findings
