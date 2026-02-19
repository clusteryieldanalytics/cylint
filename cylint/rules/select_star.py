"""CY004: SELECT * in SQL strings passed to spark.sql()."""

import ast
import re

from cylint.models import Finding, RuleMeta, Severity
from cylint.rules import BaseRule, register_rule
from cylint.tracker import DataFrameTracker

SELECT_STAR_PATTERN = re.compile(
    r"\bSELECT\s+\*\s+FROM\b",
    re.IGNORECASE,
)


@register_rule
class SelectStarRule(BaseRule):
    META = RuleMeta(
        rule_id="CY004",
        name="select-star",
        description="SELECT * in SQL strings prevents column pruning",
        default_severity=Severity.INFO,
    )

    def check(self, tree: ast.Module, tracker: DataFrameTracker, filepath: str) -> list[Finding]:
        findings = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            # Match spark.sql("...")
            if not (isinstance(func, ast.Attribute) and func.attr == "sql"):
                continue

            if not node.args:
                continue

            sql_str = self._extract_sql_string(node.args[0])
            if sql_str and SELECT_STAR_PATTERN.search(sql_str):
                findings.append(self._make_finding(
                    filepath=filepath,
                    line=node.lineno,
                    col=node.col_offset,
                    message=(
                        "SQL query uses SELECT *. Specify only needed columns "
                        "to enable column pruning."
                    ),
                    suggestion=(
                        "Columnar formats like Parquet can skip unread columns entirely. "
                        "Replace SELECT * with explicit column names."
                    ),
                ))
        return findings

    def _extract_sql_string(self, node: ast.expr) -> str | None:
        """Extract string content from a constant or f-string."""
        # Simple string: "SELECT * FROM ..."
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value

        # f-string: f"SELECT * FROM {table}"
        if isinstance(node, ast.JoinedStr):
            parts = []
            for value in node.values:
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    parts.append(value.value)
                else:
                    parts.append("___")  # placeholder for expressions
            return "".join(parts)

        # String concatenation: "SELECT * " + "FROM ..."
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = self._extract_sql_string(node.left)
            right = self._extract_sql_string(node.right)
            if left and right:
                return left + right

        # Variable — can't resolve, skip
        return None
