"""CY035: .union() / .unionAll() — use .unionByName() to avoid column misalignment."""

import ast

from cylint.models import Finding, RuleMeta, Severity
from cylint.rules import BaseRule, register_rule
from cylint.tracker import DataFrameTracker, find_root_name

_UNION_METHODS = frozenset({"union", "unionAll"})


@register_rule
class UnionByNameRule(BaseRule):
    META = RuleMeta(
        rule_id="CY035",
        name="union-column-order",
        description=".union()/.unionAll() matches by column position — use .unionByName()",
        default_severity=Severity.WARNING,
    )

    def check(self, tree: ast.Module, tracker: DataFrameTracker, filepath: str) -> list[Finding]:
        findings = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            if func.attr not in _UNION_METHODS:
                continue

            # Must be on a tracked DataFrame to avoid false-positives on
            # Python set.union() or other non-Spark union methods
            root = find_root_name(func.value)
            if root is None or not tracker.is_tracked(root):
                continue

            method = func.attr
            findings.append(self._make_finding(
                filepath=filepath,
                line=node.lineno,
                col=node.col_offset,
                message=(
                    f".{method}() matches columns by position, not by name. "
                    "If the two DataFrames have schemas that differ in column order, "
                    "data silently lands in the wrong columns. "
                    "Use .unionByName() to match on column names instead."
                ),
                suggestion=(
                    "Replace with .unionByName(other) or "
                    ".unionByName(other, allowMissingColumns=True) "
                    "if schemas may not be identical."
                ),
            ))
        return findings
