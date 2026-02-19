"""CY007: .crossJoin() or implicit cartesian join."""

import ast

from cylint.models import Finding, RuleMeta, Severity
from cylint.rules import BaseRule, register_rule
from cylint.tracker import DataFrameTracker, find_root_name


@register_rule
class CrossJoinRule(BaseRule):
    META = RuleMeta(
        rule_id="CY007",
        name="cross-join",
        description="Cross join or cartesian join detected",
        default_severity=Severity.CRITICAL,
    )

    def check(self, tree: ast.Module, tracker: DataFrameTracker, filepath: str) -> list[Finding]:
        findings = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue

            # Explicit .crossJoin()
            if func.attr == "crossJoin":
                findings.append(self._make_finding(
                    filepath=filepath,
                    line=node.lineno,
                    col=node.col_offset,
                    message=(
                        "Cross join detected. This produces rows_left × rows_right "
                        "output rows. On non-trivial tables, this is almost always "
                        "unintentional."
                    ),
                    suggestion="Add a join condition.",
                ))
                continue

            # Implicit cross join: .join(other) with no `on` parameter
            if func.attr == "join":
                if self._is_missing_join_condition(node):
                    findings.append(self._make_finding(
                        filepath=filepath,
                        line=node.lineno,
                        col=node.col_offset,
                        message=(
                            ".join() called without a join condition. "
                            "This produces a cartesian product."
                        ),
                        suggestion="Add an `on` parameter: .join(other, on='key')",
                    ))

        return findings

    def _is_missing_join_condition(self, call_node: ast.Call) -> bool:
        """Check if a .join() call is missing the join condition (on parameter)."""
        # .join(other) — only 1 positional arg, no 'on' keyword
        has_on_kwarg = any(kw.arg == "on" for kw in call_node.keywords)
        if has_on_kwarg:
            return False

        # If there's a second positional arg, that's the join condition
        if len(call_node.args) >= 2:
            return False

        # Only 1 arg (the other DataFrame), no condition
        return len(call_node.args) == 1
