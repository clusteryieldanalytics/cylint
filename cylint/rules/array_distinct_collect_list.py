"""CY033: array_distinct(collect_list(x)) — use collect_set(x) instead."""

import ast
from typing import Optional

from cylint.models import Finding, RuleMeta, Severity
from cylint.rules import BaseRule, register_rule
from cylint.tracker import DataFrameTracker


def _func_name(node: ast.expr) -> Optional[str]:
    """Return the base function name from a Call node's func, ignoring module prefix."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


@register_rule
class CollectListDedupRule(BaseRule):
    META = RuleMeta(
        rule_id="CY033",
        name="collect-list-dedup",
        description="array_distinct(collect_list(x)) — use collect_set(x) instead",
        default_severity=Severity.WARNING,
    )

    def check(self, tree: ast.Module, tracker: DataFrameTracker, filepath: str) -> list[Finding]:
        findings = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            # Outer call must be array_distinct (bare or F.array_distinct)
            if _func_name(node.func) != "array_distinct":
                continue
            # Must have exactly one positional argument
            if not node.args:
                continue
            inner = node.args[0]
            # Inner argument must be a collect_list call
            if not isinstance(inner, ast.Call):
                continue
            if _func_name(inner.func) != "collect_list":
                continue

            findings.append(self._make_finding(
                filepath=filepath,
                line=node.lineno,
                col=node.col_offset,
                message=(
                    "array_distinct(collect_list(x)) performs dedup in a separate step. "
                    "Use collect_set(x) to collect unique values in a single aggregation."
                ),
                suggestion="Replace array_distinct(collect_list(x)) with collect_set(x).",
            ))
        return findings
