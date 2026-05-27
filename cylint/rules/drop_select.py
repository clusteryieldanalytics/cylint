"""CY032: .drop() followed by .select() — the drop is redundant."""

import ast

from cylint.models import Finding, RuleMeta, Severity
from cylint.rules import BaseRule, register_rule
from cylint.tracker import DataFrameTracker, find_root_name


@register_rule
class DropSelectRule(BaseRule):
    META = RuleMeta(
        rule_id="CY032",
        name="drop-before-select",
        description=".drop() before .select() is redundant — select already picks the columns",
        default_severity=Severity.INFO,
    )

    def check(self, tree: ast.Module, tracker: DataFrameTracker, filepath: str) -> list[Finding]:
        findings = []
        for node in ast.walk(tree):
            # Look for the terminal .select(...) call
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            if func.attr != "select":
                continue

            # The receiver of .select() must itself be a .drop(...) call
            receiver = func.value
            if not isinstance(receiver, ast.Call):
                continue
            recv_func = receiver.func
            if not isinstance(recv_func, ast.Attribute):
                continue
            if recv_func.attr != "drop":
                continue

            # The root of the whole chain must be a tracked DataFrame
            root = find_root_name(recv_func.value)
            if root is None or not tracker.is_tracked(root):
                continue

            findings.append(self._make_finding(
                filepath=filepath,
                line=receiver.lineno,
                col=receiver.col_offset,
                message=(
                    ".drop() followed by .select() is redundant. "
                    ".select() already picks exactly the columns you want, "
                    "so the preceding .drop() is a no-op."
                ),
                suggestion="Remove the .drop() call and rely solely on .select().",
            ))
        return findings
