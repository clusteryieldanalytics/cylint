"""CY010: .join() without explicit `how=` type."""

import ast

from cylint.models import Finding, RuleMeta, Severity
from cylint.rules import BaseRule, register_rule
from cylint.tracker import DataFrameTracker, find_root_name


@register_rule
class JoinTypeRule(BaseRule):
    META = RuleMeta(
        rule_id="CY010",
        name="join-missing-how",
        description=".join() without explicit how= argument",
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
            if func.attr != "join":
                continue

            # Must be on a tracked DataFrame
            root = find_root_name(func.value)
            if root is None or not tracker.is_tracked(root):
                continue

            # Check for how= keyword
            kw_names = {kw.arg for kw in node.keywords}
            if "how" in kw_names:
                continue

            # Third positional arg is `how` by PySpark convention — do not flag
            if len(node.args) >= 3:
                continue

            findings.append(self._make_finding(
                filepath=filepath,
                line=node.lineno,
                col=node.col_offset,
                message=(
                    f".join() on `{root}` has no explicit `how` argument. "
                    "Defaults to inner join. Specify how='inner' (or left/right/outer) "
                    "to make intent explicit and prevent silent behaviour changes."
                ),
                suggestion="Add how='inner' (or the intended join type).",
            ))

        return findings
