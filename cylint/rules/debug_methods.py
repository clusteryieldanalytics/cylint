"""CY012: Debug/inspection methods left in production code."""

import ast

from cylint.models import Finding, RuleMeta, Severity
from cylint.rules import BaseRule, register_rule
from cylint.tracker import DataFrameTracker, find_root_name

# Method name → severity level
DEBUG_METHODS = {
    "show": Severity.WARNING,
    "display": Severity.WARNING,
    "printSchema": Severity.WARNING,
    "explain": Severity.INFO,
}

# Human-readable messages per method
DEBUG_MESSAGES = {
    "show": ".show() on `{df}` triggers data collection. Remove from production code.",
    "display": ".display() on `{df}` triggers data rendering. Remove from production code.",
    "printSchema": ".printSchema() on `{df}` is a diagnostic call. Remove from production code.",
    "explain": ".explain() on `{df}` is a diagnostic call. Remove from production code.",
}


@register_rule
class DebugMethodsRule(BaseRule):
    META = RuleMeta(
        rule_id="CY012",
        name="debug-methods",
        description="Debug/inspection method left in production code",
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

            method = func.attr
            if method not in DEBUG_METHODS:
                continue

            # Must be on a tracked DataFrame
            root = find_root_name(func.value)
            if root is None or not tracker.is_tracked(root):
                continue

            sev = DEBUG_METHODS[method]
            msg = DEBUG_MESSAGES[method].format(df=root)

            findings.append(Finding(
                rule_id=self.META.rule_id,
                severity=sev,
                message=msg,
                filepath=filepath,
                line=node.lineno,
                col=node.col_offset,
                suggestion="Remove or guard behind a debug flag.",
            ))

        return findings
