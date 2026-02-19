"""CY001: .collect() called on unfiltered DataFrame."""

import ast

from cylint.models import Finding, RuleMeta, Severity
from cylint.rules import BaseRule, register_rule
from cylint.tracker import (
    DataFrameTracker,
    chain_has_filter,
    find_root_name,
)


@register_rule
class CollectRule(BaseRule):
    META = RuleMeta(
        rule_id="CY001",
        name="collect-unfiltered",
        description=".collect() called without filtering or limiting",
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
            if func.attr != "collect":
                continue

            # Check if this is on a tracked DataFrame
            root = find_root_name(func.value)
            if root is None:
                continue
            if not tracker.is_tracked(root):
                continue

            # Check if the chain has filtering
            if chain_has_filter(func.value):
                continue

            # Also check if the tracked DataFrame itself was filtered at definition
            info = tracker.get_info(root)
            if info and info.has_filter:
                continue

            findings.append(self._make_finding(
                filepath=filepath,
                line=node.lineno,
                col=node.col_offset,
                message=(
                    ".collect() called without filtering. "
                    "This pulls all data to the driver and can cause OOM."
                ),
                suggestion=(
                    "Consider .limit(N).collect(), .take(N), "
                    "or using .show() for inspection."
                ),
            ))
        return findings
