"""CY006: .toPandas() on unfiltered DataFrame."""

import ast

from cylint.models import Finding, RuleMeta, Severity
from cylint.rules import BaseRule, register_rule
from cylint.tracker import (
    DataFrameTracker,
    chain_has_filter,
    find_root_name,
)


@register_rule
class ToPandasRule(BaseRule):
    META = RuleMeta(
        rule_id="CY006",
        name="topandas-unfiltered",
        description=".toPandas() on unfiltered DataFrame",
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
            if func.attr != "toPandas":
                continue

            root = find_root_name(func.value)
            if root is None:
                continue
            if not tracker.is_tracked(root):
                continue

            # Check chain for filtering
            if chain_has_filter(func.value):
                continue

            # Check if the DataFrame was filtered at definition
            info = tracker.get_info(root)
            if info and info.has_filter:
                continue

            findings.append(self._make_finding(
                filepath=filepath,
                line=node.lineno,
                col=node.col_offset,
                message=(
                    ".toPandas() called on a DataFrame without filtering or aggregation. "
                    "This collects all data to the driver."
                ),
                suggestion=(
                    "Apply .filter(), .limit(), or .agg() first."
                ),
            ))
        return findings
