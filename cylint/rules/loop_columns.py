"""CY011: Column transformations in a loop (withColumnRenamed, drop, etc.)."""

import ast

from cylint.models import Finding, RuleMeta, Severity
from cylint.rules import BaseRule, register_rule
from cylint.tracker import DataFrameTracker, find_root_name


# Methods that create per-column plan nodes when called in a loop.
# withColumn is included for completeness but CY003 takes priority on the same line.
LOOP_TRANSFORM_METHODS = frozenset({
    "withColumnRenamed", "drop", "withColumn",
})


@register_rule
class LoopColumnsRule(BaseRule):
    META = RuleMeta(
        rule_id="CY011",
        name="loop-column-transform",
        description="Per-column DataFrame transformation inside a loop",
        default_severity=Severity.WARNING,
    )

    def check(self, tree: ast.Module, tracker: DataFrameTracker, filepath: str) -> list[Finding]:
        # First, collect lines already flagged by CY003 (withColumn-loop)
        # by running CY003's detection logic. We import here to avoid circular deps.
        cy003_lines = self._get_cy003_lines(tree, tracker)

        findings = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.For, ast.While)):
                continue
            findings.extend(self._check_loop(node, tracker, filepath, cy003_lines))
        return findings

    def _check_loop(
        self,
        loop_node: ast.AST,
        tracker: DataFrameTracker,
        filepath: str,
        cy003_lines: set[int],
    ) -> list[Finding]:
        findings = []
        for node in ast.walk(loop_node):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            method = func.attr
            if method not in LOOP_TRANSFORM_METHODS:
                continue

            # Check if on a tracked DataFrame
            root = find_root_name(func.value)
            if root is None or not tracker.is_tracked(root):
                continue

            # Skip if CY003 already flagged this line
            if node.lineno in cy003_lines:
                continue

            findings.append(self._make_finding(
                filepath=filepath,
                line=node.lineno,
                col=node.col_offset,
                message=(
                    f".{method}() inside a for loop creates one plan node per "
                    "iteration. Collect all transformations and apply in a single "
                    ".select() or .withColumnsRenamed() call outside the loop."
                ),
                suggestion=(
                    "Use .select() with all column expressions, or "
                    ".withColumnsRenamed() for renames."
                ),
            ))
            # One finding per loop to avoid noise
            break

        return findings

    def _get_cy003_lines(self, tree: ast.Module, tracker: DataFrameTracker) -> set[int]:
        """Detect lines where CY003 would fire (.withColumn in loop)."""
        lines: set[int] = set()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.For, ast.While)):
                continue
            for inner in ast.walk(node):
                if not isinstance(inner, ast.Call):
                    continue
                func = inner.func
                if not isinstance(func, ast.Attribute):
                    continue
                if func.attr not in ("withColumn", "withColumns"):
                    continue
                root = find_root_name(func.value)
                if root is not None and tracker.is_tracked(root):
                    lines.add(inner.lineno)
        return lines
