"""CY003: .withColumn() inside a loop."""

import ast

from cylint.models import Finding, RuleMeta, Severity
from cylint.rules import BaseRule, register_rule
from cylint.tracker import DataFrameTracker


@register_rule
class WithColumnLoopRule(BaseRule):
    META = RuleMeta(
        rule_id="CY003",
        name="withcolumn-loop",
        description=".withColumn() inside a loop creates O(n²) plan complexity",
        default_severity=Severity.CRITICAL,
    )

    def check(self, tree: ast.Module, tracker: DataFrameTracker, filepath: str) -> list[Finding]:
        findings = []
        # Walk to find for/while loops
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                findings.extend(self._check_loop(node, tracker, filepath))
        return findings

    def _check_loop(
        self, loop_node: ast.AST, tracker: DataFrameTracker, filepath: str
    ) -> list[Finding]:
        findings = []
        # Look for .withColumn() calls inside the loop body
        for node in ast.walk(loop_node):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            if func.attr not in ("withColumn", "withColumns"):
                continue

            # Check if this is on a tracked DataFrame or any variable
            # (conservative: flag all .withColumn in loops since it's almost
            # always a DataFrame operation)
            # Extra check: see if the result is reassigned to the same variable
            parent = self._find_parent_assign(loop_node, node)
            if parent is not None or self._is_on_tracked(func.value, tracker):
                findings.append(self._make_finding(
                    filepath=filepath,
                    line=node.lineno,
                    col=node.col_offset,
                    message=(
                        ".withColumn() inside a loop creates O(n²) plan complexity. "
                        "Each call creates a new plan node; Catalyst slows exponentially."
                    ),
                    suggestion=(
                        "Use .select([...]) with all column expressions instead. "
                        "Example: df.select([F.col(c).alias(c) for c in cols])"
                    ),
                ))
                # Only one finding per loop to avoid noise
                break

        return findings

    def _is_on_tracked(self, node: ast.expr, tracker: DataFrameTracker) -> bool:
        """Check if the method is called on a tracked DataFrame."""
        if isinstance(node, ast.Name):
            return tracker.is_tracked(node.id)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            return self._is_on_tracked(node.func.value, tracker)
        return False

    def _find_parent_assign(self, loop_node: ast.AST, target_call: ast.Call) -> ast.Assign | None:
        """Check if the withColumn call is part of a reassignment like df = df.withColumn(...)."""
        for node in ast.walk(loop_node):
            if isinstance(node, ast.Assign):
                if self._contains_node(node.value, target_call):
                    return node
        return None

    def _contains_node(self, tree: ast.AST, target: ast.AST) -> bool:
        """Check if target node is contained within tree."""
        if tree is target:
            return True
        for child in ast.iter_child_nodes(tree):
            if self._contains_node(child, target):
                return True
        return False
