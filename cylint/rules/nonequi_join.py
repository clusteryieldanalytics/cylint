"""CY015: Non-equi join condition producing implicit cartesian product."""

import ast

from cylint.models import Finding, RuleMeta, Severity
from cylint.rules import BaseRule, register_rule
from cylint.rules.crossjoin import get_cy007_lines
from cylint.tracker import DataFrameTracker, find_root_name


@register_rule
class NonEquiJoinRule(BaseRule):
    META = RuleMeta(
        rule_id="CY015",
        name="nonequi-join",
        description="Non-equi join condition produces O(n*m) row comparisons",
        default_severity=Severity.CRITICAL,
    )

    def check(self, tree: ast.Module, tracker: DataFrameTracker, filepath: str) -> list[Finding]:
        # Collect lines flagged by CY007 to avoid double-reporting
        cy007_lines = get_cy007_lines(tree, tracker)

        findings = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            if func.attr != "join":
                continue

            # Skip lines already covered by CY007
            if node.lineno in cy007_lines:
                continue

            # Extract condition: positional arg[1] or keyword on=
            condition = None
            if len(node.args) >= 2:
                condition = node.args[1]
            else:
                for kw in node.keywords:
                    if kw.arg == "on":
                        condition = kw.value
                        break
            if condition is None:
                continue

            # String or list condition is always equi — skip
            if isinstance(condition, (ast.Constant, ast.List)):
                if isinstance(condition, ast.Constant) and isinstance(condition.value, str):
                    continue
                if isinstance(condition, ast.List):
                    continue

            if self._is_nonequi_condition(condition):
                root = find_root_name(func.value)
                if root is None or not tracker.is_tracked(root):
                    continue

                findings.append(self._make_finding(
                    filepath=filepath,
                    line=node.lineno,
                    col=node.col_offset,
                    message=(
                        f".join() on `{root}` uses a non-equi condition. "
                        "This produces O(n*m) row comparisons — equivalent to a cross join. "
                        "Ensure this is intentional. If a range join is required, consider "
                        "a bucketed or sorted merge approach instead."
                    ),
                    suggestion=(
                        "Use an equi-join condition (==) or confirm this "
                        "cartesian-like join is intentional."
                    ),
                ))

        return findings

    def _is_nonequi_condition(self, condition_node: ast.expr) -> bool:
        """Return True if the join condition is non-equi."""
        # Boolean literal: True
        if isinstance(condition_node, ast.Constant) and condition_node.value is True:
            return True

        # F.lit(True)
        if self._is_lit_true(condition_node):
            return True

        # Walk the expression tree looking for any == comparison
        for node in ast.walk(condition_node):
            if isinstance(node, ast.Compare):
                if any(isinstance(op, ast.Eq) for op in node.ops):
                    return False  # found equi — not a non-equi join

        # No == found — this is non-equi
        return True

    def _is_lit_true(self, node: ast.expr) -> bool:
        """Detect F.lit(True) or pyspark.sql.functions.lit(True)."""
        if not isinstance(node, ast.Call):
            return False
        func = node.func
        name = None
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        if name != "lit":
            return False
        return (
            bool(node.args)
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value is True
        )

