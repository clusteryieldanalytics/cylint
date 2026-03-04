"""CY015: Non-equi join condition producing implicit cartesian product."""

import ast

from cylint.models import Finding, RuleMeta, Severity
from cylint.rules import BaseRule, register_rule
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
        cy007_lines = self._get_cy007_lines(tree)

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

            # Need at least 2 args (other_df, condition)
            if len(node.args) < 2:
                continue

            condition = node.args[1]

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

    def _get_cy007_lines(self, tree: ast.Module) -> set[int]:
        """Detect lines where CY007 would fire (crossJoin or join without condition)."""
        lines: set[int] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            if func.attr == "crossJoin":
                lines.add(node.lineno)
            elif func.attr == "join":
                # .join(other) with no condition
                has_on = any(kw.arg == "on" for kw in node.keywords)
                if not has_on and len(node.args) < 2:
                    lines.add(node.lineno)
        return lines
