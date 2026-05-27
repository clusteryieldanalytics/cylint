"""CY038: .explode() before .filter() in a chain — filter first to reduce rows."""

import ast

from cylint.models import Finding, RuleMeta, Severity
from cylint.rules import BaseRule, register_rule
from cylint.tracker import DataFrameTracker, find_root_name, get_chain_methods

_FILTER_METHODS = frozenset({"filter", "where"})


def _check_chain(node: ast.expr, tracker: DataFrameTracker) -> bool:
    """Return True if node is a call chain on a tracked DF with explode before filter."""
    root = find_root_name(node)
    if root is None or not tracker.is_tracked(root):
        return False
    methods = get_chain_methods(node)
    if "explode" not in methods:
        return False
    explode_idx = methods.index("explode")
    filter_indices = [i for i, m in enumerate(methods) if m in _FILTER_METHODS]
    # Fire only if at least one filter comes AFTER explode
    return any(i > explode_idx for i in filter_indices)


@register_rule
class ExplodeFilterOrderRule(BaseRule):
    META = RuleMeta(
        rule_id="CY038",
        name="explode-before-filter",
        description=".explode() before .filter() multiplies rows before removing them",
        default_severity=Severity.WARNING,
    )

    def check(self, tree: ast.Module, tracker: DataFrameTracker, filepath: str) -> list[Finding]:
        findings = []
        seen_lines: set[int] = set()

        for stmt in ast.walk(tree):
            # Check top-level expressions and assignments to avoid firing on every
            # intermediate call node in the same chain
            if isinstance(stmt, ast.Expr):
                expr = stmt.value
            elif isinstance(stmt, ast.Assign):
                expr = stmt.value
            elif isinstance(stmt, (ast.AugAssign, ast.AnnAssign)):
                expr = stmt.value
            else:
                continue

            if expr is None or not isinstance(expr, ast.Call):
                continue

            if not _check_chain(expr, tracker):
                continue

            # Find the line of the explode() call in the chain to report accurately
            line = self._find_explode_line(expr) or expr.lineno
            if line in seen_lines:
                continue
            seen_lines.add(line)

            findings.append(self._make_finding(
                filepath=filepath,
                line=line,
                col=expr.col_offset,
                message=(
                    ".explode() appears before .filter() in the chain. "
                    "explode() multiplies rows — running it before filter() "
                    "means you are filtering a larger dataset than necessary."
                ),
                suggestion=(
                    "Reorder: apply .filter(...) before .explode(...) "
                    "to reduce the number of rows being exploded."
                ),
            ))
        return findings

    def _find_explode_line(self, node: ast.expr) -> int | None:
        """Walk a chain to find the line number of the .explode() call."""
        current = node
        while isinstance(current, ast.Call) and isinstance(current.func, ast.Attribute):
            if current.func.attr == "explode":
                return current.lineno
            current = current.func.value
        return None
