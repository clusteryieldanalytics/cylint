"""CY007: .crossJoin() or implicit cartesian join."""

import ast

from cylint.models import Finding, RuleMeta, Severity
from cylint.rules import BaseRule, register_rule
from cylint.tracker import DataFrameTracker, find_root_name


def is_missing_join_condition(call_node: ast.Call) -> bool:
    """Check if a .join() call is missing the join condition (on parameter)."""
    has_on_kwarg = any(kw.arg == "on" for kw in call_node.keywords)
    if has_on_kwarg:
        return False

    # If there's a second positional arg, that's the join condition
    if len(call_node.args) >= 2:
        return False

    # Only 1 arg (the other DataFrame), no condition
    return len(call_node.args) == 1


def iter_cy007_nodes(
    tree: ast.Module, tracker: DataFrameTracker
) -> list[tuple[ast.Call, str]]:
    """Yield (call_node, kind) pairs for every CY007 match.

    *kind* is ``"crossJoin"`` or ``"join"``.
    Used by CY007 itself for finding generation and by CY015 for dedup.
    """
    matches: list[tuple[ast.Call, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue

        # crossJoin is PySpark-specific — no tracking needed
        if func.attr == "crossJoin":
            matches.append((node, "crossJoin"))
        elif func.attr == "join":
            # Must be on a tracked DataFrame — .join() is also a common
            # string/list method and would otherwise false-positive heavily.
            root = find_root_name(func.value)
            if root is None or not tracker.is_tracked(root):
                continue
            if is_missing_join_condition(node):
                matches.append((node, "join"))
    return matches


def get_cy007_lines(tree: ast.Module, tracker: DataFrameTracker) -> set[int]:
    """Return line numbers where CY007 would fire.

    Convenience wrapper used by CY015 for dedup.
    """
    return {node.lineno for node, _ in iter_cy007_nodes(tree, tracker)}


@register_rule
class CrossJoinRule(BaseRule):
    META = RuleMeta(
        rule_id="CY007",
        name="cross-join",
        description="Cross join or cartesian join detected",
        default_severity=Severity.CRITICAL,
    )

    def check(self, tree: ast.Module, tracker: DataFrameTracker, filepath: str) -> list[Finding]:
        findings = []
        for node, kind in iter_cy007_nodes(tree, tracker):
            if kind == "crossJoin":
                findings.append(self._make_finding(
                    filepath=filepath,
                    line=node.lineno,
                    col=node.col_offset,
                    message=(
                        "Cross join detected. This produces rows_left × rows_right "
                        "output rows. On non-trivial tables, this is almost always "
                        "unintentional."
                    ),
                    suggestion="Add a join condition.",
                ))
            else:
                findings.append(self._make_finding(
                    filepath=filepath,
                    line=node.lineno,
                    col=node.col_offset,
                    message=(
                        ".join() called without a join condition. "
                        "This produces a cartesian product."
                    ),
                    suggestion="Add an `on` parameter: .join(other, on='key')",
                ))
        return findings
