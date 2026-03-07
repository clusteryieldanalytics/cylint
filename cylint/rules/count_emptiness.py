"""CY020: .count() used only as an emptiness check."""

import ast

from cylint.models import Finding, RuleMeta, Severity
from cylint.rules import BaseRule, register_rule
from cylint.tracker import DataFrameTracker, find_root_name


def _is_zero(node: ast.expr) -> bool:
    """Check if a node is the literal 0."""
    return isinstance(node, ast.Constant) and node.value == 0


def _is_one(node: ast.expr) -> bool:
    """Check if a node is the literal 1."""
    return isinstance(node, ast.Constant) and node.value == 1


def _is_emptiness_compare(left: ast.expr, ops: list, comparators: list) -> bool:
    """Check if a Compare node is an emptiness check against 0 or 1.

    Matches: count > 0, count == 0, count != 0, count >= 1, count < 1,
             0 < count, 0 == count, etc.
    """
    if len(ops) != 1 or len(comparators) != 1:
        return False
    op = ops[0]
    comp = comparators[0]

    # count > 0  → "is non-empty"
    # count == 0 → "is empty"
    # count != 0 → "is non-empty"
    # count <= 0 → "is empty" (for non-negative counts)
    # Exclude: count >= 0 (tautology), count < 0 (contradiction)
    if _is_zero(comp) and isinstance(op, (ast.Gt, ast.Eq, ast.NotEq, ast.LtE)):
        return True
    # count >= 1, count < 1
    if _is_one(comp) and isinstance(op, (ast.GtE, ast.Lt)):
        return True

    return False


def _is_emptiness_compare_reversed(left: ast.expr, ops: list, comparators: list) -> bool:
    """Check reversed comparison: 0 < count, 0 == count, etc."""
    if len(ops) != 1 or len(comparators) != 1:
        return False
    op = ops[0]

    # 0 < count  → "is non-empty"
    # 0 == count → "is empty"
    # 0 != count → "is non-empty"
    # 0 >= count → "is empty" (for non-negative counts)
    # Exclude: 0 <= count (tautology), 0 > count (contradiction)
    if _is_zero(left) and isinstance(op, (ast.Lt, ast.Eq, ast.NotEq, ast.GtE)):
        return True
    # 1 <= count, 1 > count
    if _is_one(left) and isinstance(op, (ast.LtE, ast.Gt)):
        return True

    return False


def _is_count_call_on_tracked(node: ast.expr, tracker: DataFrameTracker) -> bool:
    """Check if node is df.count() where df is a tracked DataFrame."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if not isinstance(func, ast.Attribute):
        return False
    if func.attr != "count":
        return False
    root = find_root_name(func.value)
    if root is None:
        return False
    return tracker.is_tracked(root)


_MESSAGE = (
    ".count() used only to check emptiness — this materializes the "
    "entire DataFrame lineage. Use `df.head(1)` or `len(df.head(1)) > 0` "
    "instead, which stops after one row."
)


@register_rule
class CountEmptinessRule(BaseRule):
    META = RuleMeta(
        rule_id="CY020",
        name="count-emptiness",
        description=".count() compared against 0 for emptiness check — full scan wasted",
        default_severity=Severity.WARNING,
    )

    def check(self, tree: ast.Module, tracker: DataFrameTracker, filepath: str) -> list[Finding]:
        findings = []

        # Track assigned count variables: var_name -> (line, is_only_emptiness)
        count_assigns: dict[str, tuple[int, str]] = {}  # var -> (line, df_root)

        for node in ast.walk(tree):
            # --- Pattern 1: Inline comparison ---
            # if df.count() > 0:
            if isinstance(node, ast.Compare):
                left = node.left
                if _is_count_call_on_tracked(left, tracker):
                    if _is_emptiness_compare(left, node.ops, node.comparators):
                        findings.append(self._make_finding(
                            filepath=filepath,
                            line=node.lineno,
                            col=node.col_offset,
                            message=_MESSAGE,
                        ))
                        continue

                # Reversed: 0 < df.count()
                if len(node.comparators) == 1 and _is_count_call_on_tracked(node.comparators[0], tracker):
                    if _is_emptiness_compare_reversed(left, node.ops, node.comparators):
                        findings.append(self._make_finding(
                            filepath=filepath,
                            line=node.lineno,
                            col=node.col_offset,
                            message=_MESSAGE,
                        ))
                        continue

            # --- Pattern 2: Truthy check ---
            # if not df.count():
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
                if _is_count_call_on_tracked(node.operand, tracker):
                    findings.append(self._make_finding(
                        filepath=filepath,
                        line=node.lineno,
                        col=node.col_offset,
                        message=_MESSAGE,
                    ))
                    continue

            # --- Record count assignments for Pattern 3 ---
            # n = df.count()
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and _is_count_call_on_tracked(node.value, tracker):
                    root = find_root_name(node.value.func.value)
                    if root:
                        count_assigns[target.id] = (node.lineno, root)

        # --- Pattern 3: Assigned-then-compared ---
        if count_assigns:
            self._check_assigned_counts(tree, count_assigns, tracker, filepath, findings)

        return findings

    def _check_assigned_counts(
        self,
        tree: ast.Module,
        count_assigns: dict[str, tuple[int, str]],
        tracker: DataFrameTracker,
        filepath: str,
        findings: list[Finding],
    ):
        """Check if assigned count variables are used only for emptiness checks."""
        # Collect assignment target lines to skip Name nodes that are targets
        assign_lines: dict[str, int] = {v: line for v, (line, _) in count_assigns.items()}

        # For each assigned count var, check ALL uses in the AST
        var_uses: dict[str, list[str]] = {v: [] for v in count_assigns}

        # Collect Name node IDs to exclude: assignment targets + names inside
        # Compare/UnaryOp nodes that we already handle structurally
        handled_name_ids: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        handled_name_ids.add(id(target))
            # Names inside Compare nodes are handled by the Compare branch
            if isinstance(node, ast.Compare):
                if isinstance(node.left, ast.Name) and node.left.id in var_uses:
                    handled_name_ids.add(id(node.left))
                for comp in node.comparators:
                    if isinstance(comp, ast.Name) and comp.id in var_uses:
                        handled_name_ids.add(id(comp))
            # Names inside UnaryOp(Not) are handled by the UnaryOp branch
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
                if isinstance(node.operand, ast.Name) and node.operand.id in var_uses:
                    handled_name_ids.add(id(node.operand))

        for node in ast.walk(tree):
            # Compare: n > 0, n == 0, etc.
            if isinstance(node, ast.Compare):
                if isinstance(node.left, ast.Name) and node.left.id in var_uses:
                    if _is_emptiness_compare(node.left, node.ops, node.comparators):
                        var_uses[node.left.id].append("emptiness")
                        continue
                    var_uses[node.left.id].append("other")
                    continue
                if len(node.comparators) == 1 and isinstance(node.comparators[0], ast.Name):
                    name = node.comparators[0].id
                    if name in var_uses:
                        if _is_emptiness_compare_reversed(node.left, node.ops, node.comparators):
                            var_uses[name].append("emptiness")
                            continue
                        var_uses[name].append("other")
                        continue

            # not n
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
                if isinstance(node.operand, ast.Name) and node.operand.id in var_uses:
                    var_uses[node.operand.id].append("emptiness")
                    continue

            # Any other reference to the variable is a non-emptiness use
            if isinstance(node, ast.Name) and node.id in var_uses:
                if id(node) in handled_name_ids:
                    continue
                var_uses[node.id].append("other")

        # Fire only if ALL uses are emptiness checks
        for var_name, uses in var_uses.items():
            if not uses:
                continue
            if all(u == "emptiness" for u in uses):
                line = count_assigns[var_name][0]
                findings.append(self._make_finding(
                    filepath=filepath,
                    line=line,
                    col=0,
                    message=_MESSAGE,
                ))
