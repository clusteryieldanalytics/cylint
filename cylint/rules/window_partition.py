"""CY017: Window function with .orderBy() but no .partitionBy() — full-table sort."""

import ast
from dataclasses import dataclass

from cylint.models import Finding, RuleMeta, Severity
from cylint.rules import BaseRule, register_rule


@dataclass
class _WindowSpecInfo:
    has_order_by: bool
    has_partition_by: bool
    line: int
    col: int


def _is_window_class(node: ast.expr) -> bool:
    """Check if node is the Window class reference (Window or module.Window)."""
    if isinstance(node, ast.Name) and node.id == "Window":
        return True
    if isinstance(node, ast.Attribute) and node.attr == "Window":
        return True
    return False


_WINDOW_CHAIN_METHODS = frozenset({"orderBy", "partitionBy", "rangeBetween", "rowsBetween"})


def _analyze_window_chain(node: ast.expr) -> _WindowSpecInfo | None:
    """Walk a method chain and determine if it's a Window spec with orderBy/partitionBy."""
    has_order_by = False
    has_partition_by = False
    line = getattr(node, "lineno", 0)
    col = getattr(node, "col_offset", 0)

    current = node
    while isinstance(current, ast.Call) and isinstance(current.func, ast.Attribute):
        method = current.func.attr
        if method == "orderBy":
            has_order_by = True
        elif method == "partitionBy":
            has_partition_by = True
        elif method not in _WINDOW_CHAIN_METHODS:
            break

        receiver = current.func.value
        if _is_window_class(receiver):
            if has_order_by:
                return _WindowSpecInfo(
                    has_order_by=True,
                    has_partition_by=has_partition_by,
                    line=line,
                    col=col,
                )
            return None  # partitionBy only — no sort, not relevant
        current = receiver

    return None


@register_rule
class WindowPartitionRule(BaseRule):
    META = RuleMeta(
        rule_id="CY017",
        name="window-no-partition",
        description="Window.orderBy() without .partitionBy() forces a full-table sort into one partition",
        default_severity=Severity.WARNING,
    )

    _MESSAGE = (
        ".over() uses a Window spec with .orderBy() but no .partitionBy(). "
        "This forces a full-table sort into a single partition — "
        "all data processed by one task. "
        "Add .partitionBy(key) to limit sort scope, or add "
        "# cy:ignore CY017 if a global ordering is intentional."
    )

    def check(self, tree: ast.Module, tracker, filepath: str) -> list[Finding]:
        # Phase 1: Collect all Window spec assignments.
        # This runs over the full AST before Phase 2, so if a variable is
        # reassigned with .partitionBy() *after* an .over() use, Phase 1
        # will see it as partitioned and Phase 2 won't fire. This is
        # intentional conservatism — avoids false positives at the cost of
        # missing some true positives in unusual ordering.
        window_specs: dict[str, _WindowSpecInfo] = {}

        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue
                var_name = target.id
                info = _analyze_window_chain(node.value)
                if info is not None:
                    window_specs[var_name] = info
                    continue
                # Handle reassignment: w = w.partitionBy("user_id")
                if (isinstance(node.value, ast.Call)
                        and isinstance(node.value.func, ast.Attribute)
                        and node.value.func.attr == "partitionBy"
                        and isinstance(node.value.func.value, ast.Name)
                        and node.value.func.value.id in window_specs):
                    window_specs[var_name] = _WindowSpecInfo(
                        has_order_by=window_specs[node.value.func.value.id].has_order_by,
                        has_partition_by=True,
                        line=node.lineno,
                        col=node.col_offset,
                    )

        # Phase 2: Find .over() calls using unpartitioned specs
        findings = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "over":
                continue
            if not node.args:
                continue

            arg = node.args[0]

            # Case 1: Inline — F.row_number().over(Window.orderBy("ts"))
            inline_info = _analyze_window_chain(arg)
            if inline_info is not None:
                if inline_info.has_order_by and not inline_info.has_partition_by:
                    findings.append(self._make_finding(
                        filepath=filepath,
                        line=node.lineno,
                        col=node.col_offset,
                        message=self._MESSAGE,
                    ))
                continue

            # Case 2: Variable reference — F.row_number().over(w)
            if isinstance(arg, ast.Name) and arg.id in window_specs:
                info = window_specs[arg.id]
                if info.has_order_by and not info.has_partition_by:
                    findings.append(self._make_finding(
                        filepath=filepath,
                        line=node.lineno,
                        col=node.col_offset,
                        message=self._MESSAGE,
                    ))

        return findings
