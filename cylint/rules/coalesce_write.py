"""CY013: .coalesce(1) before .write() — single-executor bottleneck."""

import ast

from cylint.models import Finding, RuleMeta, Severity
from cylint.rules import BaseRule, register_rule
from cylint.tracker import DataFrameTracker, get_chain_methods, find_root_name

WRITE_TERMINALS = frozenset({
    "save", "saveAsTable", "parquet", "csv", "json",
    "orc", "text", "insertInto", "writeTo",
})


@register_rule
class CoalesceWriteRule(BaseRule):
    META = RuleMeta(
        rule_id="CY013",
        name="coalesce-one-write",
        description=".coalesce(1) before write forces all data through one task",
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
            if func.attr not in WRITE_TERMINALS:
                continue

            # Walk the chain looking for .coalesce(1)
            found, coalesce_line = self._has_coalesce_1_in_chain(node)
            if not found:
                continue

            # Try to find the root DataFrame name
            root = find_root_name(node)

            findings.append(self._make_finding(
                filepath=filepath,
                line=coalesce_line or node.lineno,
                col=node.col_offset,
                message=(
                    f".coalesce(1) before write"
                    f"{f' on `{root}`' if root else ''}"
                    " forces all data through a single executor task. "
                    "For large DataFrames this causes OOM. "
                    "Use .repartition(N) for a controlled file count, or set "
                    ".option('maxRecordsPerFile', N) on the writer."
                ),
                suggestion=(
                    "Remove .coalesce(1) or use .repartition(N) with N > 1, "
                    "or .option('maxRecordsPerFile', N)."
                ),
            ))

        return findings

    def _has_coalesce_1_in_chain(self, node: ast.expr) -> tuple[bool, int | None]:
        """Walk a method chain upward. Return (found, line_number)."""
        current = node
        while True:
            if isinstance(current, ast.Call):
                if isinstance(current.func, ast.Attribute):
                    method = current.func.attr
                    if method == "coalesce":
                        if (current.args
                                and isinstance(current.args[0], ast.Constant)
                                and current.args[0].value == 1):
                            return True, current.lineno
                    current = current.func.value
                else:
                    break
            elif isinstance(current, ast.Attribute):
                current = current.value
            else:
                break
        return False, None
