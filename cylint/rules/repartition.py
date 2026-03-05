"""CY008: .repartition() immediately before .write()."""

import ast

from cylint.models import Finding, RuleMeta, Severity
from cylint.rules import BaseRule, register_rule
from cylint.tracker import DataFrameTracker

# Write methods that indicate the end of a write chain
WRITE_METHODS = frozenset({
    "parquet", "csv", "json", "orc", "text", "save", "saveAsTable",
    "insertInto", "jdbc",
})


def _extract_repartition_info(call_node: ast.Call) -> tuple[bool, set[str]]:
    """Extract argument types from a .repartition() call.

    Returns (has_count, column_names).
    """
    has_count = False
    column_names: set[str] = set()

    for arg in call_node.args:
        if isinstance(arg, ast.Constant):
            if isinstance(arg.value, int):
                has_count = True
            elif isinstance(arg.value, str):
                column_names.add(arg.value)
        # F.col("name") or col("name")
        elif isinstance(arg, ast.Call) and isinstance(arg.func, (ast.Name, ast.Attribute)):
            func_name = arg.func.id if isinstance(arg.func, ast.Name) else arg.func.attr
            if func_name == "col" and arg.args and isinstance(arg.args[0], ast.Constant):
                column_names.add(arg.args[0].value)

    return has_count, column_names


def _extract_partition_by_columns(call_node: ast.Call) -> set[str]:
    """Extract string column names from a .partitionBy() call."""
    columns: set[str] = set()
    for arg in call_node.args:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            columns.add(arg.value)
    return columns


def _walk_chain(node: ast.expr) -> list[tuple[str, ast.AST]]:
    """Walk a method chain backward, returning (method_name, ast_node) in execution order.

    For Call nodes, the ast_node is the Call. For bare Attributes (e.g. .write),
    the ast_node is the Attribute.
    """
    items: list[tuple[str, ast.AST]] = []
    current = node
    while True:
        if isinstance(current, ast.Call):
            if isinstance(current.func, ast.Attribute):
                items.append((current.func.attr, current))
                current = current.func.value
            else:
                break
        elif isinstance(current, ast.Attribute):
            items.append((current.attr, current))
            current = current.value
        else:
            break
    items.reverse()
    return items


@register_rule
class RepartitionWriteRule(BaseRule):
    META = RuleMeta(
        rule_id="CY008",
        name="repartition-before-write",
        description=".repartition() immediately before .write() triggers unnecessary shuffle",
        default_severity=Severity.INFO,
    )

    def check(self, tree: ast.Module, tracker: DataFrameTracker, filepath: str) -> list[Finding]:
        findings = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue

            # Walk the full chain, preserving AST nodes
            chain = _walk_chain(node)
            method_names = [name for name, _ in chain]

            # Look for pattern: ...repartition(...)...write...parquet/csv/etc
            repartition_idx = None
            write_idx = None

            for i, name in enumerate(method_names):
                if name == "repartition":
                    repartition_idx = i
                if name == "write":
                    write_idx = i
                if name in WRITE_METHODS and write_idx is not None:
                    if repartition_idx is None:
                        break

                    # Skip if a transform sits between .repartition() and .write —
                    # the repartition may serve the intermediate operation (e.g. a
                    # groupBy or filter), not the write itself.
                    between = method_names[repartition_idx + 1:write_idx]
                    if between:
                        break

                    # Found repartition before write — check suppression.
                    # .repartition() is always called with parens so this is
                    # always an ast.Call; guard defensively against bare attribute.
                    repart_node = chain[repartition_idx][1]
                    if not isinstance(repart_node, ast.Call):
                        break

                    has_count, repart_cols = _extract_repartition_info(repart_node)

                    # If repartition has an integer count, always flag
                    if not has_count and repart_cols:
                        # Collect partitionBy columns from the write chain
                        partition_by_cols: set[str] = set()
                        for j in range(repartition_idx + 1, len(chain)):
                            cname, cnode = chain[j]
                            if cname == "partitionBy" and isinstance(cnode, ast.Call):
                                partition_by_cols = _extract_partition_by_columns(cnode)
                                break

                        # Suppress if repartition columns are a subset of partitionBy
                        if partition_by_cols and repart_cols.issubset(partition_by_cols):
                            break

                    findings.append(self._make_finding(
                        filepath=filepath,
                        line=repart_node.lineno,
                        col=repart_node.col_offset,
                        message=(
                            ".repartition() before write triggers a full shuffle."
                        ),
                        suggestion=(
                            "For fewer output files, consider .coalesce(N) "
                            "(avoids shuffle) or .option('maxRecordsPerFile', N)."
                        ),
                    ))
                    break  # one finding per chain

        return findings
