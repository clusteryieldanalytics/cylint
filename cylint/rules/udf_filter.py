"""CY009: UDF in .filter()/.where() blocks predicate pushdown."""

import ast

from cylint.models import Finding, RuleMeta, Severity
from cylint.rules import BaseRule, register_rule
from cylint.tracker import DataFrameTracker, find_root_name


@register_rule
class UdfFilterRule(BaseRule):
    META = RuleMeta(
        rule_id="CY009",
        name="udf-filter-pushdown",
        description="UDF in .filter()/.where() blocks predicate pushdown",
        default_severity=Severity.CRITICAL,
    )

    def check(self, tree: ast.Module, tracker: DataFrameTracker, filepath: str) -> list[Finding]:
        udf_names = _collect_udf_names(tree)
        findings = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            if func.attr not in ("filter", "where"):
                continue

            # Must be on a tracked DataFrame
            root = find_root_name(func.value)
            if root is None or not tracker.is_tracked(root):
                continue

            # Check the condition argument
            if not node.args:
                continue
            condition = node.args[0]
            if _is_udf_call(condition, udf_names):
                findings.append(self._make_finding(
                    filepath=filepath,
                    line=node.lineno,
                    col=node.col_offset,
                    message=(
                        f".filter() / .where() on `{root}` uses a UDF as the condition. "
                        "This prevents predicate pushdown to the scan layer. "
                        "Spark will read the entire table before applying the filter. "
                        "Rewrite using built-in PySpark functions to enable pushdown."
                    ),
                    suggestion=(
                        "Replace the UDF with built-in functions "
                        "(e.g., F.col(), F.when(), F.regexp_extract())."
                    ),
                ))

        return findings


def _collect_udf_names(tree: ast.Module) -> set[str]:
    """Walk the AST to find all variable/function names registered as UDFs."""
    udf_names: set[str] = set()

    for node in ast.walk(tree):
        # Pattern: my_udf = udf(...) or my_udf = pandas_udf(...)
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and isinstance(node.value, ast.Call):
                    _register_udf_if_any(target.id, node.value, udf_names)

        # Pattern: @udf or @pandas_udf decorator on a function def
        if isinstance(node, ast.FunctionDef):
            for dec in node.decorator_list:
                if isinstance(dec, ast.Name) and dec.id in ("udf", "pandas_udf"):
                    udf_names.add(node.name)
                elif isinstance(dec, ast.Attribute) and dec.attr in ("udf", "pandas_udf"):
                    udf_names.add(node.name)
                elif isinstance(dec, ast.Call):
                    # @udf("boolean") or @pandas_udf("long")
                    inner = dec.func
                    if isinstance(inner, ast.Name) and inner.id in ("udf", "pandas_udf"):
                        udf_names.add(node.name)
                    elif isinstance(inner, ast.Attribute) and inner.attr in ("udf", "pandas_udf"):
                        udf_names.add(node.name)

    return udf_names


def _register_udf_if_any(name: str, value_node: ast.Call, udf_names: set[str]):
    """Add name to udf_names if the RHS looks like a UDF definition."""
    func = value_node.func
    # Direct call: udf(...) or pandas_udf(...)
    if isinstance(func, ast.Name) and func.id in ("udf", "pandas_udf"):
        udf_names.add(name)
        return
    # Attribute call: F.udf(...) or functions.udf(...)
    if isinstance(func, ast.Attribute) and func.attr in ("udf", "pandas_udf"):
        udf_names.add(name)
        return


def _is_udf_call(node: ast.expr, udf_names: set[str]) -> bool:
    """True if node is a UDF invocation or lambda."""
    # Lambda in filter position
    if isinstance(node, ast.Lambda):
        return True

    if not isinstance(node, ast.Call):
        return False

    func = node.func

    # Inline: udf(...)(...) — nested Call where inner is udf/pandas_udf
    if isinstance(func, ast.Call):
        inner = func.func
        if isinstance(inner, ast.Name) and inner.id in ("udf", "pandas_udf"):
            return True
        if isinstance(inner, ast.Attribute) and inner.attr in ("udf", "pandas_udf"):
            return True

    # Known UDF variable: my_udf(col("x"))
    if isinstance(func, ast.Name) and func.id in udf_names:
        return True

    return False
