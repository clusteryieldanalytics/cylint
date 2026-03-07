"""CY031: for row in df.collect() — driver-side row iteration."""

import ast

from cylint.models import Finding, RuleMeta, Severity
from cylint.rules import BaseRule, register_rule
from cylint.tracker import DataFrameTracker, find_root_name

COLLECT_METHODS = frozenset({"collect", "toLocalIterator"})

_MSG_COLLECT = (
    "Iterating over df.collect() processes all data row-by-row on the "
    "driver in Python. This defeats Spark's distributed execution model. "
    "Rewrite as DataFrame transformations (.select(), .withColumn(), "
    ".groupBy(), etc.) to stay distributed."
)

_MSG_TO_LOCAL_ITERATOR = (
    "Iterating over df.toLocalIterator() processes data row-by-row on "
    "the driver. While memory-efficient (one partition at a time), this "
    "is still single-threaded Python. Rewrite as DataFrame transformations "
    "to leverage distributed compute."
)

_MSG_TO_PANDAS_ITERROWS = (
    "Iterating over df.toPandas().iterrows() converts to pandas then "
    "iterates row-by-row. This is double waste: the full table is pulled "
    "to the driver AND iterated in Python. Rewrite as DataFrame "
    "transformations or use vectorized pandas UDFs if Python logic is needed."
)


def _is_collect_on_tracked(node: ast.expr, tracker: DataFrameTracker) -> str | None:
    """Check if node is df.collect() or df.toLocalIterator() on a tracked DF.

    Returns the method name if matched, None otherwise.
    """
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if not isinstance(func, ast.Attribute):
        return None
    if func.attr not in COLLECT_METHODS:
        return None
    root = find_root_name(func.value)
    if root is None:
        return None
    if not tracker.is_tracked(root):
        return None
    return func.attr


def _is_topandas_iterrows(node: ast.expr, tracker: DataFrameTracker) -> bool:
    """Check if node is df.toPandas().iterrows() on a tracked DF."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if not isinstance(func, ast.Attribute):
        return False
    if func.attr not in ("iterrows", "itertuples", "items"):
        return False
    # The receiver should be df.toPandas()
    inner = func.value
    if not isinstance(inner, ast.Call):
        return False
    if not isinstance(inner.func, ast.Attribute):
        return False
    if inner.func.attr != "toPandas":
        return False
    root = find_root_name(inner.func.value)
    if root is None:
        return False
    return tracker.is_tracked(root)


def _is_topandas_on_tracked(node: ast.expr, tracker: DataFrameTracker) -> bool:
    """Check if node is df.toPandas() on a tracked DF."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if not isinstance(func, ast.Attribute):
        return False
    if func.attr != "toPandas":
        return False
    root = find_root_name(func.value)
    if root is None:
        return False
    return tracker.is_tracked(root)


def _get_message(method: str) -> str:
    if method == "toLocalIterator":
        return _MSG_TO_LOCAL_ITERATOR
    return _MSG_COLLECT


@register_rule
class CollectIterationRule(BaseRule):
    META = RuleMeta(
        rule_id="CY031",
        name="collect-iteration",
        description="for row in df.collect() — driver-side row iteration defeats Spark",
        default_severity=Severity.WARNING,
    )

    def check(self, tree: ast.Module, tracker: DataFrameTracker, filepath: str) -> list[Finding]:
        findings = []

        # Track collect/toLocalIterator assignments: var_name -> (line, method)
        collect_assigns: dict[str, tuple[int, str]] = {}
        # Track toPandas() assignments: var_name -> line
        topandas_assigns: dict[str, int] = {}

        for node in ast.walk(tree):
            # Record collect assignments: rows = df.collect()
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name):
                    method = _is_collect_on_tracked(node.value, tracker)
                    if method is not None:
                        collect_assigns[target.id] = (node.lineno, method)
                    # Record toPandas assignments: pdf = df.toPandas()
                    elif _is_topandas_on_tracked(node.value, tracker):
                        topandas_assigns[target.id] = node.lineno

            # --- For loops ---
            if isinstance(node, ast.For):
                # Direct collect in for-loop
                method = _is_collect_on_tracked(node.iter, tracker)
                if method is not None:
                    findings.append(self._make_finding(
                        filepath=filepath,
                        line=node.lineno,
                        col=node.col_offset,
                        message=_get_message(method),
                    ))
                    continue

                # toPandas().iterrows() in for-loop
                if _is_topandas_iterrows(node.iter, tracker):
                    findings.append(self._make_finding(
                        filepath=filepath,
                        line=node.lineno,
                        col=node.col_offset,
                        message=_MSG_TO_PANDAS_ITERROWS,
                    ))
                    continue

                # Assigned-then-iterated: rows = df.collect(); for row in rows:
                if isinstance(node.iter, ast.Name) and node.iter.id in collect_assigns:
                    _, method = collect_assigns[node.iter.id]
                    findings.append(self._make_finding(
                        filepath=filepath,
                        line=node.lineno,
                        col=node.col_offset,
                        message=_get_message(method),
                    ))
                    continue

                # toPandas assigned then iterated: pdf = df.toPandas(); for r in pdf.iterrows():
                if (isinstance(node.iter, ast.Call)
                        and isinstance(node.iter.func, ast.Attribute)
                        and node.iter.func.attr in ("iterrows", "itertuples", "items")
                        and isinstance(node.iter.func.value, ast.Name)
                        and node.iter.func.value.id in topandas_assigns):
                    findings.append(self._make_finding(
                        filepath=filepath,
                        line=node.lineno,
                        col=node.col_offset,
                        message=_MSG_TO_PANDAS_ITERROWS,
                    ))
                    continue

            # --- List/Set/Dict/Generator comprehensions ---
            if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
                for gen in node.generators:
                    method = _is_collect_on_tracked(gen.iter, tracker)
                    if method is not None:
                        findings.append(self._make_finding(
                            filepath=filepath,
                            line=node.lineno,
                            col=node.col_offset,
                            message=_get_message(method),
                        ))
                        break

                    if isinstance(gen.iter, ast.Name) and gen.iter.id in collect_assigns:
                        _, method = collect_assigns[gen.iter.id]
                        findings.append(self._make_finding(
                            filepath=filepath,
                            line=node.lineno,
                            col=node.col_offset,
                            message=_get_message(method),
                        ))
                        break

        return findings
