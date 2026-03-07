"""CY018: Schema inference on CSV/JSON read — double file scan."""

import ast

from cylint.models import Finding, RuleMeta, Severity
from cylint.rules import BaseRule, register_rule

INFERENCE_FORMATS = frozenset({"csv", "json"})
SAFE_FORMATS = frozenset({"parquet", "orc", "delta", "avro"})


def _get_read_chain_methods(node: ast.expr) -> list[tuple[str, ast.Call]]:
    """Walk a method chain collecting (method_name, call_node) pairs.

    Returns methods in outermost-first order.
    """
    methods: list[tuple[str, ast.Call]] = []
    current = node
    while isinstance(current, ast.Call) and isinstance(current.func, ast.Attribute):
        methods.append((current.func.attr, current))
        current = current.func.value
    methods.reverse()
    return methods


def _is_spark_read_root(node: ast.expr) -> bool:
    """Check if node is spark.read or similar."""
    if isinstance(node, ast.Attribute) and node.attr == "read":
        if isinstance(node.value, ast.Name) and node.value.id in ("spark", "sqlContext"):
            return True
    return False


def _find_read_root(node: ast.expr) -> bool:
    """Walk a chain to see if it's rooted at spark.read."""
    current = node
    while True:
        if isinstance(current, ast.Call):
            if isinstance(current.func, ast.Attribute):
                current = current.func.value
            else:
                return False
        elif isinstance(current, ast.Attribute):
            if _is_spark_read_root(current):
                return True
            current = current.value
        elif isinstance(current, ast.Name):
            return False
        else:
            return False


def _resolve_format(methods: list[tuple[str, ast.Call]]) -> str | None:
    """Determine the read format from the method chain."""
    for method_name, call_node in methods:
        # Direct format method: .csv(), .json(), .parquet(), etc.
        if method_name in ("csv", "json", "parquet", "orc", "avro", "text"):
            return method_name
        # .format("csv").load() pattern
        if method_name == "format" and call_node.args:
            arg = call_node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                return arg.value.lower()
    return None


def _has_schema_call(methods: list[tuple[str, ast.Call]]) -> bool:
    """Check if .schema() is present in the chain."""
    return any(m == "schema" for m, _ in methods)


def _has_infer_schema_false(methods: list[tuple[str, ast.Call]]) -> bool:
    """Check if .option("inferSchema", "false"/False) is in the chain."""
    for method_name, call_node in methods:
        if method_name == "option" and len(call_node.args) >= 2:
            key_arg = call_node.args[0]
            val_arg = call_node.args[1]
            if isinstance(key_arg, ast.Constant) and key_arg.value == "inferSchema":
                if isinstance(val_arg, ast.Constant):
                    if val_arg.value in ("false", False):
                        return True
    return False


def _has_infer_schema_true(methods: list[tuple[str, ast.Call]]) -> bool:
    """Check if .option("inferSchema", "true"/True) is in the chain."""
    for method_name, call_node in methods:
        if method_name == "option" and len(call_node.args) >= 2:
            key_arg = call_node.args[0]
            val_arg = call_node.args[1]
            if isinstance(key_arg, ast.Constant) and key_arg.value == "inferSchema":
                if isinstance(val_arg, ast.Constant):
                    if val_arg.value in ("true", True):
                        return True
    return False


_MSG_IMPLICIT = (
    "spark.read.{fmt}() without .schema() — Spark will infer types by "
    "reading the entire file twice: once for schema inference, once for "
    "data. Specify an explicit StructType schema to eliminate the "
    "inference pass."
)

_MSG_EXPLICIT = (
    '.option("inferSchema", "true") forces a full extra read of the file '
    "to determine column types. Replace with an explicit StructType schema "
    "to eliminate the inference pass."
)


@register_rule
class SchemaInferenceRule(BaseRule):
    META = RuleMeta(
        rule_id="CY018",
        name="schema-inference",
        description="spark.read.csv()/json() without explicit schema — double file scan",
        default_severity=Severity.WARNING,
    )

    def check(self, tree: ast.Module, tracker, filepath: str) -> list[Finding]:
        findings = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue

            # Only check terminal read methods (not "format" — that's intermediate)
            terminal = node.func.attr
            if terminal not in ("csv", "json", "load"):
                continue

            # Must be rooted at spark.read
            if not _find_read_root(node):
                continue

            # Collect the full method chain
            methods = _get_read_chain_methods(node)

            # Determine format
            fmt = _resolve_format(methods)
            if fmt is None or fmt not in INFERENCE_FORMATS:
                continue

            # Check for .schema()
            if _has_schema_call(methods):
                continue

            # Check for inferSchema=false
            if _has_infer_schema_false(methods):
                continue

            # Check for explicit inferSchema=true
            explicit = _has_infer_schema_true(methods)

            msg = _MSG_EXPLICIT if explicit else _MSG_IMPLICIT.format(fmt=fmt)

            findings.append(self._make_finding(
                filepath=filepath,
                line=node.lineno,
                col=node.col_offset,
                message=msg,
            ))

        return findings
