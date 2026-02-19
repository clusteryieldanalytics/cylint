"""CY002: UDF where a builtin PySpark function exists."""

import ast
from typing import Optional

from cylint.models import Finding, RuleMeta, Severity
from cylint.rules import BaseRule, register_rule
from cylint.tracker import DataFrameTracker

# Maps Python string/math/date method names to PySpark builtin equivalents
UDF_BUILTIN_MAP: dict[str, str] = {
    # String operations
    "lower": "F.lower()",
    "upper": "F.upper()",
    "strip": "F.trim()",
    "lstrip": "F.ltrim()",
    "rstrip": "F.rtrim()",
    "startswith": "F.col(...).startswith()",
    "endswith": "F.col(...).endswith()",
    "replace": "F.regexp_replace()",
    "split": "F.split()",
    "len": "F.length()",
    "find": "F.locate()",
    "zfill": "F.lpad()",
    # Math operations
    "abs": "F.abs()",
    "round": "F.round()",
    "ceil": "F.ceil()",
    "floor": "F.floor()",
    "sqrt": "F.sqrt()",
    "log": "F.log()",
    "exp": "F.exp()",
    "pow": "F.pow()",
    # Type conversions
    "int": "F.col(...).cast('int')",
    "float": "F.col(...).cast('float')",
    "str": "F.col(...).cast('string')",
    "bool": "F.col(...).cast('boolean')",
    # Null handling
    "is None": "F.isnull() / F.coalesce()",
}

# Python builtins that have PySpark equivalents
PYTHON_BUILTIN_MAP: dict[str, str] = {
    "len": "F.length()",
    "abs": "F.abs()",
    "round": "F.round()",
    "max": "F.greatest()",
    "min": "F.least()",
    "str": "F.col(...).cast('string')",
    "int": "F.col(...).cast('int')",
    "float": "F.col(...).cast('float')",
}


@register_rule
class UDFRule(BaseRule):
    META = RuleMeta(
        rule_id="CY002",
        name="udf-replaceable",
        description="UDF where a builtin PySpark function exists",
        default_severity=Severity.WARNING,
    )

    def check(self, tree: ast.Module, tracker: DataFrameTracker, filepath: str) -> list[Finding]:
        findings = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            # Pattern 1: udf(lambda x: x.method())
            # Pattern 2: udf(some_function)
            # Pattern 3: @udf decorator
            finding = self._check_udf_call(node, filepath)
            if finding:
                findings.append(finding)

        # Also check @udf decorated functions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                finding = self._check_udf_decorator(node, filepath)
                if finding:
                    findings.append(finding)

        return findings

    def _check_udf_call(self, node: ast.Call, filepath: str) -> Optional[Finding]:
        """Check udf(lambda/func) calls."""
        func = node.func
        # Match: udf(...) or F.udf(...) or spark.udf.register(...)
        is_udf_call = False
        if isinstance(func, ast.Name) and func.id == "udf":
            is_udf_call = True
        elif isinstance(func, ast.Attribute) and func.attr == "udf":
            is_udf_call = True
        elif (isinstance(func, ast.Attribute) and func.attr == "register"
              and isinstance(func.value, ast.Attribute)
              and func.value.attr == "udf"):
            is_udf_call = True

        if not is_udf_call or not node.args:
            return None

        arg = node.args[0]

        # Check if it's a lambda
        if isinstance(arg, ast.Lambda):
            return self._analyze_lambda(arg, node, filepath)

        # Check if it's a reference to a simple function we can analyze
        # (We'd need the function body, which is harder — skip for MVP)
        return None

    def _analyze_lambda(
        self, lam: ast.Lambda, udf_node: ast.Call, filepath: str
    ) -> Optional[Finding]:
        """Analyze a lambda body for replaceable operations."""
        body = lam.body

        # Pattern: lambda x: x.lower()
        if isinstance(body, ast.Call) and isinstance(body.func, ast.Attribute):
            method_name = body.func.attr
            if method_name in UDF_BUILTIN_MAP:
                return self._make_finding(
                    filepath=filepath,
                    line=udf_node.lineno,
                    col=udf_node.col_offset,
                    message=(
                        f"UDF performs .{method_name}() which has a builtin equivalent: "
                        f"{UDF_BUILTIN_MAP[method_name]}. "
                        f"Builtins run in the JVM and avoid Python serialization overhead."
                    ),
                    suggestion=f"Replace with {UDF_BUILTIN_MAP[method_name]}",
                )

        # Pattern: lambda x: builtin(x) e.g., lambda x: len(x)
        if isinstance(body, ast.Call) and isinstance(body.func, ast.Name):
            func_name = body.func.id
            if func_name in PYTHON_BUILTIN_MAP:
                return self._make_finding(
                    filepath=filepath,
                    line=udf_node.lineno,
                    col=udf_node.col_offset,
                    message=(
                        f"UDF uses {func_name}() which has a builtin equivalent: "
                        f"{PYTHON_BUILTIN_MAP[func_name]}. "
                        f"Builtins run in the JVM and avoid Python serialization overhead."
                    ),
                    suggestion=f"Replace with {PYTHON_BUILTIN_MAP[func_name]}",
                )

        # Pattern: lambda x: x is None → F.isnull()
        if isinstance(body, ast.Compare):
            if (len(body.ops) == 1 and isinstance(body.ops[0], ast.Is)
                    and isinstance(body.comparators[0], ast.Constant)
                    and body.comparators[0].value is None):
                return self._make_finding(
                    filepath=filepath,
                    line=udf_node.lineno,
                    col=udf_node.col_offset,
                    message=(
                        "UDF checks for None which has a builtin equivalent: "
                        "F.isnull() / F.coalesce(). "
                        "Builtins run in the JVM and avoid Python serialization overhead."
                    ),
                    suggestion="Replace with F.isnull() or F.coalesce()",
                )

        return None

    def _check_udf_decorator(self, node: ast.FunctionDef, filepath: str) -> Optional[Finding]:
        """Check @udf decorated functions."""
        for decorator in node.decorator_list:
            is_udf = False
            if isinstance(decorator, ast.Name) and decorator.id == "udf":
                is_udf = True
            elif isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Name) and decorator.func.id == "udf":
                    is_udf = True
                elif isinstance(decorator.func, ast.Attribute) and decorator.func.attr == "udf":
                    is_udf = True

            if not is_udf:
                continue

            # Analyze function body for simple return patterns
            if len(node.body) == 1 and isinstance(node.body[0], ast.Return):
                ret = node.body[0].value
                if ret is None:
                    continue

                # Same patterns as lambda analysis
                if isinstance(ret, ast.Call) and isinstance(ret.func, ast.Attribute):
                    method_name = ret.func.attr
                    if method_name in UDF_BUILTIN_MAP:
                        return self._make_finding(
                            filepath=filepath,
                            line=node.lineno,
                            col=node.col_offset,
                            message=(
                                f"UDF performs .{method_name}() which has a builtin equivalent: "
                                f"{UDF_BUILTIN_MAP[method_name]}. "
                                f"Builtins run in the JVM and avoid Python serialization overhead."
                            ),
                            suggestion=f"Replace with {UDF_BUILTIN_MAP[method_name]}",
                        )
        return None
