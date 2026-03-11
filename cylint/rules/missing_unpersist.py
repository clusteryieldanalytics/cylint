"""CY025: .cache()/.persist() without .unpersist() in same scope."""

import ast

from cylint.models import Finding, RuleMeta, Severity
from cylint.rules import BaseRule, register_rule
from cylint.tracker import DataFrameTracker, get_chain_methods


def _expr_key(node: ast.expr) -> str | None:
    """Extract a dotted name key from an expression for tracking.

    Supports:
        ast.Name("df")           -> "df"
        ast.Attribute(Name("self"), "df") -> "self.df"
    Returns None for anything more complex.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return f"{node.value.id}.{node.attr}"
    return None


def _find_receiver_key(call_node: ast.Call) -> str | None:
    """Get the key of the object a method is called on.

    For `df.cache()` returns "df".
    For `self.df.cache()` returns "self.df".
    """
    if not isinstance(call_node.func, ast.Attribute):
        return None
    return _expr_key(call_node.func.value)


@register_rule
class MissingUnpersistRule(BaseRule):
    META = RuleMeta(
        rule_id="CY025",
        name="missing-unpersist",
        description=".cache()/.persist() without corresponding .unpersist() in same scope",
        default_severity=Severity.WARNING,
    )

    def check(self, tree: ast.Module, tracker: DataFrameTracker, filepath: str) -> list[Finding]:
        findings = []

        # Process module-level code and each function separately
        module_findings = self._check_scope(tree, tracker, filepath)
        findings.extend(module_findings)

        # Function/method scopes
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                scope_findings = self._check_scope(node, tracker, filepath)
                findings.extend(scope_findings)

        return findings

    def _check_scope(self, scope: ast.AST, tracker: DataFrameTracker, filepath: str) -> list[Finding]:
        """Check a single scope for cache without unpersist."""
        cached_vars: dict[str, tuple[int, str]] = {}  # key -> (line, method)
        unpersisted: set[str] = set()

        # TODO: CY005 deduplication — if CY005 fires on a single-use cache,
        # CY025 should not also fire. Requires cross-rule communication.
        nodes = self._get_scope_level_nodes(scope)

        for node in nodes:
            # Detect cache/persist in assignment: df2 = df.filter(...).cache()
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    key = _expr_key(target)
                    if key is None or not isinstance(node.value, ast.Call):
                        continue
                    methods = set(get_chain_methods(node.value))
                    cache_method = self._detect_cache_method(methods)
                    if cache_method is not None:
                        cached_vars[key] = (node.lineno, cache_method)
                    # Also detect unpersist in assignment: result = df.unpersist()
                    if "unpersist" in methods:
                        recv = _find_receiver_key(node.value)
                        if recv is not None:
                            unpersisted.add(recv)

            # Detect cache/persist in annotated assignment: df: DataFrame = spark.table(...).cache()
            if isinstance(node, ast.AnnAssign) and node.value is not None:
                target_key = _expr_key(node.target) if isinstance(node.target, (ast.Name, ast.Attribute)) else None
                if target_key is not None and isinstance(node.value, ast.Call):
                    methods = set(get_chain_methods(node.value))
                    cache_method = self._detect_cache_method(methods)
                    if cache_method is not None:
                        cached_vars[target_key] = (node.lineno, cache_method)
                    if "unpersist" in methods:
                        recv = _find_receiver_key(node.value)
                        if recv is not None:
                            unpersisted.add(recv)

            # Detect standalone cache/persist: df.cache() / self.df.persist()
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                func = node.value.func
                if isinstance(func, ast.Attribute) and func.attr in ("cache", "persist"):
                    key = _expr_key(func.value)
                    if key is not None:
                        cached_vars[key] = (node.lineno, func.attr)

            # Detect .unpersist() calls anywhere in walked nodes
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "unpersist":
                    key = _expr_key(node.func.value)
                    if key is not None:
                        unpersisted.add(key)

        # Emit findings for cached vars without unpersist
        findings = []
        for var_name, (line, method) in cached_vars.items():
            if var_name in unpersisted:
                continue

            findings.append(self._make_finding(
                filepath=filepath,
                line=line,
                col=0,
                message=(
                    f".{method}() on `{var_name}` without .unpersist() in the same scope. "
                    "Cached DataFrames consume executor memory until the session ends. "
                    f"Add `{var_name}.unpersist()` after the cached data is no longer needed."
                ),
            ))

        return findings

    @staticmethod
    def _detect_cache_method(methods: set[str]) -> str | None:
        if "cache" in methods:
            return "cache"
        if "persist" in methods:
            return "persist"
        return None

    def _get_scope_level_nodes(self, scope: ast.AST) -> list[ast.AST]:
        """Get AST nodes in scope, excluding nested function/class bodies."""
        body = getattr(scope, "body", [])
        nodes = []
        for stmt in body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            for node in ast.walk(stmt):
                nodes.append(node)
        return nodes
