"""CY025: .cache()/.persist() without .unpersist() in same scope."""

import ast

from cylint.models import Finding, RuleMeta, Severity
from cylint.rules import BaseRule, register_rule
from cylint.tracker import DataFrameTracker, get_chain_methods


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
        is_module = isinstance(scope, ast.Module)

        cached_vars: dict[str, tuple[int, str]] = {}  # var_name -> (line, method)
        unpersisted: set[str] = set()

        # TODO: CY005 deduplication — if CY005 fires on a single-use cache,
        # CY025 should not also fire. Requires cross-rule communication.
        nodes = self._get_scope_level_nodes(scope)

        for node in nodes:
            # Detect cache/persist in assignment via method chain inspection
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if not isinstance(target, ast.Name):
                        continue
                    if not isinstance(node.value, ast.Call):
                        continue
                    methods = set(get_chain_methods(node.value))
                    cache_method = None
                    if "cache" in methods:
                        cache_method = "cache"
                    elif "persist" in methods:
                        cache_method = "persist"
                    if cache_method is not None:
                        cached_vars[target.id] = (node.lineno, cache_method)

            # Detect standalone cache/persist: df.cache()
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                func = node.value.func
                if isinstance(func, ast.Attribute) and func.attr in ("cache", "persist"):
                    if isinstance(func.value, ast.Name):
                        cached_vars[func.value.id] = (node.lineno, func.attr)

            # Detect .unpersist() calls (standalone or in expression)
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                func = node.value.func
                if isinstance(func, ast.Attribute) and func.attr == "unpersist":
                    if isinstance(func.value, ast.Name):
                        unpersisted.add(func.value.id)

            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "unpersist":
                    if isinstance(node.func.value, ast.Name):
                        unpersisted.add(node.func.value.id)

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
