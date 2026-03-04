"""CY005: .cache() / .persist() with no downstream reuse."""

import ast

from cylint.models import Finding, RuleMeta, Severity
from cylint.rules import BaseRule, register_rule
from cylint.tracker import DataFrameTracker, find_root_name


@register_rule
class CacheRule(BaseRule):
    META = RuleMeta(
        rule_id="CY005",
        name="cache-no-reuse",
        description=".cache() / .persist() with single downstream use",
        default_severity=Severity.WARNING,
    )

    def check(self, tree: ast.Module, tracker: DataFrameTracker, filepath: str) -> list[Finding]:
        findings = []

        # Step 1: Find all assignments where .cache() or .persist() is called
        #         on a tracked DataFrame.
        cached_vars: dict[str, int] = {}  # var_name → line_number

        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if not isinstance(node.value, ast.Call):
                continue
            func = node.value.func
            if not isinstance(func, ast.Attribute):
                continue
            if func.attr not in ("cache", "persist"):
                continue

            # Must be on a tracked DataFrame
            root = find_root_name(func.value)
            if root is not None and not tracker.is_tracked(root):
                continue

            for target in node.targets:
                if isinstance(target, ast.Name):
                    cached_vars[target.id] = node.lineno

        # Also catch: df.cache() as a standalone expression (no assignment)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Expr):
                continue
            if not isinstance(node.value, ast.Call):
                continue
            func = node.value.func
            if not isinstance(func, ast.Attribute):
                continue
            if func.attr not in ("cache", "persist"):
                continue
            # Get the variable it's called on — must be a tracked DataFrame
            if isinstance(func.value, ast.Name):
                if not tracker.is_tracked(func.value.id):
                    continue
                cached_vars[func.value.id] = node.lineno

        if not cached_vars:
            return findings

        # Step 2: Count uses of each cached variable after the cache line
        use_counts: dict[str, int] = {name: 0 for name in cached_vars}

        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in use_counts:
                # Only count uses after the cache line
                if hasattr(node, "lineno") and node.lineno > cached_vars[node.id]:
                    use_counts[node.id] += 1

        # Step 3: Flag cached vars with 0 or 1 downstream use
        for name, count in use_counts.items():
            if count <= 1:
                findings.append(self._make_finding(
                    filepath=filepath,
                    line=cached_vars[name],
                    col=0,
                    message=(
                        f"Cached DataFrame '{name}' is only used "
                        f"{count} time{'s' if count != 1 else ''} after caching. "
                        f"Remove .cache() to free executor memory."
                    ),
                    suggestion=(
                        "Cache is only beneficial when the same DataFrame "
                        "is used in multiple actions."
                    ),
                ))

        return findings
