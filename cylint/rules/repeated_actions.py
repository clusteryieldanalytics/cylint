"""CY014: Repeated terminal actions on the same DataFrame without .cache()."""

import ast
from collections import defaultdict

from cylint.models import Finding, RuleMeta, Severity
from cylint.rules import BaseRule, register_rule
from cylint.tracker import DataFrameTracker, find_root_name

# Actions that represent real work — count for CY014
TERMINAL_ACTIONS = frozenset({
    "count", "collect", "take", "first", "head",
    "toPandas", "toLocalIterator",
    # write terminals (when .write is chained):
    "save", "saveAsTable", "parquet", "csv", "json", "orc",
    "insertInto", "writeTo",
})

# Debug methods — excluded from CY014 action count, handled by CY012
DEBUG_ACTIONS = frozenset({"show", "display", "printSchema", "explain"})

CACHE_METHODS = frozenset({"cache", "persist"})


@register_rule
class RepeatedActionsRule(BaseRule):
    META = RuleMeta(
        rule_id="CY014",
        name="repeated-actions-no-cache",
        description="Multiple terminal actions without .cache() — recomputes full lineage each time",
        default_severity=Severity.CRITICAL,
    )

    def check(self, tree: ast.Module, tracker: DataFrameTracker, filepath: str) -> list[Finding]:
        actions: dict[str, list[tuple[str, int]]] = defaultdict(list)

        for node in _walk_ordered(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue

            method = func.attr
            root = find_root_name(func.value)
            if root is None or not tracker.is_tracked(root):
                continue

            # Skip cache/persist — not actions, tracker handles cache status
            if method in CACHE_METHODS:
                continue

            # Skip debug methods — CY012 handles these
            if method in DEBUG_ACTIONS:
                continue

            # Record terminal action
            if method in TERMINAL_ACTIONS:
                actions[root].append((method, node.lineno))

        # Emit findings
        findings = []
        for df_name, acts in actions.items():
            if len(acts) < 2:
                continue

            # Single source of truth: tracker knows cache status
            info = tracker.get_info(df_name)
            if info and info.has_cache:
                continue

            action_count = len(acts)
            action_names = [a[0] for a in acts]
            first_line = acts[0][1]

            display_names = ", ".join(f".{n}()" for n in action_names[:3])
            if action_count > 3:
                display_names += " ..."

            findings.append(Finding(
                rule_id=self.META.rule_id,
                severity=self.severity,
                message=(
                    f"DataFrame `{df_name}` has {action_count} terminal actions "
                    f"({display_names}) without .cache(). "
                    "Each action recomputes the full lineage from source. "
                    "Add .cache() before the first action, and .unpersist() when done."
                ),
                filepath=filepath,
                line=first_line,
                col=0,
                action_count=action_count,
            ))

        return findings


def _walk_ordered(tree: ast.AST) -> list[ast.AST]:
    """Walk AST nodes in approximate source order (by line number)."""
    nodes = list(ast.walk(tree))
    nodes.sort(key=lambda n: (getattr(n, "lineno", 0), getattr(n, "col_offset", 0)))
    return nodes
