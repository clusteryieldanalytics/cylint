"""CY014: Repeated terminal actions on the same DataFrame without .cache()."""

import ast
from collections import defaultdict

from cylint.models import Finding, RuleMeta, Severity
from cylint.rules import BaseRule, register_rule
from cylint.tracker import DataFrameTracker, find_root_name

TERMINAL_ACTIONS = frozenset({
    "count", "collect", "take", "first", "head",
    "show", "toPandas", "toLocalIterator",
    # write terminals (when .write is chained):
    "save", "saveAsTable", "parquet", "csv", "json", "orc",
    "insertInto", "writeTo",
})

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
        # Per-variable tracking: actions and cache points
        actions: dict[str, list[tuple[str, int]]] = defaultdict(list)  # df -> [(method, line)]
        cached_at: dict[str, int] = {}  # df -> line of first cache/persist

        # Walk AST in source order (ast.walk is not ordered, so use _walk_ordered)
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

            # Record cache/persist
            if method in CACHE_METHODS:
                if root not in cached_at:
                    cached_at[root] = node.lineno
                continue

            # Record terminal action
            if method in TERMINAL_ACTIONS:
                actions[root].append((method, node.lineno))
                continue

            # Write chain: df.write.parquet(...) — the root is the df before .write
            if method == "write":
                # The actual terminal will be caught separately
                continue

        # Also handle: df2 = df.cache() — mark df as cached
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            value = node.value
            if not isinstance(value, ast.Call):
                continue
            if not isinstance(value.func, ast.Attribute):
                continue
            if value.func.attr in CACHE_METHODS:
                source = find_root_name(value.func.value)
                if source and tracker.is_tracked(source) and source not in cached_at:
                    cached_at[source] = node.lineno

        # Also handle variable reassignment: if df = df.filter(...), reset actions
        # We track this by checking if the same name is assigned from a chain on itself
        reassigned_lines = self._find_reassignment_lines(tree, tracker)

        # Emit findings for uncached repeats
        findings = []
        for df_name, acts in actions.items():
            if len(acts) < 2:
                continue

            # Filter out actions that happen after a reassignment of the same variable
            filtered_acts = self._filter_actions_by_reassignment(
                df_name, acts, reassigned_lines.get(df_name, [])
            )
            if len(filtered_acts) < 2:
                continue

            cache_line = cached_at.get(df_name, float("inf"))
            second_action_line = filtered_acts[1][1]
            if cache_line < second_action_line:
                continue  # cached before second action — OK

            action_count = len(filtered_acts)
            action_names = [a[0] for a in filtered_acts]
            first_line = filtered_acts[0][1]

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

    def _find_reassignment_lines(
        self, tree: ast.Module, tracker: DataFrameTracker
    ) -> dict[str, list[int]]:
        """Find lines where a tracked DF is reassigned from a chain on itself."""
        reassigned: dict[str, list[int]] = defaultdict(list)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue
                name = target.id
                if not tracker.is_tracked(name):
                    continue
                # Check if the RHS is a method chain on the same variable
                root = find_root_name(node.value)
                if root == name:
                    reassigned[name].append(node.lineno)
        return reassigned

    def _filter_actions_by_reassignment(
        self,
        df_name: str,
        acts: list[tuple[str, int]],
        reassign_lines: list[int],
    ) -> list[tuple[str, int]]:
        """Keep only actions in the last contiguous segment (after last reassignment)."""
        if not reassign_lines:
            return acts

        # Find the last reassignment line before any action
        last_reassign = max(reassign_lines)
        # Only keep actions after the last reassignment
        return [(m, l) for m, l in acts if l > last_reassign] or acts


def _walk_ordered(tree: ast.AST) -> list[ast.AST]:
    """Walk AST nodes in approximate source order (by line number)."""
    nodes = list(ast.walk(tree))
    nodes.sort(key=lambda n: (getattr(n, "lineno", 0), getattr(n, "col_offset", 0)))
    return nodes
