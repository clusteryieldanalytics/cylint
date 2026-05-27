"""CY036: df.write.insertInto() — use df.writeTo().append()/.overwritePartitions()."""

import ast
from typing import Optional

from cylint.models import Finding, RuleMeta, Severity
from cylint.rules import BaseRule, register_rule
from cylint.tracker import DataFrameTracker, find_root_name


def _get_overwrite_value(call_node: ast.Call) -> Optional[bool]:
    """Return the value of the overwrite= kwarg, or None if not present."""
    for kw in call_node.keywords:
        if kw.arg == "overwrite":
            if isinstance(kw.value, ast.Constant):
                return bool(kw.value.value)
    return None


@register_rule
class InsertIntoRule(BaseRule):
    META = RuleMeta(
        rule_id="CY036",
        name="insert-into-deprecated",
        description="df.write.insertInto() — use df.writeTo().append()/.overwritePartitions()",
        default_severity=Severity.WARNING,
    )

    def check(self, tree: ast.Module, tracker: DataFrameTracker, filepath: str) -> list[Finding]:
        findings = []
        for node in ast.walk(tree):
            # Looking for: <expr>.write.insertInto(table, ...)
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            if func.attr != "insertInto":
                continue

            # Receiver must be a .write attribute
            write_attr = func.value
            if not isinstance(write_attr, ast.Attribute):
                continue
            if write_attr.attr != "write":
                continue

            # Root of the chain must be a tracked DataFrame
            root = find_root_name(write_attr.value)
            if root is None or not tracker.is_tracked(root):
                continue

            overwrite = _get_overwrite_value(node)

            if overwrite:
                message = (
                    "df.write.insertInto(table, overwrite=True) uses the legacy "
                    "DataFrameWriter API and overwrites the whole table. "
                    "Use df.writeTo(table).overwritePartitions() instead — it "
                    "only replaces matching partitions and supports fine-grained "
                    "partition overwrite semantics."
                )
                suggestion = "Replace with df.writeTo(table).overwritePartitions()"
            else:
                message = (
                    "df.write.insertInto(table) uses the legacy DataFrameWriter API. "
                    "Use df.writeTo(table).append() instead — it is more explicit, "
                    "supports table properties, and is compatible with Iceberg/Delta."
                )
                suggestion = "Replace with df.writeTo(table).append()"

            findings.append(self._make_finding(
                filepath=filepath,
                line=node.lineno,
                col=node.col_offset,
                message=message,
                suggestion=suggestion,
            ))
        return findings
