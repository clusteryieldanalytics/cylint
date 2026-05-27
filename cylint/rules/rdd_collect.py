"""CY034: df.rdd.collect() — use .toPandas() with Arrow instead."""

import ast

from cylint.models import Finding, RuleMeta, Severity
from cylint.rules import BaseRule, register_rule
from cylint.tracker import DataFrameTracker, find_root_name


@register_rule
class RddCollectRule(BaseRule):
    META = RuleMeta(
        rule_id="CY034",
        name="rdd-collect",
        description="df.rdd.collect() — use .toPandas() with Arrow for faster driver collection",
        default_severity=Severity.WARNING,
    )

    def check(self, tree: ast.Module, tracker: DataFrameTracker, filepath: str) -> list[Finding]:
        findings = []
        for node in ast.walk(tree):
            # Looking for: <expr>.rdd.collect()
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            if func.attr != "collect":
                continue

            # The receiver of .collect() must be an .rdd attribute access
            rdd_attr = func.value
            if not isinstance(rdd_attr, ast.Attribute):
                continue
            if rdd_attr.attr != "rdd":
                continue

            # The receiver of .rdd must be a tracked DataFrame
            root = find_root_name(rdd_attr.value)
            if root is None or not tracker.is_tracked(root):
                continue

            findings.append(self._make_finding(
                filepath=filepath,
                line=node.lineno,
                col=node.col_offset,
                message=(
                    ".rdd.collect() collects rows through the Python RDD API, "
                    "which serialises every row individually. "
                    ".toPandas() with Arrow enabled is up to 10x faster for "
                    "driver-side collection."
                ),
                suggestion=(
                    "Enable Arrow: spark.conf.set("
                    "'spark.sql.execution.arrow.pyspark.enabled', 'true'), "
                    "then replace .rdd.collect() with .toPandas()."
                ),
            ))
        return findings
