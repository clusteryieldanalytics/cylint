"""CY008: .repartition() immediately before .write()."""

import ast

from cylint.models import Finding, RuleMeta, Severity
from cylint.rules import BaseRule, register_rule
from cylint.tracker import DataFrameTracker, get_chain_methods

# Write methods that indicate the end of a write chain
WRITE_METHODS = frozenset({
    "parquet", "csv", "json", "orc", "text", "save", "saveAsTable",
    "insertInto", "jdbc",
})


@register_rule
class RepartitionWriteRule(BaseRule):
    META = RuleMeta(
        rule_id="CY008",
        name="repartition-before-write",
        description=".repartition() immediately before .write() triggers unnecessary shuffle",
        default_severity=Severity.INFO,
    )

    def check(self, tree: ast.Module, tracker: DataFrameTracker, filepath: str) -> list[Finding]:
        findings = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            # Look for write-terminal method chains
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue

            # Get the full method chain
            methods = get_chain_methods(node)

            # Look for pattern: ...repartition(...)...write...parquet/csv/etc
            # The chain is in order, so check if repartition precedes write
            repartition_idx = None
            write_idx = None

            for i, method in enumerate(methods):
                if method == "repartition":
                    repartition_idx = i
                if method == "write":
                    write_idx = i
                if method in WRITE_METHODS and write_idx is not None:
                    # Found a write terminal
                    if repartition_idx is not None and repartition_idx < write_idx:
                        # Check there's no other transformation between repartition and write
                        between = methods[repartition_idx + 1:write_idx]
                        non_write_between = [m for m in between if m not in ("mode", "format", "option", "options", "partitionBy", "bucketBy", "sortBy")]
                        if not non_write_between:
                            findings.append(self._make_finding(
                                filepath=filepath,
                                line=node.lineno,
                                col=node.col_offset,
                                message=(
                                    ".repartition() before write triggers a full shuffle."
                                ),
                                suggestion=(
                                    "For fewer output files, consider .coalesce(N) "
                                    "(avoids shuffle) or .option('maxRecordsPerFile', N)."
                                ),
                            ))
                            break  # one finding per chain

        return findings
