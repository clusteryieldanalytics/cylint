"""CY016: Invalid escape sequence in source file.

Detection is handled in the engine during ast.parse() — this module
registers the rule so it appears in ``cy rules`` and can be
configured / disabled like any other rule.
"""

import ast

from cylint.models import Finding, RuleMeta, Severity
from cylint.rules import BaseRule, register_rule
from cylint.tracker import DataFrameTracker


@register_rule
class InvalidEscapeRule(BaseRule):
    META = RuleMeta(
        rule_id="CY016",
        name="invalid-escape-sequence",
        description="Invalid escape sequence in string literal (use raw strings for regex)",
        default_severity=Severity.INFO,
    )

    def check(self, tree: ast.Module, tracker: DataFrameTracker, filepath: str) -> list[Finding]:
        # Findings are emitted by the engine when it captures SyntaxWarnings
        # during ast.parse().  Nothing to do here.
        return []
