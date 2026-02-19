"""Rule registry and base class for lint rules."""

import ast
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from cylint.models import Finding, RuleMeta, Severity

if TYPE_CHECKING:
    from cylint.tracker import DataFrameTracker

# Global rule registry
_RULES: dict[str, type["BaseRule"]] = {}


def register_rule(cls: type["BaseRule"]) -> type["BaseRule"]:
    """Decorator to register a rule class."""
    _RULES[cls.META.rule_id] = cls
    return cls


def get_all_rules() -> dict[str, type["BaseRule"]]:
    """Return all registered rules."""
    return dict(_RULES)


class BaseRule(ABC):
    """Base class for all lint rules."""

    META: RuleMeta  # Subclasses must define this

    def __init__(self, severity_override: Severity | None = None):
        self.severity = severity_override or self.META.default_severity

    @abstractmethod
    def check(
        self,
        tree: ast.Module,
        tracker: "DataFrameTracker",
        filepath: str,
    ) -> list[Finding]:
        """Run this rule against a parsed AST and return findings."""
        ...

    def _make_finding(
        self,
        filepath: str,
        line: int,
        message: str,
        col: int = 0,
        suggestion: str | None = None,
    ) -> Finding:
        return Finding(
            rule_id=self.META.rule_id,
            severity=self.severity,
            message=message,
            filepath=filepath,
            line=line,
            col=col,
            suggestion=suggestion,
        )
