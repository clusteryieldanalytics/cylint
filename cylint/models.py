"""Data models for cylint linter findings."""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


class Severity(IntEnum):
    """Finding severity levels, ordered for filtering."""
    INFO = 0
    WARNING = 1
    CRITICAL = 2

    @classmethod
    def from_string(cls, s: str) -> "Severity":
        return cls[s.upper()]

    def __str__(self) -> str:
        return self.name.lower()


@dataclass
class Finding:
    """A single lint finding."""
    rule_id: str
    severity: Severity
    message: str
    filepath: str
    line: int
    col: int = 0
    end_line: Optional[int] = None
    suggestion: Optional[str] = None
    action_count: Optional[int] = None  # CY014: number of repeated actions detected

    def __str__(self) -> str:
        return (
            f"{self.filepath}:{self.line}:{self.col}: "
            f"{self.rule_id} [{self.severity}] {self.message}"
        )


@dataclass
class RuleMeta:
    """Metadata for a lint rule."""
    rule_id: str
    name: str
    description: str
    default_severity: Severity


@dataclass
class LintResult:
    """Aggregated result from linting one or more files."""
    findings: list[Finding] = field(default_factory=list)
    files_scanned: int = 0
    errors: dict[str, str] = field(default_factory=dict)  # filepath → error msg

    @property
    def count_by_severity(self) -> dict[Severity, int]:
        counts = {s: 0 for s in Severity}
        for f in self.findings:
            counts[f.severity] += 1
        return counts

    @property
    def exit_code(self) -> int:
        """0 = clean, 1 = warnings only, 2 = critical findings."""
        counts = self.count_by_severity
        if counts[Severity.CRITICAL] > 0:
            return 2
        if counts[Severity.WARNING] > 0:
            return 1
        if counts[Severity.INFO] > 0:
            return 1
        return 0
