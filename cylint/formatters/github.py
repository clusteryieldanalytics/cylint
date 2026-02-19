"""GitHub Actions annotation format for inline PR comments."""

from cylint.models import LintResult, Severity

# Map our severity to GitHub annotation levels
_GITHUB_LEVELS = {
    Severity.CRITICAL: "error",
    Severity.WARNING: "warning",
    Severity.INFO: "notice",
}


def format_result(result: LintResult) -> str:
    """Format lint result as GitHub Actions workflow commands."""
    lines = []
    for f in result.findings:
        level = _GITHUB_LEVELS.get(f.severity, "notice")
        msg = f.message
        if f.suggestion:
            msg += f" {f.suggestion}"
        # Escape special characters for GitHub Actions
        msg = msg.replace("%", "%25").replace("\n", "%0A").replace("\r", "%0D")
        lines.append(
            f"::{level} file={f.filepath},line={f.line},col={f.col},"
            f"title={f.rule_id}::{msg}"
        )
    return "\n".join(lines)
