"""Text formatter with ANSI colors for terminal output."""

from cylint.models import Finding, LintResult, Severity


# ANSI color codes
class _Colors:
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    GRAY = "\033[90m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


SEVERITY_COLORS = {
    Severity.CRITICAL: _Colors.RED,
    Severity.WARNING: _Colors.YELLOW,
    Severity.INFO: _Colors.CYAN,
}


def format_finding(finding: Finding, use_color: bool = True) -> str:
    """Format a single finding for terminal display."""
    if use_color:
        c = SEVERITY_COLORS.get(finding.severity, _Colors.GRAY)
        location = f"{_Colors.BOLD}{finding.filepath}:{finding.line}:{finding.col}{_Colors.RESET}"
        rule = f"{c}{finding.rule_id}{_Colors.RESET}"
        severity = f"{c}[{finding.severity}]{_Colors.RESET}"
        message = finding.message
        lines = [f"{location}: {rule} {severity} {message}"]
        if finding.suggestion:
            lines.append(f"  {_Colors.GRAY}{finding.suggestion}{_Colors.RESET}")
    else:
        location = f"{finding.filepath}:{finding.line}:{finding.col}"
        lines = [f"{location}: {finding.rule_id} [{finding.severity}] {finding.message}"]
        if finding.suggestion:
            lines.append(f"  {finding.suggestion}")
    return "\n".join(lines)


def format_result(result: LintResult, use_color: bool = True) -> str:
    """Format the full lint result for terminal display."""
    if not result.findings and not result.errors:
        if use_color:
            return f"\n{_Colors.BOLD}✓ No issues found in {result.files_scanned} file(s).{_Colors.RESET}\n"
        return f"\nNo issues found in {result.files_scanned} file(s).\n"

    lines = []

    # Group findings by file
    by_file: dict[str, list[Finding]] = {}
    for f in result.findings:
        by_file.setdefault(f.filepath, []).append(f)

    for filepath, findings in by_file.items():
        lines.append("")
        for finding in findings:
            lines.append(format_finding(finding, use_color))

    # Errors
    for filepath, error in result.errors.items():
        if use_color:
            lines.append(f"\n{_Colors.RED}Error parsing {filepath}: {error}{_Colors.RESET}")
        else:
            lines.append(f"\nError parsing {filepath}: {error}")

    # Summary
    counts = result.count_by_severity
    parts = []
    if counts[Severity.CRITICAL]:
        part = f"{counts[Severity.CRITICAL]} critical"
        parts.append(f"{_Colors.RED}{part}{_Colors.RESET}" if use_color else part)
    if counts[Severity.WARNING]:
        part = f"{counts[Severity.WARNING]} warning(s)"
        parts.append(f"{_Colors.YELLOW}{part}{_Colors.RESET}" if use_color else part)
    if counts[Severity.INFO]:
        part = f"{counts[Severity.INFO]} info"
        parts.append(f"{_Colors.CYAN}{part}{_Colors.RESET}" if use_color else part)

    total = len(result.findings)
    summary = f"Found {total} issue(s) ({', '.join(parts)}) in {result.files_scanned} file(s)."
    if use_color:
        lines.append(f"\n{_Colors.BOLD}{summary}{_Colors.RESET}\n")
    else:
        lines.append(f"\n{summary}\n")

    return "\n".join(lines)
