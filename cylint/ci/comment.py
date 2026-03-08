"""PR comment formatting from enriched server response."""

from __future__ import annotations


def format_linter_finding(finding: dict) -> str:
    """Format an enriched linter finding for PR comment output.

    Args:
        finding: Enriched linter finding dict with absoluteLine (notebook)
                 or line (plain .py), rule, message, and optional savings.
    """
    filepath = finding.get("file", "")
    line = finding.get("absoluteLine") or finding.get("line", "?")
    rule = finding.get("rule", "")
    message = finding.get("message", "")
    savings = finding.get("savings")

    parts = [f"> {filepath}:{line}"]
    severity_icon = "\u274c" if savings and savings > 1000 else "\u26a0\ufe0f"

    detail = f"  {severity_icon} {rule}: {message}"
    if savings:
        detail += f" \u2014 save ~${savings:,}/month"
    parts.append(detail)

    return "\n".join(parts)


def format_plan_finding(finding: dict) -> str:
    """Format a plan detector finding for PR comment output.

    Args:
        finding: Plan detector finding with resolved absolute provenance
                 (triggerLineAbsolute, constructionSpanStart, constructionSpanEnd).
    """
    filepath = finding.get("file", "")
    trigger = finding.get("triggerLineAbsolute", "?")
    span_start = finding.get("constructionSpanStart")
    span_end = finding.get("constructionSpanEnd")
    detector = finding.get("detectorId", "")
    savings = finding.get("savings", 0)
    message = finding.get("message", detector)

    header = f"> {filepath}:{trigger}"
    if span_start and span_end:
        header += f" (code spans lines {span_start}-{span_end})"

    detail = f"  \u274c {message}"
    if savings:
        detail += f" \u2014 save ~${savings:,}/month"

    return "\n".join([header, detail])


def format_pr_comment(
    linter_findings: list[dict],
    plan_findings: list[dict] | None = None,
) -> str:
    """Format all findings into a single PR comment body.

    Args:
        linter_findings: Enriched linter findings from server response.
        plan_findings: Plan detector findings with resolved provenance.
    """
    sections: list[str] = []

    if plan_findings:
        sections.append("### Plan Detector Findings\n")
        for f in plan_findings:
            sections.append(format_plan_finding(f))

    if linter_findings:
        sections.append("### Linter Findings\n")
        for f in linter_findings:
            sections.append(format_linter_finding(f))

    if not sections:
        return "No findings detected."

    return "\n\n".join(sections)
