"""PR comment formatting from enriched server response.

Formats linter findings, plan detector findings, and change classifications
into a PR comment body with annotations. Supports both enriched (with $
amounts) and unenriched (pattern-only) output.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FormattedOutput:
    """Structured output for posting to a PR."""
    markdown: str = ""
    annotations: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "markdown": self.markdown,
            "annotations": self.annotations,
            "stats": self.stats,
        }


# ---------------------------------------------------------------------------
# Confidence tier formatting (from change-classification-spec §9)
# ---------------------------------------------------------------------------

PRECISE_TYPES = frozenset({
    "filter_removed", "source_changed",
    "broadcast_hint_removed", "broadcast_hint_added",
    "projection_changed",
})
BOUNDED_TYPES = frozenset({"filter_modified", "filter_added"})
STRUCTURAL_TYPES = frozenset({"operation_removed"})


def _tier_label(change_type: str) -> str:
    if change_type in PRECISE_TYPES:
        return "High confidence"
    if change_type in BOUNDED_TYPES:
        return "Range estimate"
    if change_type in STRUCTURAL_TYPES:
        return "Qualitative"
    return "Info"


def _tier_icon(change_type: str) -> str:
    if change_type in PRECISE_TYPES:
        return "\u2705"
    if change_type in BOUNDED_TYPES:
        return "\u26a0\ufe0f"
    return "\u2139\ufe0f"


# ---------------------------------------------------------------------------
# Individual finding formatters
# ---------------------------------------------------------------------------

def format_linter_finding(finding: dict) -> str:
    """Format an enriched linter finding for PR comment output."""
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
    """Format a plan detector finding for PR comment output."""
    filepath = finding.get("file", "")
    trigger = finding.get("triggerLineAbsolute") or finding.get("absoluteTriggerLine", "?")
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


def format_change_finding(finding: dict, enriched: bool) -> str:
    """Format a change-type finding for PR comment output."""
    change_type = finding.get("type", "unknown")
    icon = _tier_icon(change_type)
    label = _tier_label(change_type)
    message = finding.get("message", change_type)
    savings = finding.get("savings")

    parts = [f"  {icon} **{label}**: {message}"]
    if enriched and savings:
        parts[0] += f" \u2014 cost impact: ${savings:,}/month"

    return "\n".join(parts)


def format_change_classification(classification: dict) -> str:
    """Format a change classification (unenriched, no $ amounts)."""
    change_type = classification.get("type", "unknown")
    icon = _tier_icon(change_type)
    table = classification.get("sourceTable", "")
    scope = classification.get("scope", "")

    detail = f"  {icon} {change_type}"
    if table:
        detail += f" on `{table}`"
    if scope:
        detail += f" ({scope})"
    return detail


# ---------------------------------------------------------------------------
# Annotation builder
# ---------------------------------------------------------------------------

def _build_annotations(
    findings: list[dict],
    plan_findings: list[dict],
) -> list[dict]:
    """Build annotation dicts for inline PR comments."""
    annotations: list[dict] = []
    for f in findings:
        line = f.get("absoluteLine") or f.get("line")
        if not line:
            continue
        annotations.append({
            "file": f.get("file", ""),
            "line": line,
            "level": "warning",
            "message": f"{f.get('rule', '')}: {f.get('message', '')}",
        })

    for pf in plan_findings:
        line = pf.get("absoluteTriggerLine") or pf.get("triggerLineAbsolute")
        if not line:
            continue
        annotations.append({
            "file": pf.get("file", ""),
            "line": line,
            "level": "error",
            "message": pf.get("message", pf.get("detectorId", "")),
        })

    return annotations


# ---------------------------------------------------------------------------
# Main format_output
# ---------------------------------------------------------------------------

def format_output(
    findings: list[dict],
    change_findings: list[dict],
    plan_findings: list[dict],
    change_classifications: list[dict],
    match_stats: dict | None,
    enriched: bool,
) -> FormattedOutput:
    """Format all findings into a PR comment and annotations."""
    sections: list[str] = []

    # Header
    total = len(findings) + len(plan_findings) + len(change_findings)
    if total == 0 and not change_classifications:
        return FormattedOutput(
            markdown="## Cluster Yield\n\nNo findings detected.",
            annotations=[],
            stats={"total": 0},
        )

    header = f"## Cluster Yield \u2014 {total} finding{'s' if total != 1 else ''}"

    # Add total savings if enriched
    if enriched:
        total_savings = sum(
            f.get("savings", 0)
            for f in [*findings, *plan_findings, *change_findings]
        )
        if total_savings > 0:
            header += f" (${total_savings:,}/month impact)"

    sections.append(header)

    # Plan findings
    if plan_findings:
        sections.append("### Plan Detector Findings\n")
        for f in plan_findings:
            sections.append(format_plan_finding(f))

    # Linter findings
    if findings:
        sections.append("### Linter Findings\n")
        for f in findings:
            sections.append(format_linter_finding(f))

    # Change findings (enriched)
    if change_findings:
        sections.append("### Change Impact\n")
        for f in change_findings:
            sections.append(format_change_finding(f, enriched=True))

    # Change classifications (unenriched)
    if change_classifications and not enriched:
        sections.append("### Detected Changes\n")
        for c in change_classifications:
            sections.append(format_change_classification(c))

    # Match stats warning
    if match_stats:
        match_rate = match_stats.get("fingerprintMatchRate")
        if match_rate is not None and match_rate < 0.8:
            sections.append(
                f"\n> \u26a0\ufe0f Fingerprint match rate: {match_rate:.0%}. "
                "Some notebook cells may have changed since last snapshot."
            )

    markdown = "\n\n".join(sections)
    annotations = _build_annotations(findings, plan_findings)

    return FormattedOutput(
        markdown=markdown,
        annotations=annotations,
        stats={"total": total, "enriched": enriched},
    )


# Keep backwards compat for direct usage
def format_pr_comment(
    linter_findings: list[dict],
    plan_findings: list[dict] | None = None,
) -> str:
    """Format all findings into a single PR comment body."""
    result = format_output(
        findings=linter_findings,
        change_findings=[],
        plan_findings=plan_findings or [],
        change_classifications=[],
        match_stats=None,
        enriched=False,
    )
    return result.markdown
