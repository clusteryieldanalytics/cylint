"""Enrich request assembly, HTTP client, and response processing.

Converts linter findings and changed lines into the request shape expected
by the server's ``/enrich`` endpoint, posts the request, and resolves
provenance in the server response back to absolute line numbers.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from cylint.ci.cell_map import (
    absolute_to_cell,
    build_cell_map,
    cell_to_absolute,
    is_databricks_notebook,
)
from cylint.models import Finding


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

ENRICH_ENDPOINT = "/api/v1/environments/{env}/enrich"
DEFAULT_TIMEOUT = 30


@dataclass
class EnrichRequest:
    """Payload for the /enrich endpoint."""
    files: list[dict] = field(default_factory=list)
    linter_findings: list[dict] = field(default_factory=list)
    change_types: list[dict] = field(default_factory=list)
    environment: str = ""


@dataclass
class EnrichResponse:
    """Parsed response from the /enrich endpoint."""
    findings: list[dict] = field(default_factory=list)
    plan_findings: list[dict] = field(default_factory=list)
    change_findings: list[dict] = field(default_factory=list)
    match_stats: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Request assembly
# ---------------------------------------------------------------------------

def convert_finding(
    finding: Finding,
    cell_map: dict[str, int] | None,
) -> dict:
    """Convert a single Finding to the enrich request JSON shape.

    For Databricks notebooks (cell_map provided), emits cellFingerprint +
    cellLine.  For plain .py scripts (cell_map is None), emits file + line.
    """
    base: dict = {
        "rule": finding.rule_id,
        "file": finding.filepath,
        "message": finding.message,
        "tables": [],
    }

    if cell_map is not None:
        result = absolute_to_cell(finding.line, cell_map)
        if result:
            base["cellFingerprint"] = result[0]
            base["cellLine"] = result[1]
        base["absoluteLine"] = finding.line
    else:
        base["line"] = finding.line

    return base


def convert_changed_lines(
    lines: list[int],
    filepath: str,
    cell_map: dict[str, int] | None,
) -> list[dict]:
    """Convert a list of absolute changed-line numbers to enrich request shape."""
    result: list[dict] = []
    for line in lines:
        if cell_map is not None:
            converted = absolute_to_cell(line, cell_map)
            if converted:
                result.append({
                    "cellFingerprint": converted[0],
                    "cellLine": converted[1],
                })
        else:
            result.append({"file": filepath, "line": line})
    return result


def build_enrich_request(
    filepath: str,
    source: str,
    findings: list[Finding],
    changed_lines: list[int],
) -> dict:
    """Build the full enrich request payload for a single file.

    Automatically detects Databricks notebooks and uses the appropriate
    coordinate system.
    """
    cell_map = build_cell_map(source) if is_databricks_notebook(source) else None

    return {
        "files": [{"path": filepath}],
        "linterFindings": [
            convert_finding(f, cell_map) for f in findings
        ],
        "changedLines": convert_changed_lines(changed_lines, filepath, cell_map),
    }


# ---------------------------------------------------------------------------
# Response processing
# ---------------------------------------------------------------------------

def resolve_provenance(
    finding: dict,
    cell_map: dict[str, int],
) -> dict:
    """Convert a plan detector finding's provenance to absolute lines.

    Args:
        finding: A plan detector finding from the server response.
        cell_map: The cell fingerprint → start_line map.

    Returns:
        Dict with triggerLineAbsolute, constructionLinesAbsolute,
        constructionSpanStart, constructionSpanEnd.
    """
    # Trigger line
    trigger_fp = finding.get("triggerCellFingerprint")
    trigger_line = finding.get("triggerLine")
    trigger_absolute = None
    if trigger_fp and trigger_line:
        trigger_absolute = cell_to_absolute(trigger_fp, trigger_line, cell_map)

    # Construction lines → absolute set
    absolute_construction: set[int] = set()
    for entry in finding.get("constructionLines", []):
        fp = entry.get("cellFingerprint")
        if fp:
            for cl in entry.get("lines", []):
                abs_line = cell_to_absolute(fp, cl, cell_map)
                if abs_line is not None:
                    absolute_construction.add(abs_line)

    span_start = min(absolute_construction) if absolute_construction else None
    span_end = max(absolute_construction) if absolute_construction else None

    return {
        "triggerLineAbsolute": trigger_absolute,
        "constructionLinesAbsolute": sorted(absolute_construction),
        "constructionSpanStart": span_start,
        "constructionSpanEnd": span_end,
    }


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------

def post_enrich(
    request: EnrichRequest,
    api_key: str,
    base_url: str = "https://api.clusteryield.app",
    timeout: int = DEFAULT_TIMEOUT,
) -> EnrichResponse | None:
    """POST to the /enrich endpoint. Returns None on failure (non-blocking).

    Enrichment failure should not block the CI pipeline. The orchestrator
    falls back to unenriched findings + change classifications.
    """
    url = f"{base_url}{ENRICH_ENDPOINT.format(env=request.environment)}"
    payload: dict = {
        "files": request.files,
        "linterFindings": request.linter_findings,
    }
    if request.change_types:
        payload["changeTypes"] = request.change_types

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            return EnrichResponse(
                findings=data.get("findings", []),
                plan_findings=data.get("planFindings", []),
                change_findings=data.get("changeFindings", []),
                match_stats=data.get("matchStats", {}),
            )
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as e:
        print(
            f"Warning: enrichment failed ({e}). Showing unenriched findings.",
            file=sys.stderr,
        )
        return None
