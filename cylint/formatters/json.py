"""JSON formatter for structured output."""

import json

from cylint.models import Finding, LintResult


def _format_finding(f: Finding) -> dict:
    """Format a single finding as a dict, including cell coords if present."""
    d: dict = {
        "rule_id": f.rule_id,
        "severity": str(f.severity),
        "message": f.message,
        "filepath": f.filepath,
        "line": f.line,
        "col": f.col,
        "suggestion": f.suggestion,
    }
    if f.cell_fingerprint is not None:
        d["cellFingerprint"] = f.cell_fingerprint
        d["cellLine"] = f.cell_line
    return d


def format_result(result: LintResult, *, export_cells: bool = False) -> str:
    """Format lint result as JSON."""
    output: dict = {
        "files_scanned": result.files_scanned,
        "total_findings": len(result.findings),
        "exit_code": result.exit_code,
        "counts": {str(k): v for k, v in result.count_by_severity.items()},
        "findings": [_format_finding(f) for f in result.findings],
        "errors": result.errors,
    }
    if export_cells and result.cell_maps:
        output["cellMaps"] = {
            filepath: cell_map
            for filepath, cell_map in result.cell_maps.items()
        }
    return json.dumps(output, indent=2)
