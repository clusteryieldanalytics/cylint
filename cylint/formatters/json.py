"""JSON formatter for structured output."""

import json

from cylint.models import LintResult


def format_result(result: LintResult) -> str:
    """Format lint result as JSON."""
    output = {
        "files_scanned": result.files_scanned,
        "total_findings": len(result.findings),
        "exit_code": result.exit_code,
        "counts": {str(k): v for k, v in result.count_by_severity.items()},
        "findings": [
            {
                "rule_id": f.rule_id,
                "severity": str(f.severity),
                "message": f.message,
                "filepath": f.filepath,
                "line": f.line,
                "col": f.col,
                "suggestion": f.suggestion,
            }
            for f in result.findings
        ],
        "errors": result.errors,
    }
    return json.dumps(output, indent=2)
