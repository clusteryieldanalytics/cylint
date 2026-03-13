"""CI orchestrator — runs the full PR enrichment flow as a single command.

Orchestrates: lint -> cell maps -> diff classify -> enrich -> provenance resolution -> format.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from cylint.ci.cell_map import build_cell_map, is_databricks_notebook
from cylint.ci.comment import FormattedOutput, format_output
from cylint.ci.enrich import (
    EnrichRequest,
    EnrichResponse,
    convert_finding,
    post_enrich,
)
from cylint.engine import LintEngine
from cylint.models import Finding


@dataclass
class CIResult:
    """Structured output from cy ci."""

    findings: list[dict] = field(default_factory=list)
    change_classifications: list[dict] = field(default_factory=list)
    enriched_findings: list[dict] = field(default_factory=list)
    change_findings: list[dict] = field(default_factory=list)
    plan_findings: list[dict] = field(default_factory=list)
    comment: FormattedOutput | None = None
    stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "findings": self.findings,
            "changeClassifications": self.change_classifications,
            "enrichedFindings": self.enriched_findings,
            "changeFindings": self.change_findings,
            "planFindings": self.plan_findings,
            "comment": self.comment.to_dict() if self.comment else None,
            "stats": self.stats,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class CIOrchestrator:
    """Runs the full CI enrichment flow."""

    def __init__(
        self,
        paths: list[str],
        base_ref: str | None = None,
        api_key: str | None = None,
        environment: str | None = None,
        base_url: str = "https://api.clusteryield.app",
        timeout: int = 30,
    ):
        self.paths = paths
        self.base_ref = base_ref
        self.api_key = api_key
        self.environment = environment
        self.base_url = base_url
        self.timeout = timeout

    def run(self) -> CIResult:
        """Execute the full CI flow and return structured results."""
        # Step 1: Run linter
        engine = LintEngine()
        lint_result = engine.lint_paths(self.paths)
        raw_findings = lint_result.findings

        # Step 2: Build cell maps for notebook files
        cell_maps = self._build_cell_maps()

        # Step 3: Convert findings to server format
        server_findings = [
            convert_finding(f, cell_maps.get(f.filepath))
            for f in raw_findings
        ]

        # Step 4: Change classification (when --base-ref is provided)
        classifications: list[dict] = []
        if self.base_ref:
            from cylint.diff import DiffClassifier
            classifier = DiffClassifier(base_ref=self.base_ref)
            raw_classifications = classifier.classify_all(self.paths)
            classifications = [c.to_dict() for c in raw_classifications]

        # Step 5: POST to /enrich (if api_key provided)
        enrich_response: EnrichResponse | None = None
        if self.api_key and self.environment:
            request = EnrichRequest(
                files=[{"path": p} for p in self._get_file_paths()],
                linter_findings=server_findings,
                change_types=classifications,
                environment=self.environment,
            )
            enrich_response = post_enrich(
                request,
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )

        # Step 6: Resolve provenance back to absolute lines
        if enrich_response:
            self._resolve_provenance(enrich_response, cell_maps)

        # Step 7: Format output
        enriched = enrich_response is not None
        linter_for_comment = enrich_response.findings if enriched else server_findings
        plan_for_comment = enrich_response.plan_findings if enriched else []
        change_for_comment = enrich_response.change_findings if enriched else []

        comment = format_output(
            findings=linter_for_comment,
            change_findings=change_for_comment,
            plan_findings=plan_for_comment,
            change_classifications=classifications,
            match_stats=enrich_response.match_stats if enriched else None,
            enriched=enriched,
        )

        return CIResult(
            findings=server_findings,
            change_classifications=classifications,
            enriched_findings=enrich_response.findings if enriched else [],
            change_findings=enrich_response.change_findings if enriched else [],
            plan_findings=enrich_response.plan_findings if enriched else [],
            comment=comment,
            stats={
                "filesScanned": lint_result.files_scanned,
                "filesWithFindings": len({f.filepath for f in raw_findings}),
                "linterFindings": len(raw_findings),
                "changeClassifications": len(classifications),
                "enriched": enriched,
                "matchStats": enrich_response.match_stats if enriched else {},
                "errors": lint_result.errors,
            },
        )

    def _build_cell_maps(self) -> dict[str, dict[str, int]]:
        """Build cell maps for all .py files being linted."""
        cell_maps: dict[str, dict[str, int]] = {}
        for filepath in self._get_file_paths():
            try:
                source = Path(filepath).read_text(encoding="utf-8")
                if is_databricks_notebook(source):
                    cell_maps[filepath] = build_cell_map(source)
            except (FileNotFoundError, OSError):
                continue
        return cell_maps

    def _get_file_paths(self) -> list[str]:
        """Expand paths to individual .py files."""
        files: list[str] = []
        for path in self.paths:
            p = Path(path)
            if p.is_file() and p.suffix == ".py":
                files.append(str(p))
            elif p.is_dir():
                files.extend(str(f) for f in sorted(p.rglob("*.py")))
        return files

    def _resolve_provenance(
        self,
        response: EnrichResponse,
        cell_maps: dict[str, dict[str, int]],
    ) -> None:
        """Convert server-returned provenance back to absolute lines.

        Mutates plan_findings in place.
        """
        from cylint.ci.cell_map import cell_to_absolute

        for pf in response.plan_findings:
            fp = pf.get("triggerCellFingerprint")
            trigger_line = pf.get("triggerLine")
            file_path = pf.get("file")
            if fp and trigger_line and file_path:
                cm = cell_maps.get(file_path)
                if cm:
                    absolute = cell_to_absolute(fp, trigger_line, cm)
                    if absolute:
                        pf["absoluteTriggerLine"] = absolute

            for cl_entry in pf.get("constructionLines", []):
                cl_fp = cl_entry.get("cellFingerprint")
                cl_file = cl_entry.get("file", file_path)
                cm = cell_maps.get(cl_file) if cl_file else None
                if cm and cl_fp:
                    cl_entry["absoluteLines"] = [
                        cell_to_absolute(cl_fp, ln, cm)
                        for ln in cl_entry.get("lines", [])
                        if cell_to_absolute(cl_fp, ln, cm) is not None
                    ]
