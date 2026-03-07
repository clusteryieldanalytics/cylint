"""Core linter engine: parse files, track DataFrames, run rules, collect findings."""

import ast
import os
import warnings
from pathlib import Path

from cylint.models import Finding, LintResult, Severity
from cylint.tracker import (
    FILTER_METHODS,
    ChainInfo,
    DataFrameTracker,
    get_chain_methods,
    is_dataframe_method_chain,
    is_spark_source,
)

# Import rules to trigger registration
import cylint.rules.collect          # noqa: F401
import cylint.rules.udf              # noqa: F401
import cylint.rules.withcolumn       # noqa: F401
import cylint.rules.select_star      # noqa: F401
import cylint.rules.cache            # noqa: F401
import cylint.rules.topandas         # noqa: F401
import cylint.rules.crossjoin        # noqa: F401
import cylint.rules.repartition      # noqa: F401
import cylint.rules.udf_filter       # noqa: F401
import cylint.rules.join_type        # noqa: F401
import cylint.rules.loop_columns     # noqa: F401
import cylint.rules.debug_methods    # noqa: F401
import cylint.rules.coalesce_write   # noqa: F401
import cylint.rules.repeated_actions # noqa: F401
import cylint.rules.nonequi_join     # noqa: F401
import cylint.rules.invalid_escape   # noqa: F401
import cylint.rules.window_partition   # noqa: F401
import cylint.rules.schema_inference   # noqa: F401
import cylint.rules.count_emptiness    # noqa: F401
import cylint.rules.missing_unpersist  # noqa: F401
import cylint.rules.collect_iteration  # noqa: F401

from cylint.rules import get_all_rules


class LintEngine:
    """Main linting engine."""

    def __init__(
        self,
        enabled_rules: dict[str, Severity] | None = None,
        disabled_rules: set[str] | None = None,
        min_severity: Severity = Severity.INFO,
    ):
        all_rules = get_all_rules()

        self.rules = []
        for rule_id, rule_cls in all_rules.items():
            if disabled_rules and rule_id in disabled_rules:
                continue
            severity = None
            if enabled_rules and rule_id in enabled_rules:
                severity = enabled_rules[rule_id]
            rule = rule_cls(severity_override=severity)
            if rule.severity >= min_severity:
                self.rules.append(rule)

        self.min_severity = min_severity

    def lint_file(self, filepath: str) -> list[Finding]:
        """Lint a single Python file and return findings."""
        source = Path(filepath).read_text(encoding="utf-8")
        return self.lint_source(source, filepath)

    def lint_source(self, source: str, filepath: str = "<string>") -> list[Finding]:
        """Lint source code string and return findings."""
        # Capture SyntaxWarnings (e.g. invalid escape sequences) during parse
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", SyntaxWarning)
            try:
                tree = ast.parse(source, filename=filepath)
            except SyntaxError:
                return []

        # Convert captured SyntaxWarnings into CY016 findings
        findings = self._warnings_to_findings(caught, filepath)

        # Phase 1: Build DataFrame tracker
        tracker = self._build_tracker(tree)

        # Phase 2: Run all rules
        for rule in self.rules:
            rule_findings = rule.check(tree, tracker, filepath)
            findings.extend(rule_findings)

        # Sort by line number
        findings.sort(key=lambda f: (f.filepath, f.line, f.col))
        return findings

    def _warnings_to_findings(
        self, caught: list[warnings.WarningMessage], filepath: str
    ) -> list[Finding]:
        """Convert captured SyntaxWarnings to CY016 findings."""
        # Check if CY016 is active
        cy016 = next((r for r in self.rules if r.META.rule_id == "CY016"), None)
        if cy016 is None:
            return []

        findings = []
        for w in caught:
            if not issubclass(w.category, SyntaxWarning):
                continue
            findings.append(Finding(
                rule_id="CY016",
                severity=cy016.severity,
                message=str(w.message),
                filepath=filepath,
                line=w.lineno or 0,
                col=0,
                suggestion="Use a raw string: r'...' instead of '...'",
            ))
        return findings

    def lint_paths(self, paths: list[str], exclude: list[str] | None = None) -> LintResult:
        """Lint one or more files/directories and return aggregated result."""
        result = LintResult()
        exclude_set = set(exclude) if exclude else set()

        for path in paths:
            p = Path(path)
            if p.is_file():
                if p.suffix == ".py" and not self._is_excluded(p, exclude_set):
                    result.files_scanned += 1
                    try:
                        result.findings.extend(self.lint_file(str(p)))
                    except Exception as e:
                        result.errors[str(p)] = str(e)
            elif p.is_dir():
                for py_file in sorted(p.rglob("*.py")):
                    if self._is_excluded(py_file, exclude_set):
                        continue
                    result.files_scanned += 1
                    try:
                        result.findings.extend(self.lint_file(str(py_file)))
                    except Exception as e:
                        result.errors[str(py_file)] = str(e)

        result.findings.sort(key=lambda f: (f.filepath, f.line, f.col))
        return result

    def _build_tracker(self, tree: ast.Module) -> DataFrameTracker:
        """Walk the AST to identify DataFrame variables."""
        tracker = DataFrameTracker()
        self._visit_node(tree, tracker)
        return tracker

    def _visit_node(self, node: ast.AST, tracker: DataFrameTracker):
        """Recursively visit nodes to track DataFrame assignments."""
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Assign):
                self._handle_assign(child, tracker)
            elif isinstance(child, ast.AnnAssign) and child.value:
                self._handle_ann_assign(child, tracker)
            elif isinstance(child, ast.Expr) and isinstance(child.value, ast.Call):
                self._handle_expr_call(child.value, tracker)
            # Recurse into all children
            self._visit_node(child, tracker)

    def _handle_assign(self, node: ast.Assign, tracker: DataFrameTracker):
        """Handle simple assignment: df = spark.read.parquet(...)."""
        value = node.value
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            name = target.id

            # Direct Spark source
            if is_spark_source(value):
                chain_info = ChainInfo(source_line=node.lineno)
                tracker.track(name, node.lineno, chain_info)
                continue

            # Simple alias: df2 = df
            if isinstance(value, ast.Name) and tracker.is_tracked(value.id):
                parent_info = tracker.get_info(value.id)
                chain_info = ChainInfo(
                    source_line=node.lineno,
                    has_filter=parent_info.has_filter if parent_info else False,
                    has_cache=parent_info.has_cache if parent_info else False,
                )
                tracker.track(name, node.lineno, chain_info)
                continue

            # Method chain on tracked DataFrame
            root = is_dataframe_method_chain(value, tracker)
            if root is not None:
                methods = set(get_chain_methods(value))
                has_filter = bool(FILTER_METHODS & methods)
                has_cache = bool({"cache", "persist"} & methods)
                parent_info = tracker.get_info(root)
                chain_info = ChainInfo(
                    source_line=node.lineno,
                    has_filter=has_filter or (parent_info.has_filter if parent_info else False),
                    has_cache=has_cache or (parent_info.has_cache if parent_info else False),
                )
                tracker.track(name, node.lineno, chain_info)
                continue

            # Check for cache/persist in chain
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
                if value.func.attr in ("cache", "persist"):
                    root_name = None
                    if isinstance(value.func.value, ast.Name):
                        root_name = value.func.value.id
                    if root_name and tracker.is_tracked(root_name):
                        parent_info = tracker.get_info(root_name)
                        chain_info = ChainInfo(
                            source_line=node.lineno,
                            has_filter=parent_info.has_filter if parent_info else False,
                            has_cache=True,
                        )
                        tracker.track(name, node.lineno, chain_info)

    def _handle_ann_assign(self, node: ast.AnnAssign, tracker: DataFrameTracker):
        """Handle annotated assignment: df: DataFrame = spark.read..."""
        if not isinstance(node.target, ast.Name) or node.value is None:
            return
        name = node.target.id
        if is_spark_source(node.value):
            tracker.track(name, node.lineno)
        elif isinstance(node.value, ast.Name) and tracker.is_tracked(node.value.id):
            parent_info = tracker.get_info(node.value.id)
            chain_info = ChainInfo(
                source_line=node.lineno,
                has_filter=parent_info.has_filter if parent_info else False,
                has_cache=parent_info.has_cache if parent_info else False,
            )
            tracker.track(name, node.lineno, chain_info)
        else:
            root = is_dataframe_method_chain(node.value, tracker)
            if root is not None:
                methods = set(get_chain_methods(node.value))
                has_filter = bool(FILTER_METHODS & methods)
                has_cache = bool({"cache", "persist"} & methods)
                parent_info = tracker.get_info(root)
                chain_info = ChainInfo(
                    source_line=node.lineno,
                    has_filter=has_filter or (parent_info.has_filter if parent_info else False),
                    has_cache=has_cache or (parent_info.has_cache if parent_info else False),
                )
                tracker.track(name, node.lineno, chain_info)

    def _handle_expr_call(self, call_node: ast.Call, tracker: DataFrameTracker):
        """Handle standalone expression calls like df.cache() or df.persist()."""
        func = call_node.func
        if not isinstance(func, ast.Attribute):
            return
        if func.attr not in ("cache", "persist"):
            return
        if not isinstance(func.value, ast.Name):
            return
        name = func.value.id
        if not tracker.is_tracked(name):
            return
        info = tracker.get_info(name)
        if info and not info.has_cache:
            info.has_cache = True

    def _is_excluded(self, filepath: Path, exclude_set: set[str]) -> bool:
        """Check if a file matches any exclude pattern."""
        path_str = str(filepath)
        for pattern in exclude_set:
            if pattern in path_str:
                return True
        return False
