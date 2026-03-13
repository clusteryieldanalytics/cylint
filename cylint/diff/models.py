"""Data models for change classification."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FilterOp:
    """A recorded .filter() / .where() call."""
    line: int
    expr_hash: str  # SHA-256[:16] of the argument AST subtree


@dataclass
class SelectOp:
    """A recorded .select() call."""
    line: int
    col_count: int | None = None     # number of columns if statically determinable
    col_names: list[str] | None = None  # column name literals if determinable


@dataclass
class JoinOp:
    """A recorded .join() call."""
    line: int
    right_table: str | None = None   # resolved right-side table
    key_expr_hash: str = ""          # hash of the join key AST subtree


@dataclass
class GroupByOp:
    """A recorded .groupBy() call."""
    line: int
    key_expr_hash: str = ""          # hash of the groupBy key arguments AST subtree


@dataclass
class UdfOp:
    """A UDF call on a DataFrame's lineage."""
    line: int
    context: str = "other"           # "filter", "withColumn", "select", or "other"
    name: str | None = None          # UDF function name if resolvable, None for inline/lambda


@dataclass
class WriteOp:
    """A recorded .write.*() call."""
    line: int
    format: str | None = None        # "parquet", "csv", etc.
    target: str | None = None        # table name or path if determinable


@dataclass
class TrackedOperation:
    """Per-DataFrame operation summary — the unit of comparison between branches."""
    variable: str                     # DataFrame variable name
    source_table: str | None = None   # resolved table/path (from spark.table("x"))
    line: int = 0                     # line of DataFrame creation
    filters: list[FilterOp] = field(default_factory=list)
    selects: list[SelectOp] = field(default_factory=list)
    joins: list[JoinOp] = field(default_factory=list)
    broadcasts: list[int] = field(default_factory=list)   # line numbers of F.broadcast()
    groupbys: list[GroupByOp] = field(default_factory=list)
    caches: list[int] = field(default_factory=list)
    udfs: list[UdfOp] = field(default_factory=list)
    writes: list[WriteOp] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "variable": self.variable,
            "sourceTable": self.source_table,
            "line": self.line,
            "filters": [{"line": f.line, "exprHash": f.expr_hash} for f in self.filters],
            "selects": [{"line": s.line, "colCount": s.col_count, "colNames": s.col_names} for s in self.selects],
            "joins": [{"line": j.line, "rightTable": j.right_table, "keyExprHash": j.key_expr_hash} for j in self.joins],
            "broadcasts": self.broadcasts,
            "groupbys": [{"line": g.line, "keyExprHash": g.key_expr_hash} for g in self.groupbys],
            "caches": self.caches,
            "udfs": [{"line": u.line, "context": u.context, "name": u.name} for u in self.udfs],
            "writes": [{"line": w.line, "format": w.format, "target": w.target} for w in self.writes],
        }


@dataclass
class ChangeClassification:
    """Output of the classifier for a single detected change."""
    file: str                         # relative file path
    line: int                         # PR-branch line (or base-branch for removals)
    change_type: str                  # core change type label
    confidence: str                   # "high" | "medium"
    source_table: str | None = None   # resolved table, null if unresolved
    scope: str = ""                   # human-readable context: ".filter()", "spark.table()", etc.
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "file": self.file,
            "line": self.line,
            "changeType": self.change_type,
            "confidence": self.confidence,
            "sourceTable": self.source_table,
            "scope": self.scope,
            "metadata": self.metadata,
        }


@dataclass
class OperationMatch:
    """Internal: matched pair of base/PR operations."""
    base_op: TrackedOperation
    pr_op: TrackedOperation
    confidence: str                   # "high" | "medium" | "low"
    match_strategy: str               # "source_table" | "variable_scope"


@dataclass
class ChangedFile:
    """A file changed between base and PR branches."""
    status: str           # "M" modified, "A" added, "D" deleted, "R" renamed
    path: str             # current path (PR branch)
    old_path: str | None = None  # only set for renames
