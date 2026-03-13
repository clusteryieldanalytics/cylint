"""DiffClassifier — orchestrates change classification for PySpark files."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

from cylint.diff.ast_hash import hash_ast_args, hash_ast_subtree
from cylint.diff.detectors import classify_changes
from cylint.diff.git_utils import ChangedFile, get_base_source, get_changed_files
from cylint.diff.matchers import match_operations
from cylint.diff.models import (
    ChangeClassification,
    FilterOp,
    GroupByOp,
    JoinOp,
    SelectOp,
    TrackedOperation,
    UdfOp,
    WriteOp,
)
from cylint.tracker import (
    DataFrameTracker,
    is_dataframe_method_chain,
    is_spark_source,
)


class DiffClassifier:
    """Orchestrates change classification for a set of changed files."""

    def __init__(self, base_ref: str):
        self.base_ref = base_ref

    def classify_file(
        self, file_path: str, old_path: str | None = None
    ) -> list[ChangeClassification]:
        """Classify changes in a single file.

        Args:
            file_path: Path to the file in the PR branch (working copy).
            old_path: Path in the base branch (for renames). Defaults to file_path.
        """
        base_path = old_path or file_path
        base_source = get_base_source(self.base_ref, base_path)
        try:
            pr_source = Path(file_path).read_text(encoding="utf-8")
        except FileNotFoundError:
            pr_source = None

        # Edge cases
        if base_source is None and pr_source is not None:
            return []  # new file — no classification
        if base_source is not None and pr_source is None:
            # deleted file — emit operation_removed for all tracked ops
            base_ops = extract_operations(base_source)
            return [
                ChangeClassification(
                    file=file_path,
                    line=op.line,
                    change_type="operation_removed",
                    confidence="medium",
                    source_table=op.source_table,
                    scope=f"{op.variable} (deleted)",
                    metadata={},
                )
                for op in base_ops
            ]
        if base_source is None and pr_source is None:
            return []

        # Normal case: both versions exist
        try:
            base_ops = extract_operations(base_source)
        except SyntaxError:
            print(f"Warning: cannot parse base version of {base_path}", file=sys.stderr)
            return []
        try:
            pr_ops = extract_operations(pr_source)
        except SyntaxError:
            print(f"Warning: cannot parse PR version of {file_path}", file=sys.stderr)
            return []

        matches, unmatched_base, _ = match_operations(base_ops, pr_ops)
        return classify_changes(matches, unmatched_base, file_path)

    def classify_all(self, paths: list[str]) -> list[ChangeClassification]:
        """Classify changes across all changed files in the given paths."""
        changed_files = get_changed_files(self.base_ref)
        # Filter to .py files within the specified paths
        changed_files = [
            f for f in changed_files
            if f.path.endswith(".py") and self._matches_paths(f.path, paths)
        ]

        all_classifications: list[ChangeClassification] = []
        for cf in changed_files:
            if cf.status == "A":
                continue  # new file — standard linter covers it
            classifications = self.classify_file(cf.path, old_path=cf.old_path)
            all_classifications.extend(classifications)
        return all_classifications

    def _matches_paths(self, file_path: str, paths: list[str]) -> bool:
        """Check if file_path is within any of the specified paths."""
        for p in paths:
            if file_path.startswith(p) or file_path == p:
                return True
        return False


# ---------------------------------------------------------------------------
# Operation extraction — parse source and build TrackedOperations
# ---------------------------------------------------------------------------

def extract_operations(source: str) -> list[TrackedOperation]:
    """Parse PySpark source and extract per-variable TrackedOperations.

    This is the bridge between the AST and the change classifier. It reuses
    the tracker's source-detection helpers but records richer operation data
    (filter hashes, select column counts, broadcast lines, etc.).
    """
    tree = ast.parse(source)
    tracker = DataFrameTracker()
    ops: dict[str, TrackedOperation] = {}

    _walk(tree, tracker, ops)
    # Second pass: detect UDF usage on tracked DataFrames
    udf_names = _collect_udf_names(tree)
    if udf_names:
        _record_udf_usage(tree, tracker, ops, udf_names)
    return list(ops.values())


def _ensure_op(
    ops: dict[str, TrackedOperation], name: str, line: int
) -> TrackedOperation:
    if name not in ops:
        ops[name] = TrackedOperation(variable=name, line=line)
    return ops[name]


def _inherit_ops(target: TrackedOperation, parent: TrackedOperation):
    """Copy parent's source table and operations into target.

    Safe for self-referential reassignment (df = df.filter(...)) —
    snapshots the parent lists before extending.
    """
    target.source_table = parent.source_table
    if target is parent:
        # Self-reassignment: nothing to inherit, ops already there
        return
    target.filters.extend(parent.filters)
    target.selects.extend(parent.selects)
    target.joins.extend(parent.joins)
    target.broadcasts.extend(parent.broadcasts)
    target.groupbys.extend(parent.groupbys)
    target.caches.extend(parent.caches)
    target.udfs.extend(parent.udfs)
    target.writes.extend(parent.writes)


def _extract_source_table(node: ast.Call) -> str | None:
    """Extract the table name/path from spark.table("x"), spark.read.parquet("x"),
    or spark.read.format("csv").load("x")."""
    func = node.func
    # spark.table("x")
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        if func.attr == "table" and node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                return arg.value
    # spark.read.parquet("x"), spark.read.csv("x"), etc.
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Attribute):
        if isinstance(func.value.value, ast.Name):
            if func.value.attr == "read" and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    return arg.value
    # spark.read.format("csv").load("x")
    if isinstance(func, ast.Attribute) and func.attr == "load" and node.args:
        arg = node.args[0]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value
    return None


def _walk(node: ast.AST, tracker: DataFrameTracker, ops: dict[str, TrackedOperation]):
    """Walk the AST to track DataFrames and record operations."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.Assign):
            _handle_assign(child, tracker, ops)
        elif isinstance(child, ast.AnnAssign) and child.value:
            _handle_ann_assign(child, tracker, ops)
        elif isinstance(child, ast.Expr):
            _handle_expr(child, tracker, ops)
        _walk(child, tracker, ops)


def _handle_assign(node: ast.Assign, tracker: DataFrameTracker, ops: dict[str, TrackedOperation]):
    """Handle assignment: track DataFrame + record operations."""
    value = node.value
    for target in node.targets:
        if not isinstance(target, ast.Name):
            continue
        name = target.id

        # Direct spark source
        if is_spark_source(value):
            from cylint.tracker import ChainInfo
            tracker.track(name, node.lineno, ChainInfo(source_line=node.lineno))
            op = _ensure_op(ops, name, node.lineno)
            if isinstance(value, ast.Call):
                op.source_table = _extract_source_table(value)
            # Record operations in the chain (e.g. spark.table("x").filter(...))
            _record_chain_ops(value, op)
            continue

        # Simple alias: df2 = df
        if isinstance(value, ast.Name) and tracker.is_tracked(value.id):
            from cylint.tracker import ChainInfo
            parent_info = tracker.get_info(value.id)
            tracker.track(name, node.lineno, ChainInfo(
                source_line=node.lineno,
                has_filter=parent_info.has_filter if parent_info else False,
                has_cache=parent_info.has_cache if parent_info else False,
            ))
            # Inherit parent's source table and operations
            if value.id in ops:
                parent_op = ops[value.id]
                op = _ensure_op(ops, name, node.lineno)
                _inherit_ops(op, parent_op)
            continue

        # Method chain on tracked DataFrame
        root = is_dataframe_method_chain(value, tracker)
        if root is not None:
            from cylint.tracker import ChainInfo, FILTER_METHODS, get_chain_methods
            methods = set(get_chain_methods(value))
            has_filter = bool(FILTER_METHODS & methods)
            has_cache = bool({"cache", "persist"} & methods)
            parent_info = tracker.get_info(root)
            tracker.track(name, node.lineno, ChainInfo(
                source_line=node.lineno,
                has_filter=has_filter or (parent_info.has_filter if parent_info else False),
                has_cache=has_cache or (parent_info.has_cache if parent_info else False),
            ))
            op = _ensure_op(ops, name, node.lineno)
            if root in ops:
                _inherit_ops(op, ops[root])
            # Record new operations from this chain
            _record_chain_ops(value, op)
            continue

        # cache/persist on tracked DataFrame
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
            if value.func.attr in ("cache", "persist"):
                root_name = None
                if isinstance(value.func.value, ast.Name):
                    root_name = value.func.value.id
                if root_name and tracker.is_tracked(root_name):
                    from cylint.tracker import ChainInfo
                    parent_info = tracker.get_info(root_name)
                    tracker.track(name, node.lineno, ChainInfo(
                        source_line=node.lineno,
                        has_filter=parent_info.has_filter if parent_info else False,
                        has_cache=True,
                    ))
                    op = _ensure_op(ops, name, node.lineno)
                    if root_name in ops:
                        _inherit_ops(op, ops[root_name])
                    op.caches.append(node.lineno)
                    continue

        # Fallback: unrecognized chain ending in a terminal method (e.g. count(),
        # collect()). Still record intermediate ops on the root DataFrame.
        root_name = _find_root_name_for_call(value, tracker)
        if root_name and root_name in ops:
            _record_chain_ops(value, ops[root_name])


def _handle_ann_assign(node: ast.AnnAssign, tracker: DataFrameTracker, ops: dict[str, TrackedOperation]):
    """Handle annotated assignment: df: DataFrame = spark.table(...) / df.filter(...)."""
    if not isinstance(node.target, ast.Name) or node.value is None:
        return
    name = node.target.id
    value = node.value

    if is_spark_source(value):
        from cylint.tracker import ChainInfo
        tracker.track(name, node.lineno, ChainInfo(source_line=node.lineno))
        op = _ensure_op(ops, name, node.lineno)
        if isinstance(value, ast.Call):
            op.source_table = _extract_source_table(value)
        _record_chain_ops(value, op)
    elif isinstance(value, ast.Name) and tracker.is_tracked(value.id):
        from cylint.tracker import ChainInfo
        parent_info = tracker.get_info(value.id)
        tracker.track(name, node.lineno, ChainInfo(
            source_line=node.lineno,
            has_filter=parent_info.has_filter if parent_info else False,
            has_cache=parent_info.has_cache if parent_info else False,
        ))
        if value.id in ops:
            op = _ensure_op(ops, name, node.lineno)
            _inherit_ops(op, ops[value.id])
    else:
        root = is_dataframe_method_chain(value, tracker)
        if root is not None:
            from cylint.tracker import ChainInfo, FILTER_METHODS, get_chain_methods
            methods = set(get_chain_methods(value))
            has_filter = bool(FILTER_METHODS & methods)
            has_cache = bool({"cache", "persist"} & methods)
            parent_info = tracker.get_info(root)
            tracker.track(name, node.lineno, ChainInfo(
                source_line=node.lineno,
                has_filter=has_filter or (parent_info.has_filter if parent_info else False),
                has_cache=has_cache or (parent_info.has_cache if parent_info else False),
            ))
            op = _ensure_op(ops, name, node.lineno)
            if root in ops:
                _inherit_ops(op, ops[root])
            _record_chain_ops(value, op)


def _handle_expr(node: ast.Expr, tracker: DataFrameTracker, ops: dict[str, TrackedOperation]):
    """Handle expression statements (e.g. df.cache(), df.write.parquet(...))."""
    if not isinstance(node.value, ast.Call):
        return
    call = node.value
    if not isinstance(call.func, ast.Attribute):
        return

    attr = call.func.attr

    # df.cache() / df.persist() — update tracker, then record via chain ops below
    if attr in ("cache", "persist") and isinstance(call.func.value, ast.Name):
        name = call.func.value.id
        if tracker.is_tracked(name):
            info = tracker.get_info(name)
            if info and not info.has_cache:
                info.has_cache = True

    # df.write.parquet(...) / df.write.saveAsTable(...)
    if isinstance(call.func.value, ast.Attribute):
        inner = call.func.value
        if inner.attr == "write" and isinstance(inner.value, ast.Name):
            df_name = inner.value.id
            if tracker.is_tracked(df_name) and df_name in ops:
                target = None
                if call.args:
                    arg = call.args[0]
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        target = arg.value
                ops[df_name].writes.append(WriteOp(
                    line=node.lineno,
                    format=attr if attr != "save" else None,
                    target=target,
                ))

    # Record operations on tracked DataFrames (filter, select, join, groupBy, cache, etc.)
    root_name = _find_root_name_for_call(call, tracker)
    if root_name and root_name in ops:
        _record_chain_ops(call, ops[root_name])


def _find_root_name_for_call(node: ast.AST, tracker: DataFrameTracker) -> str | None:
    """Find the tracked DataFrame name at the root of a call chain."""
    if isinstance(node, ast.Name):
        return node.id if tracker.is_tracked(node.id) else None
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute):
            return _find_root_name_for_call(node.func.value, tracker)
    if isinstance(node, ast.Attribute):
        return _find_root_name_for_call(node.value, tracker)
    return None


def _record_chain_ops(node: ast.AST, op: TrackedOperation):
    """Walk a method chain and record operations on the TrackedOperation."""
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute):
            attr = func.attr
            lineno = getattr(node, "lineno", 0)

            if attr in ("filter", "where"):
                expr_hash = ""
                if node.args:
                    expr_hash = hash_ast_subtree(node.args[0])
                elif node.keywords:
                    expr_hash = hash_ast_subtree(node.keywords[0].value)
                op.filters.append(FilterOp(line=lineno, expr_hash=expr_hash))

            elif attr == "select":
                col_count = len(node.args)
                col_names = _extract_col_names(node.args)
                op.selects.append(SelectOp(line=lineno, col_count=col_count, col_names=col_names))

            elif attr == "join":
                key_hash = ""
                # Join key is typically the 2nd argument — use order-independent hash
                if len(node.args) >= 2:
                    key_arg = node.args[1]
                    # If key is a single expression, wrap in list for hash_ast_args
                    key_hash = hash_ast_args([key_arg])
                op.joins.append(JoinOp(line=lineno, key_expr_hash=key_hash))
                # Check for F.broadcast() in join arguments
                for arg in node.args:
                    _check_broadcast(arg, op, lineno)

            elif attr == "groupBy" or attr == "groupby":
                key_hash = hash_ast_args(node.args) if node.args else ""
                op.groupbys.append(GroupByOp(line=lineno, key_expr_hash=key_hash))

            elif attr in ("cache", "persist"):
                op.caches.append(lineno)

            # Write operations: .parquet(...), .saveAsTable(...), .csv(...), etc.
            # These appear as df.write.parquet("path") — attr is the format method
            elif isinstance(func.value, ast.Attribute) and func.value.attr == "write":
                target = None
                if node.args:
                    arg = node.args[0]
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        target = arg.value
                op.writes.append(WriteOp(
                    line=lineno,
                    format=attr if attr != "save" else None,
                    target=target,
                ))

            # Recurse into the chain (the receiver of the method call)
            _record_chain_ops(func.value, op)

    elif isinstance(node, ast.Attribute):
        _record_chain_ops(node.value, op)


def _check_broadcast(node: ast.AST, op: TrackedOperation, fallback_line: int):
    """Check if a node is F.broadcast(x) and record it."""
    if isinstance(node, ast.Call):
        func = node.func
        # F.broadcast(x) or functions.broadcast(x)
        if isinstance(func, ast.Attribute) and func.attr == "broadcast":
            line = getattr(node, "lineno", fallback_line)
            op.broadcasts.append(line)
        # broadcast(x) — imported directly
        elif isinstance(func, ast.Name) and func.id == "broadcast":
            line = getattr(node, "lineno", fallback_line)
            op.broadcasts.append(line)


def _extract_col_names(args: list[ast.expr]) -> list[str] | None:
    """Extract string literal column names from .select() arguments."""
    names = []
    for arg in args:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            names.append(arg.value)
        else:
            return None  # non-literal — can't determine statically
    return names if names else None


# ---------------------------------------------------------------------------
# UDF tracking — second-pass detection of UDF usage on tracked DataFrames
# ---------------------------------------------------------------------------

def _collect_udf_names(tree: ast.Module) -> set[str]:
    """Walk the AST to find all variable/function names registered as UDFs."""
    udf_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and isinstance(node.value, ast.Call):
                    func = node.value.func
                    if isinstance(func, ast.Name) and func.id in ("udf", "pandas_udf"):
                        udf_names.add(target.id)
                    elif isinstance(func, ast.Attribute) and func.attr in ("udf", "pandas_udf"):
                        udf_names.add(target.id)
        if isinstance(node, ast.FunctionDef):
            for dec in node.decorator_list:
                if isinstance(dec, ast.Name) and dec.id in ("udf", "pandas_udf"):
                    udf_names.add(node.name)
                elif isinstance(dec, ast.Attribute) and dec.attr in ("udf", "pandas_udf"):
                    udf_names.add(node.name)
                elif isinstance(dec, ast.Call):
                    inner = dec.func
                    if isinstance(inner, ast.Name) and inner.id in ("udf", "pandas_udf"):
                        udf_names.add(node.name)
                    elif isinstance(inner, ast.Attribute) and inner.attr in ("udf", "pandas_udf"):
                        udf_names.add(node.name)
    return udf_names


def _is_udf_call(node: ast.expr, udf_names: set[str]) -> bool:
    """True if node is a UDF invocation."""
    if isinstance(node, ast.Lambda):
        return True
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Call):
        inner = func.func
        if isinstance(inner, ast.Name) and inner.id in ("udf", "pandas_udf"):
            return True
        if isinstance(inner, ast.Attribute) and inner.attr in ("udf", "pandas_udf"):
            return True
    if isinstance(func, ast.Name) and func.id in udf_names:
        return True
    return False


def _resolve_udf_name(node: ast.Call, udf_names: set[str]) -> str | None:
    """Extract the UDF function name if it's a named variable."""
    func = node.func
    if isinstance(func, ast.Name) and func.id in udf_names:
        return func.id
    return None


def _record_udf_usage(
    tree: ast.Module,
    tracker: DataFrameTracker,
    ops: dict[str, TrackedOperation],
    udf_names: set[str],
):
    """Walk the AST to find UDF calls on tracked DataFrames and record them.

    Records UDFs on the assignment target variable (if the call is part of
    an assignment) or the root DataFrame (for standalone expressions).
    """
    # Build a map from call node id → assignment target name
    assign_targets: dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and isinstance(node.value, ast.Call):
                    # Walk the entire chain to tag all Call nodes in it
                    _tag_calls_with_target(node.value, target.id, assign_targets)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and isinstance(node.value, ast.Call):
                _tag_calls_with_target(node.value, node.target.id, assign_targets)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue

        attr = func.attr
        context = None
        args_to_check: list[ast.expr] = []

        if attr in ("filter", "where"):
            context = "filter"
            args_to_check = list(node.args)
        elif attr == "withColumn":
            context = "withColumn"
            if len(node.args) >= 2:
                args_to_check = [node.args[1]]
        elif attr == "select":
            context = "select"
            args_to_check = list(node.args)
        else:
            continue

        if not args_to_check:
            continue

        root = _find_root_name_for_call(node, tracker)
        if root is None or root not in ops:
            continue

        for arg in args_to_check:
            if _is_udf_call(arg, udf_names):
                udf_name = _resolve_udf_name(arg, udf_names) if isinstance(arg, ast.Call) else None
                udf_op = UdfOp(
                    line=getattr(node, "lineno", 0),
                    context=context,
                    name=udf_name,
                )
                # Record on assignment target if available, else root
                target_name = assign_targets.get(id(node))
                if target_name and target_name in ops:
                    ops[target_name].udfs.append(udf_op)
                else:
                    ops[root].udfs.append(udf_op)


def _tag_calls_with_target(node: ast.AST, target: str, mapping: dict[int, str]):
    """Tag all Call nodes in a method chain with the assignment target name."""
    if isinstance(node, ast.Call):
        mapping[id(node)] = target
        if isinstance(node.func, ast.Attribute):
            _tag_calls_with_target(node.func.value, target, mapping)
    elif isinstance(node, ast.Attribute):
        _tag_calls_with_target(node.value, target, mapping)
