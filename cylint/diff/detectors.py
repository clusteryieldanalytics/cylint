"""Change type detectors — pure functions that take an OperationMatch and produce classifications."""

from __future__ import annotations

from cylint.diff.models import ChangeClassification, OperationMatch, TrackedOperation


def detect_filter_removed(match: OperationMatch, file: str) -> list[ChangeClassification]:
    """Fires when a filter existed in base but is absent in PR."""
    base_hashes = {f.expr_hash for f in match.base_op.filters}
    pr_hashes = {f.expr_hash for f in match.pr_op.filters}

    removed = base_hashes - pr_hashes
    results = []
    for f in match.base_op.filters:
        if f.expr_hash in removed:
            results.append(ChangeClassification(
                file=file,
                line=f.line,
                change_type="filter_removed",
                confidence=match.confidence,
                source_table=match.base_op.source_table,
                scope=".filter()",
                metadata={},
            ))
    return results


def detect_filter_modified(match: OperationMatch, file: str) -> list[ChangeClassification]:
    """Fires when a filter exists in both branches but the expression changed."""
    base_filters = sorted(match.base_op.filters, key=lambda f: f.line)
    pr_filters = sorted(match.pr_op.filters, key=lambda f: f.line)

    results = []
    # Only fire on paired filters with same count
    if len(base_filters) == len(pr_filters):
        for bf, pf in zip(base_filters, pr_filters):
            if bf.expr_hash != pf.expr_hash:
                results.append(ChangeClassification(
                    file=file,
                    line=pf.line,
                    change_type="filter_modified",
                    confidence=match.confidence,
                    source_table=match.pr_op.source_table or match.base_op.source_table,
                    scope=".filter()",
                    metadata={},
                ))
    return results


def detect_filter_added(match: OperationMatch, file: str) -> list[ChangeClassification]:
    """Fires when a new filter appears in PR that wasn't in base."""
    base_hashes = {f.expr_hash for f in match.base_op.filters}
    pr_hashes = {f.expr_hash for f in match.pr_op.filters}

    added = pr_hashes - base_hashes
    results = []
    for f in match.pr_op.filters:
        if f.expr_hash in added:
            results.append(ChangeClassification(
                file=file,
                line=f.line,
                change_type="filter_added",
                confidence=match.confidence,
                source_table=match.pr_op.source_table or match.base_op.source_table,
                scope=".filter()",
                metadata={},
            ))
    return results


def detect_source_changed(match: OperationMatch, file: str) -> list[ChangeClassification]:
    """Fires when the source table/path changed between branches."""
    if match.base_op.source_table is None or match.pr_op.source_table is None:
        return []
    if match.base_op.source_table == match.pr_op.source_table:
        return []

    return [ChangeClassification(
        file=file,
        line=match.pr_op.line,
        change_type="source_changed",
        confidence=match.confidence,
        source_table=match.pr_op.source_table,
        scope="spark.table()",
        metadata={
            "oldTable": match.base_op.source_table,
            "newTable": match.pr_op.source_table,
        },
    )]


def detect_broadcast_hint_removed(match: OperationMatch, file: str) -> list[ChangeClassification]:
    """Fires when an F.broadcast() wrapper is removed."""
    base_count = len(match.base_op.broadcasts)
    pr_count = len(match.pr_op.broadcasts)

    if base_count <= pr_count:
        return []

    # Emit one classification per net removed broadcast
    results = []
    for line in sorted(match.base_op.broadcasts):
        results.append(ChangeClassification(
            file=file,
            line=line,
            change_type="broadcast_hint_removed",
            confidence=match.confidence,
            source_table=match.base_op.source_table,
            scope="F.broadcast()",
            metadata={},
        ))
    return results[:base_count - pr_count]


def detect_broadcast_hint_added(match: OperationMatch, file: str) -> list[ChangeClassification]:
    """Fires when a new F.broadcast() wrapper appears."""
    base_count = len(match.base_op.broadcasts)
    pr_count = len(match.pr_op.broadcasts)

    if pr_count <= base_count:
        return []

    results = []
    for line in sorted(match.pr_op.broadcasts):
        results.append(ChangeClassification(
            file=file,
            line=line,
            change_type="broadcast_hint_added",
            confidence=match.confidence,
            source_table=match.pr_op.source_table or match.base_op.source_table,
            scope="F.broadcast()",
            metadata={},
        ))
    return results[:pr_count - base_count]


def detect_projection_changed(match: OperationMatch, file: str) -> list[ChangeClassification]:
    """Fires when .select() column list changed between branches."""
    base_selects = sorted(match.base_op.selects, key=lambda s: s.line)
    pr_selects = sorted(match.pr_op.selects, key=lambda s: s.line)

    results = []
    if len(base_selects) == len(pr_selects):
        for bs, ps in zip(base_selects, pr_selects):
            changed = False
            if bs.col_count is not None and ps.col_count is not None:
                changed = bs.col_count != ps.col_count
            elif bs.col_names is not None and ps.col_names is not None:
                changed = set(bs.col_names) != set(ps.col_names)
            else:
                continue

            if changed:
                metadata: dict = {}
                if bs.col_count is not None and ps.col_count is not None:
                    metadata["oldColCount"] = bs.col_count
                    metadata["newColCount"] = ps.col_count
                results.append(ChangeClassification(
                    file=file,
                    line=ps.line,
                    change_type="projection_changed",
                    confidence=match.confidence,
                    source_table=match.pr_op.source_table or match.base_op.source_table,
                    scope=".select()",
                    metadata=metadata,
                ))
    return results


def detect_join_key_changed(match: OperationMatch, file: str) -> list[ChangeClassification]:
    """Fires when a join key expression changed between branches."""
    base_joins = sorted(match.base_op.joins, key=lambda j: j.line)
    pr_joins = sorted(match.pr_op.joins, key=lambda j: j.line)

    results = []
    if len(base_joins) == len(pr_joins):
        for bj, pj in zip(base_joins, pr_joins):
            if bj.key_expr_hash != pj.key_expr_hash:
                results.append(ChangeClassification(
                    file=file,
                    line=pj.line,
                    change_type="join_key_changed",
                    confidence=match.confidence,
                    source_table=match.pr_op.source_table or match.base_op.source_table,
                    scope=".join()",
                    metadata={},
                ))
    return results


def detect_aggregation_changed(match: OperationMatch, file: str) -> list[ChangeClassification]:
    """Fires when a groupBy key expression changed between branches."""
    base_gbs = sorted(match.base_op.groupbys, key=lambda g: g.line)
    pr_gbs = sorted(match.pr_op.groupbys, key=lambda g: g.line)

    results = []
    if len(base_gbs) == len(pr_gbs):
        for bg, pg in zip(base_gbs, pr_gbs):
            if bg.key_expr_hash != pg.key_expr_hash:
                results.append(ChangeClassification(
                    file=file,
                    line=pg.line,
                    change_type="aggregation_changed",
                    confidence=match.confidence,
                    source_table=match.pr_op.source_table or match.base_op.source_table,
                    scope=".groupBy()",
                    metadata={},
                ))
    return results


def detect_cache_added(match: OperationMatch, file: str) -> list[ChangeClassification]:
    """Fires when a cache/persist appears in PR but not in base."""
    if len(match.base_op.caches) == 0 and len(match.pr_op.caches) > 0:
        return [ChangeClassification(
            file=file,
            line=match.pr_op.caches[0],
            change_type="cache_added",
            confidence=match.confidence,
            source_table=match.pr_op.source_table or match.base_op.source_table,
            scope=".cache()",
            metadata={},
        )]
    return []


def detect_cache_removed(match: OperationMatch, file: str) -> list[ChangeClassification]:
    """Fires when a cache/persist existed in base but is absent in PR."""
    if len(match.base_op.caches) > 0 and len(match.pr_op.caches) == 0:
        return [ChangeClassification(
            file=file,
            line=match.base_op.caches[0],
            change_type="cache_removed",
            confidence=match.confidence,
            source_table=match.base_op.source_table,
            scope=".cache()",
            metadata={},
        )]
    return []


def detect_udf_added(match: OperationMatch, file: str) -> list[ChangeClassification]:
    """Fires when a UDF appears in PR lineage that wasn't in base."""
    base_udf_set = {(u.context, u.name) for u in match.base_op.udfs}
    pr_udf_set = {(u.context, u.name) for u in match.pr_op.udfs}

    added = pr_udf_set - base_udf_set
    results = []
    for u in match.pr_op.udfs:
        if (u.context, u.name) in added:
            results.append(ChangeClassification(
                file=file,
                line=u.line,
                change_type="udf_added",
                confidence=match.confidence,
                source_table=match.pr_op.source_table or match.base_op.source_table,
                scope=f"UDF in .{u.context}()" if u.context != "other" else "UDF",
                metadata={"udfContext": u.context},
            ))
            added.discard((u.context, u.name))
    return results


def detect_udf_removed(match: OperationMatch, file: str) -> list[ChangeClassification]:
    """Fires when a UDF in base lineage is absent in PR."""
    base_udf_set = {(u.context, u.name) for u in match.base_op.udfs}
    pr_udf_set = {(u.context, u.name) for u in match.pr_op.udfs}

    removed = base_udf_set - pr_udf_set
    results = []
    for u in match.base_op.udfs:
        if (u.context, u.name) in removed:
            results.append(ChangeClassification(
                file=file,
                line=u.line,
                change_type="udf_removed",
                confidence=match.confidence,
                source_table=match.base_op.source_table,
                scope=f"UDF in .{u.context}()" if u.context != "other" else "UDF",
                metadata={"udfContext": u.context},
            ))
            removed.discard((u.context, u.name))
    return results


# All detectors run on every match
DETECTORS = [
    # Core 7
    detect_filter_removed,
    detect_filter_modified,
    detect_filter_added,
    detect_source_changed,
    detect_broadcast_hint_removed,
    detect_broadcast_hint_added,
    detect_projection_changed,
    # Extended 6
    detect_join_key_changed,
    detect_aggregation_changed,
    detect_cache_added,
    detect_cache_removed,
    detect_udf_added,
    detect_udf_removed,
]


def classify_changes(
    matches: list[OperationMatch],
    unmatched_base: list[TrackedOperation],
    file: str,
) -> list[ChangeClassification]:
    """Run all detectors and emit operation_removed for unmatched base ops."""
    results: list[ChangeClassification] = []

    for match in matches:
        if match.confidence == "low":
            continue
        for detector in DETECTORS:
            results.extend(detector(match, file))

    # Emit operation_removed for unmatched base operations (deleted code)
    for op in unmatched_base:
        results.append(ChangeClassification(
            file=file,
            line=op.line,
            change_type="operation_removed",
            confidence="medium",
            source_table=op.source_table,
            scope=f"{op.variable} (deleted)",
            metadata={},
        ))

    # Deduplicate by (file, line, change_type)
    seen: set[tuple[str, int, str]] = set()
    deduped = []
    for c in results:
        key = (c.file, c.line, c.change_type)
        if key not in seen:
            seen.add(key)
            deduped.append(c)

    return deduped
