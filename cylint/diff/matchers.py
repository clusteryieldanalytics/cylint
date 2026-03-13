"""Operation matching — pair base-branch operations with PR-branch operations."""

from __future__ import annotations

from cylint.diff.models import OperationMatch, TrackedOperation

# Variable names too generic for reliable matching
GENERIC_NAMES = frozenset({
    "df", "data", "result", "tmp", "temp", "output", "sdf", "spark_df",
})


def match_by_source_table(
    base_ops: list[TrackedOperation],
    pr_ops: list[TrackedOperation],
) -> list[OperationMatch]:
    """Tier 1: match by same source_table (high confidence)."""
    matches = []
    used_pr: set[int] = set()
    for b in base_ops:
        if b.source_table is None:
            continue
        for i, p in enumerate(pr_ops):
            if i in used_pr:
                continue
            if p.source_table == b.source_table:
                matches.append(OperationMatch(b, p, confidence="high", match_strategy="source_table"))
                used_pr.add(i)
                break
    return matches


def match_by_variable_scope(
    base_ops: list[TrackedOperation],
    pr_ops: list[TrackedOperation],
    already_matched_base: set[int],
    already_matched_pr: set[int],
) -> list[OperationMatch]:
    """Tier 2: match by variable name, excluding generic names (medium confidence)."""
    matches = []
    for i, b in enumerate(base_ops):
        if i in already_matched_base:
            continue
        if b.variable in GENERIC_NAMES:
            continue
        for j, p in enumerate(pr_ops):
            if j in already_matched_pr:
                continue
            if p.variable == b.variable:
                matches.append(OperationMatch(b, p, confidence="medium", match_strategy="variable_scope"))
                already_matched_base.add(i)
                already_matched_pr.add(j)
                break
    return matches


def match_operations(
    base_ops: list[TrackedOperation],
    pr_ops: list[TrackedOperation],
) -> tuple[list[OperationMatch], list[TrackedOperation], list[TrackedOperation]]:
    """Match base operations to PR operations.

    Returns (matches, unmatched_base, unmatched_pr).
    """
    # Tier 1: source table
    tier1 = match_by_source_table(base_ops, pr_ops)
    matched_base = {id(m.base_op) for m in tier1}
    matched_pr = {id(m.pr_op) for m in tier1}

    # Tier 2: variable name + scope
    remaining_base = [op for op in base_ops if id(op) not in matched_base]
    remaining_pr = [op for op in pr_ops if id(op) not in matched_pr]
    tier2_base_idx: set[int] = set()
    tier2_pr_idx: set[int] = set()
    tier2 = match_by_variable_scope(remaining_base, remaining_pr, tier2_base_idx, tier2_pr_idx)
    matched_base.update(id(remaining_base[i]) for i in tier2_base_idx)
    matched_pr.update(id(remaining_pr[i]) for i in tier2_pr_idx)

    all_matches = tier1 + tier2
    unmatched_base = [op for op in base_ops if id(op) not in matched_base]
    unmatched_pr = [op for op in pr_ops if id(op) not in matched_pr]

    return all_matches, unmatched_base, unmatched_pr
