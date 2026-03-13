"""Position-independent AST hashing for structural comparison."""

from __future__ import annotations

import ast
import copy
import hashlib


def hash_ast_subtree(node: ast.AST) -> str:
    """Produce a canonical hash of an AST subtree.

    Strips line numbers and column offsets so that equivalent expressions
    at different source locations produce the same hash.
    """
    cleaned = _strip_positions(node)
    canonical = ast.dump(cleaned, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def hash_ast_args(args: list[ast.expr]) -> str:
    """Hash a list of AST argument nodes as a single canonical unit.

    Used for groupBy key columns and join key expressions where
    the argument set is the semantic identity. Order-independent:
    groupBy("a", "b") and groupBy("b", "a") produce the same hash.
    """
    canonical_parts = sorted(
        ast.dump(_strip_positions(arg), annotate_fields=True, include_attributes=False)
        for arg in args
    )
    canonical = "|".join(canonical_parts)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _strip_positions(node: ast.AST) -> ast.AST:
    """Deep-copy an AST node with all positional attributes zeroed."""
    node = copy.deepcopy(node)
    for child in ast.walk(node):
        for attr in ("lineno", "col_offset", "end_lineno", "end_col_offset"):
            if hasattr(child, attr):
                setattr(child, attr, 0)
    return node
