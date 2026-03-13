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


def _strip_positions(node: ast.AST) -> ast.AST:
    """Deep-copy an AST node with all positional attributes zeroed."""
    node = copy.deepcopy(node)
    for child in ast.walk(node):
        for attr in ("lineno", "col_offset", "end_lineno", "end_col_offset"):
            if hasattr(child, attr):
                setattr(child, attr, 0)
    return node
