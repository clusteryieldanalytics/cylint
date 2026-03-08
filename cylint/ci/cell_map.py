"""Cell map construction and coordinate conversion for Databricks notebooks.

Databricks `.py` notebook exports use ``# COMMAND ----------`` as a cell
separator.  This module builds a mapping from cell fingerprint → start line,
and converts between notebook-absolute and cell-relative coordinates.
"""

import hashlib

SEPARATOR = "# COMMAND ----------"
NOTEBOOK_HEADER = "# Databricks notebook source"


def is_databricks_notebook(source: str) -> bool:
    """Check if a .py file is a Databricks notebook export."""
    return SEPARATOR in source


def build_cell_map(source: str) -> dict[str, int]:
    """Split notebook source into cells, return fingerprint → start_line map.

    Returns:
        Dict mapping cell fingerprint (first 16 hex chars of SHA-256)
        to 1-indexed start line.
    """
    cell_map: dict[str, int] = {}
    current_line = 1  # 1-indexed
    is_first = True
    for segment in source.split(SEPARATOR + "\n"):
        if not segment and current_line == 1:
            # Skip empty leading segment (file starts with separator)
            continue

        # Strip "# Databricks notebook source" header from first segment
        if is_first:
            is_first = False
            lines = segment.splitlines(keepends=True)
            if lines and lines[0].rstrip("\n") == NOTEBOOK_HEADER:
                # Remove header line; advance past it
                current_line += 1
                segment = "".join(lines[1:])

        segment_normalized = normalize_cell_source(segment)
        fp = hashlib.sha256(segment_normalized.encode("utf-8")).hexdigest()[:16]
        cell_map[fp] = current_line
        current_line += segment.count("\n")
        current_line += 1  # for the separator line
    return cell_map

def normalize_cell_source(source: str) -> str:
    """Strip leading/trailing blank lines and trailing whitespace per line."""
    lines = source.strip("\n").splitlines(keepends=True)
    return "".join(lines)

def absolute_to_cell(
    absolute_line: int,
    cell_map: dict[str, int],
) -> tuple[str, int] | None:
    """Convert a notebook-absolute line to (fingerprint, cell-relative line).

    Finds the cell whose start_line is <= absolute_line and is the largest
    such start_line (i.e., the cell that contains the line).

    Returns:
        (cellFingerprint, cellRelativeLine) or None if line is on a separator.
    """
    best_fp: str | None = None
    best_start = 0
    for fp, start in cell_map.items():
        if start <= absolute_line and start > best_start:
            best_fp = fp
            best_start = start
    if best_fp is None:
        return None
    return (best_fp, absolute_line - best_start + 1)


def cell_to_absolute(
    fp: str,
    cell_line: int,
    cell_map: dict[str, int],
) -> int | None:
    """Convert (fingerprint, cell-relative line) back to absolute."""
    start = cell_map.get(fp)
    if start is None:
        return None
    return start + cell_line - 1
