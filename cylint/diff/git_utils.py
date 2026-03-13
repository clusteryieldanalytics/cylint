"""Git utilities for change classification."""

from __future__ import annotations

import subprocess

from cylint.diff.models import ChangedFile


def get_base_source(base_ref: str, file_path: str) -> str | None:
    """Retrieve file content from the base branch.

    Returns None if the file doesn't exist in the base branch (new file).
    Raises on git errors unrelated to file existence.
    """
    try:
        result = subprocess.run(
            ["git", "show", f"{base_ref}:{file_path}"],
            capture_output=True, text=True, check=True,
            timeout=10,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        if "does not exist" in e.stderr or "fatal: path" in e.stderr:
            return None  # new file — no base version
        raise


def get_changed_files(base_ref: str) -> list[ChangedFile]:
    """Get changed files with rename detection.

    Returns list of ChangedFile(status, path, old_path).
    """
    result = subprocess.run(
        ["git", "diff", "--name-status", "-M", f"{base_ref}...HEAD"],
        capture_output=True, text=True, check=True,
        timeout=30,
    )
    files = []
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        parts = line.split("\t")
        status = parts[0][0]  # M, A, D, R (rename has score suffix like R100)
        if status == "R":
            files.append(ChangedFile(status="R", path=parts[2], old_path=parts[1]))
        elif status == "A":
            files.append(ChangedFile(status="A", path=parts[1], old_path=None))
        elif status == "D":
            files.append(ChangedFile(status="D", path=parts[1], old_path=None))
        else:  # M or other
            files.append(ChangedFile(status="M", path=parts[1], old_path=None))
    return files
