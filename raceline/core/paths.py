"""Locate the project root no matter where the user runs commands from."""

from __future__ import annotations

import os
from pathlib import Path

_MARKERS = ("pyproject.toml", "pipeline", "raceline")


def find_project_root(start: Path | None = None) -> Path:
    """Walk upward from *start* (or cwd) until repo markers are found."""
    cur = (start or Path.cwd()).resolve()
    for directory in (cur, *cur.parents):
        if all((directory / m).exists() for m in _MARKERS):
            return directory
    raise FileNotFoundError(
        "Cannot find the raceline project root.\n"
        "  cd into the cloned GithubUser712 directory, then rerun.\n"
        "  Expected to find pyproject.toml, pipeline/, and raceline/ together.")


def resolve_path(path: str | Path, root: Path | None = None) -> Path:
    """Resolve *path* relative to project root if not absolute."""
    p = Path(path).expanduser()
    if p.is_absolute():
        return p.resolve()
    base = root or find_project_root()
    return (base / p).resolve()


def chdir_to_project() -> Path:
    """Change working directory to project root; return that path."""
    root = find_project_root()
    os.chdir(root)
    return root
