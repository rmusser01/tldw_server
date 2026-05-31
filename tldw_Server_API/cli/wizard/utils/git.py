from __future__ import annotations

import shutil

# Git command is fixed to `git status --porcelain` and runs with shell=False.
import subprocess  # nosec B404
from pathlib import Path


def is_git_repo(base: Path) -> bool:
    return (base / ".git").exists()


def changed_or_untracked_files(base: Path) -> list[str]:
    """Return a list of changed or untracked files (best-effort)."""
    if not is_git_repo(base):
        return []
    git = shutil.which("git")
    if not git:
        return []
    try:
        # --porcelain for stable output; include untracked
        out = subprocess.check_output([git, "status", "--porcelain"], cwd=str(base), text=True)  # nosec B603
        files: list[str] = []
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            # Format: XY <path>
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                files.append(parts[1])
        return files
    except (subprocess.CalledProcessError, OSError, FileNotFoundError):
        return []
