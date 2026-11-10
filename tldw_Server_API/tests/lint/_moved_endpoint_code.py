"""Code moved out of ``api/v1/endpoints`` that the endpoint-scoped guards must keep scanning.

Five guards scope themselves to the endpoints tree: the bearer-parsing ratchet, the auth-deps
import boundary, the two AuthNZ call-site guardrails and the raw-SQL ratchet. When a task moves
endpoint code elsewhere, it appends the destination here so the move never leaves that code
unguarded (TASK-13950.20 ruling 9). Entries are exact files, or a directory standing for its
``*.py`` files; a whole shared package is not listed because it holds code the guards were
never written for.

Each guard builds its file list with :func:`scan_files` and names a file with :func:`display`.
Import this module (not ``moved_files``) so a test can patch ``moved_files`` here.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

MOVED_OUT_OF_ENDPOINTS: tuple[str, ...] = ("tldw_Server_API/app/api/v1/utils/pagination.py",)


def moved_files(repo_root: Path = REPO_ROOT) -> list[Path]:
    """The listed files, sorted; a stale entry raises ``FileNotFoundError`` naming it."""
    found: set[Path] = set()
    for entry in MOVED_OUT_OF_ENDPOINTS:
        path = repo_root / entry
        if path.is_dir():
            found.update(path.rglob("*.py"))
        elif path.is_file():
            found.add(path)
        else:
            raise FileNotFoundError(f"MOVED_OUT_OF_ENDPOINTS lists {entry}, which no longer exists")
    return sorted(found)


def scan_files(endpoints_root: Path) -> list[Path]:
    """Every endpoint file, then the moved files."""
    return [*sorted(endpoints_root.rglob("*.py")), *moved_files()]


def display(path: Path) -> str:
    """The repo-relative POSIX path, or ``str(path)`` outside the repo."""
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)
