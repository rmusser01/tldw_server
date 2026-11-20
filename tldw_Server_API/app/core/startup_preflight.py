"""Startup preflight validation module.

Runs quick environmental/OS-level checks before the application starts heavy
initialization (database setup, model loading, etc.).  Each check returns a
small dict with ``name``, ``status`` ("ok" / "warn" / "fail"), and an optional
``message``.  Results are aggregated into a :class:`PreflightResult` dataclass.

When the environment variable ``TLDW_PREFLIGHT_STRICT`` is set to a truthy
value (``true``, ``1``, ``yes``), any *failed* check will raise a
``RuntimeError`` to abort startup.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.config_sections.database import database_url_env


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PreflightResult:
    """Aggregated results from all preflight checks."""

    checks: list[dict[str, Any]] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        """Return ``True`` if no check has status ``'fail'``."""
        return all(c.get("status") != "fail" for c in self.checks)

    @property
    def failures(self) -> list[str]:
        """Return ``'name: message'`` strings for every failed check."""
        return [
            f"{c['name']}: {c.get('message', '')}"
            for c in self.checks
            if c.get("status") == "fail"
        ]

    @property
    def warnings(self) -> list[str]:
        """Return ``'name: message'`` strings for every warning check."""
        return [
            f"{c['name']}: {c.get('message', '')}"
            for c in self.checks
            if c.get("status") == "warn"
        ]


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------

def check_ffmpeg_available() -> dict[str, Any]:
    """Check whether ``ffmpeg`` is on ``$PATH``."""
    name = "ffmpeg_available"
    path = shutil.which("ffmpeg")
    if path:
        return {"name": name, "status": "ok", "message": f"found at {path}"}
    return {"name": name, "status": "fail", "message": "ffmpeg not found on $PATH"}


def check_disk_space(min_mb: int = 500, path: str | None = None) -> dict[str, Any]:
    """Warn if free disk space at *path* is below *min_mb* megabytes."""
    name = "disk_space"
    check_path = path or os.getcwd()
    try:
        usage = shutil.disk_usage(check_path)
        free_mb = usage.free // (1024 * 1024)
        if free_mb < min_mb:
            return {
                "name": name,
                "status": "warn",
                "message": f"only {free_mb} MB free (minimum recommended: {min_mb} MB)",
            }
        return {"name": name, "status": "ok", "message": f"{free_mb} MB free"}
    except OSError as exc:
        return {"name": name, "status": "warn", "message": str(exc)}


def check_database_directories(base_dir: str | Path | None = None) -> dict[str, Any]:
    """Verify the ``Databases`` directory exists and is writable, creating it if needed."""
    name = "database_directories"
    if base_dir is not None:
        db_dir = Path(base_dir)
    else:
        # Default: <repo_root>/Databases
        db_dir = Path(__file__).resolve().parent.parent.parent / "Databases"

    try:
        db_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return {"name": name, "status": "fail", "message": f"cannot create {db_dir}: {exc}"}

    if not os.access(db_dir, os.W_OK):
        return {"name": name, "status": "fail", "message": f"{db_dir} is not writable"}

    return {"name": name, "status": "ok", "message": str(db_dir)}


def check_python_dependencies() -> dict[str, Any]:
    """Verify that critical Python packages are importable."""
    name = "python_dependencies"
    required = ["fastapi", "uvicorn", "pydantic", "loguru", "chromadb"]
    missing: list[str] = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        return {
            "name": name,
            "status": "fail",
            "message": f"missing packages: {', '.join(missing)}",
        }
    return {"name": name, "status": "ok"}


def check_database_connectivity() -> dict[str, Any]:
    """Attempt a basic TCP connection test for Postgres DATABASE_URLs.

    This only verifies network reachability, not authentication or schema.
    """
    name = "database_connectivity"
    db_url = database_url_env() or ""
    if not db_url.startswith(("postgresql", "postgres")):
        return {"name": name, "status": "ok", "message": "SQLite (no remote check needed)"}
    try:
        import socket
        from urllib.parse import urlparse
        parsed = urlparse(db_url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 5432
        sock = socket.create_connection((host, port), timeout=5)
        sock.close()
        return {
            "name": name,
            "status": "ok",
            "message": f"Postgres TCP reachable at {host}:{port} (auth not verified)",
        }
    except OSError as exc:
        return {
            "name": name,
            "status": "fail",
            "message": f"Cannot reach Postgres at {host}:{port}: {exc}",
        }
    except Exception as exc:  # noqa: BLE001
        return {"name": name, "status": "warn", "message": f"Could not parse DATABASE_URL: {exc}"}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _is_truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in ("true", "1", "yes")


def run_preflight_checks() -> PreflightResult:
    """Execute all preflight checks and return aggregated results.

    In strict mode (``TLDW_PREFLIGHT_STRICT=true``), raises ``RuntimeError``
    if any check fails.
    """
    result = PreflightResult()

    for check_fn in (
        check_ffmpeg_available,
        check_disk_space,
        check_database_directories,
        check_python_dependencies,
        check_database_connectivity,
    ):
        try:
            result.checks.append(check_fn())
        except Exception as exc:  # noqa: BLE001
            result.checks.append({
                "name": check_fn.__name__,
                "status": "fail",
                "message": f"unexpected error: {exc}",
            })

    # Log individual results
    for check in result.checks:
        status = check.get("status", "unknown")
        msg = check.get("message", "")
        log_line = f"Preflight [{check['name']}]: {status}" + (f" - {msg}" if msg else "")
        if status == "fail":
            logger.error(log_line)
        elif status == "warn":
            logger.warning(log_line)
        else:
            logger.info(log_line)

    # Strict mode
    if _is_truthy(os.environ.get("TLDW_PREFLIGHT_STRICT")) and not result.all_passed:
        raise RuntimeError(
            f"Preflight checks failed in strict mode: {'; '.join(result.failures)}"
        )

    return result
