"""Pre-upgrade validation helpers.

Provides safety checks to run *before* upgrading the tldw_server application:
version compatibility, database integrity, disk space, and Python version.

Results are aggregated into a :class:`PreUpgradeResult` dataclass whose
interface mirrors :class:`PreflightResult` from the startup preflight module.

Can be invoked directly::

    python -m Helper_Scripts.upgrade_helpers          # human-friendly output
    python -m Helper_Scripts.upgrade_helpers --json    # machine-readable
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PreUpgradeResult:
    """Aggregated results from all pre-upgrade checks."""

    checks: list[dict[str, Any]] = field(default_factory=list)

    # -- properties ----------------------------------------------------------

    @property
    def can_proceed(self) -> bool:
        """Return ``True`` if no check has status ``'fail'``."""
        return all(c.get("status") != "fail" for c in self.checks)

    @property
    def summary(self) -> str:
        """Return a human-readable summary like ``'3 passed, 1 warning, 0 failed'``."""
        passed = sum(1 for c in self.checks if c.get("status") == "ok")
        warnings = sum(1 for c in self.checks if c.get("status") == "warn")
        failed = sum(1 for c in self.checks if c.get("status") == "fail")
        return f"{passed} passed, {warnings} warnings, {failed} failed"

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
# Individual check helpers
# ---------------------------------------------------------------------------

def check_version_compatibility(current: str, target: str) -> bool:
    """Return ``True`` if upgrading from *current* to *target* is allowed.

    Allowed transitions:
    - Same version (no-op upgrade)
    - Target is higher than current

    Downgrades (target < current) return ``False``.
    """
    def _parse(v: str) -> tuple[int, ...]:
        # Strip optional leading 'v'
        v = v.lstrip("vV")
        return tuple(int(p) for p in v.split("."))

    return _parse(target) >= _parse(current)


def check_database_integrity(db_path: str) -> dict[str, Any]:
    """Run ``PRAGMA integrity_check`` on a SQLite database file.

    Returns a check dict with keys ``name``, ``status``, ``message``.

    If the file does not exist, returns status ``'ok'`` with a message
    indicating a new install (no database to check).
    """
    name = "db_integrity"
    path = Path(db_path)

    if not path.exists():
        return {
            "name": name,
            "status": "ok",
            "message": f"{path.name}: new install (file not found)",
        }

    try:
        conn = sqlite3.connect(str(path))
        try:
            result = conn.execute("PRAGMA integrity_check").fetchone()
            if result and result[0] == "ok":
                return {
                    "name": name,
                    "status": "ok",
                    "message": f"{path.name}: integrity ok",
                }
            return {
                "name": name,
                "status": "fail",
                "message": f"{path.name}: integrity check returned {result}",
            }
        finally:
            conn.close()
    except Exception as exc:
        return {
            "name": name,
            "status": "fail",
            "message": f"{path.name}: {exc}",
        }


def check_database_write_access(db_path: str) -> dict[str, Any]:
    """Reject migrations when the current account cannot write the store or sidecars."""
    path = Path(db_path)
    required = [(path, os.W_OK), (path.parent, os.W_OK | os.X_OK)]
    for suffix in ("-wal", "-shm", "-journal"):
        sidecar = Path(str(path) + suffix)
        if sidecar.exists():
            required.append((sidecar, os.W_OK))
    for candidate, mode in required:
        if not os.access(candidate, mode):
            return {
                "name": "db_write_access",
                "status": "fail",
                "message": f"{path.name}: migration requires write access to {candidate.name}",
            }
    return {"name": "db_write_access", "status": "ok", "message": f"{path.name}: upgrade write access ok"}


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
                "message": f"only {free_mb} MB free (minimum {min_mb} MB recommended)",
            }
        return {
            "name": name,
            "status": "ok",
            "message": f"{free_mb} MB free",
        }
    except Exception as exc:
        return {
            "name": name,
            "status": "fail",
            "message": str(exc),
        }


def check_python_version(min_version: tuple[int, int] = (3, 12)) -> dict[str, Any]:
    """Fail if the running Python version is below *min_version*."""
    name = "python_version"
    current = sys.version_info[:2]
    if current >= min_version:
        return {
            "name": name,
            "status": "ok",
            "message": f"Python {current[0]}.{current[1]}",
        }
    return {
        "name": name,
        "status": "fail",
        "message": (
            f"Python {current[0]}.{current[1]} found, "
            f"but >= {min_version[0]}.{min_version[1]} required"
        ),
    }


# ---------------------------------------------------------------------------
# Aggregated check runner
# ---------------------------------------------------------------------------

def run_pre_upgrade_checks(
    *,
    db_dir: str = "Databases",
    min_disk_mb: int = 500,
) -> PreUpgradeResult:
    """Run all pre-upgrade checks and return an aggregated result.

    Checks performed:
    1. Disk space (warning if below *min_disk_mb*)
    2. Python version (>= 3.12)
    3. Write access and SQLite integrity for ``.db``, ``.sqlite`` and ``.sqlite3`` files throughout *db_dir*
    """
    result = PreUpgradeResult()

    # 1. Disk space
    result.checks.append(check_disk_space(min_mb=min_disk_mb))

    # 2. Python version
    result.checks.append(check_python_version())

    # 3. Database integrity
    db_path = Path(db_dir)
    if db_path.is_dir():
        db_files = sorted(
            path for path in db_path.rglob("*")
            if path.is_file() and path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}
        )
        if not db_files:
            result.checks.append({
                "name": "db_integrity",
                "status": "ok",
                "message": "no SQLite database files found (fresh install)",
            })
        else:
            for db_file in db_files:
                access = check_database_write_access(str(db_file))
                result.checks.append(access)
                if access["status"] == "ok":
                    result.checks.append(check_database_integrity(str(db_file)))
    else:
        result.checks.append({
            "name": "db_integrity",
            "status": "ok",
            "message": f"{db_dir}: directory not found (fresh install)",
        })

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _main() -> None:
    """Run pre-upgrade checks and print results."""
    as_json = "--json" in sys.argv

    result = run_pre_upgrade_checks()

    if as_json:
        print(json.dumps({
            "can_proceed": result.can_proceed,
            "summary": result.summary,
            "checks": result.checks,
        }, indent=2))
    else:
        print("Pre-upgrade checks")
        print("=" * 40)
        for check in result.checks:
            status = check["status"].upper()
            marker = {"OK": "\033[32m[OK]\033[0m", "WARN": "\033[33m[WARN]\033[0m", "FAIL": "\033[31m[FAIL]\033[0m"}.get(status, f"[{status}]")
            print(f"  {marker} {check['name']}: {check.get('message', '')}")
        print()
        print(f"Summary: {result.summary}")
        print(f"Can proceed: {'yes' if result.can_proceed else 'NO'}")

    sys.exit(0 if result.can_proceed else 1)


if __name__ == "__main__":
    _main()
