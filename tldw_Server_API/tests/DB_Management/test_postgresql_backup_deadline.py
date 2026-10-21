"""Bounded PostgreSQL tool ownership; only process boundaries are mocked."""

import subprocess  # nosec B404 -- controlled process outcomes, no launches.
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management import DB_Backups as backups

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def controlled_tool_clock(monkeypatch):
    monkeypatch.setattr(backups.time, "monotonic", lambda: 0)


@pytest.mark.parametrize("stuck_cleanup", [False, True])
def test_expired_tool_terminates_only_its_child_without_force(monkeypatch, stuck_cleanup):
    calls = []

    class Child:
        pid = 12345

        def communicate(self, timeout):
            calls.append(("wait", timeout))
            if len(calls) == 1 or (stuck_cleanup and len(calls) == 3):
                raise subprocess.TimeoutExpired("pg_restore", timeout)
            return "", ""

        def terminate(self):
            calls.append(("terminate", self.pid))

        def kill(self):
            pytest.fail("force termination is prohibited")

    monkeypatch.setattr(backups.subprocess, "Popen", lambda *args, **kwargs: Child())
    with pytest.raises(TimeoutError, match="target must remain offline") as failure:
        backups._run_postgres_tool(["/fixture/pg_restore"], env={}, timeout_seconds=3)
    expected = [("wait", 3), ("terminate", 12345), ("wait", 15)] + ([("wait", 15)] if stuck_cleanup else [])
    assert calls == expected  # nosec B101 -- one owned graceful stop and bounded drain only.
    assert failure.value.process_running is False and failure.value.cleanup_exceeded is stuck_cleanup  # nosec B101 -- preserve uncertain child ownership explicitly.


def test_bounded_tool_cannot_inherit_target_overrides_or_sql_options(monkeypatch):
    captured = []

    class Child:
        returncode = 0

        def communicate(self, timeout):
            return "toc", ""

    def launch(command, **kwargs):
        captured.append(kwargs["env"])
        return Child()

    monkeypatch.setattr(backups.subprocess, "Popen", launch)
    result = backups._run_postgres_tool(
        ["/fixture/pg_dump"],
        env={
            "PATH": "/fixture",
            "PGHOSTADDR": "other",
            "PGOPTIONS": "-c role=owner",
            "PGPASSWORD": "fixture",
            "PGCONNECT_TIMEOUT": "3",
            "PGSSLMODE": "require",
        },
        timeout_seconds=10,
    )
    assert result.returncode == 0 and result.stdout == "toc"  # nosec B101 -- return the actual child completion.
    safe = captured == [{"PATH": "/fixture", "PGPASSWORD": "fixture", "PGCONNECT_TIMEOUT": "3", "PGSSLMODE": "require"}]
    assert safe  # nosec B101 -- avoid credential output while checking explicit connection scope.


def test_archive_validation_invokes_pg_restore_list_without_a_database(monkeypatch, tmp_path):
    source = tmp_path / "database.dump"
    source.write_bytes(b"PGDMP" + bytes(64))
    seen = []
    monkeypatch.setattr(backups.shutil, "which", lambda _: "/fixture/pg_restore")

    def run(command, *, env, timeout_seconds):
        seen.append((command, timeout_seconds))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(backups, "_run_postgres_tool", run, raising=False)
    assert backups.validate_postgres_backup(str(source), timeout_seconds=7) == "ok"  # nosec B101 -- structural tool validation is required before compatibility.
    assert seen == [(["/fixture/pg_restore", "--list", "--", str(source)], 7)]  # nosec B101 -- never connect or restore during validation.


@pytest.mark.parametrize("seconds", [0, -1, float("inf"), float("nan")])
def test_invalid_deadline_never_launches_a_tool(monkeypatch, seconds):
    monkeypatch.setattr(
        backups.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("invalid deadline launched process")
    )
    with pytest.raises(ValueError):
        backups._run_postgres_tool(["/fixture/pg_dump"], env={}, timeout_seconds=seconds)


@pytest.mark.parametrize("failure_type", [ValueError, KeyboardInterrupt])
def test_non_timeout_tool_failure_still_terminates_and_reaps_child(monkeypatch, failure_type):
    calls = []

    class Child:
        pid = 12345
        returncode = -15

        def communicate(self, timeout):
            calls.append(("wait", timeout))
            if len(calls) == 1:
                raise failure_type("controlled communication failure")
            return "", ""

        def terminate(self):
            calls.append(("terminate", self.pid))

    monkeypatch.setattr(backups.subprocess, "Popen", lambda *a, **kw: Child())
    with pytest.raises(failure_type):
        backups._run_postgres_tool(["/fixture/pg_restore"], env={}, timeout_seconds=3)
    assert calls == [("wait", 3), ("terminate", 12345), ("wait", 15)]  # nosec B101 -- exceptional unwinding cannot drop an active child.


def test_late_natural_tool_exit_cannot_be_reported_as_success(monkeypatch):
    now = [0]

    class Child:
        pid = 12345
        returncode = 0

        def communicate(self, timeout):
            now[0] = 4
            return "late", ""

        def terminate(self):
            pytest.fail("already exited tool was signalled")

    monkeypatch.setattr(backups.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(backups.subprocess, "Popen", lambda *a, **kw: Child())
    with pytest.raises(TimeoutError) as failure:
        backups._run_postgres_tool(["/fixture/pg_restore"], env={}, timeout_seconds=3)
    assert not failure.value.process_running  # nosec B101 -- a reaped late completion is still a failed deadline.


@pytest.mark.parametrize("outcome", ["nonzero", "communication_error", "interruption"])
def test_failed_database_dump_removes_only_its_incomplete_artifact(tmp_path, monkeypatch, outcome):
    from pathlib import Path

    from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig

    directory = tmp_path / "backups"
    directory.mkdir()
    previous = directory / "kept.dump"
    previous.write_bytes(b"previous functional backup")
    monkeypatch.setenv("TLDW_DB_BACKUP_PATH", str(directory))
    monkeypatch.setattr(backups.shutil, "which", lambda _: "/fixture/pg_dump")

    def rejected(command, **kwargs):
        Path(command[command.index("-f") + 1]).write_bytes(b"PGDMP incomplete")
        if outcome == "communication_error":
            raise ValueError("controlled I/O failure after reap")
        if outcome == "interruption":
            raise KeyboardInterrupt("controlled interruption after reap")
        return SimpleNamespace(returncode=1, stderr="complete-row access denied")

    monkeypatch.setattr(backups, "_run_postgres_tool", rejected)
    backend = SimpleNamespace(
        backend_type=BackendType.POSTGRESQL, config=DatabaseConfig(backend_type=BackendType.POSTGRESQL)
    )
    if outcome == "interruption":
        with pytest.raises(KeyboardInterrupt):
            backups.create_postgres_backup(backend, backup_dir=str(directory), label="postgresql", timeout_seconds=30)
    else:
        result = backups.create_postgres_backup(backend, backup_dir=str(directory), label="postgresql", timeout_seconds=30)
        assert result.startswith("pg_dump failed") if outcome == "nonzero" else result == "pg_dump error"  # nosec B101 -- privilege failure cannot become a backup success.
    assert list(directory.iterdir()) == [previous]  # nosec B101 -- remove only this known failed output, never an earlier backup.


def test_bounded_database_dump_publishes_only_its_completed_artifact(tmp_path, monkeypatch):
    from pathlib import Path

    from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig

    monkeypatch.setenv("TLDW_DB_BACKUP_PATH", str(tmp_path))
    monkeypatch.setattr(backups.shutil, "which", lambda _: "/fixture/pg_dump")
    def dump(command, **kwargs):
        target = Path(command[command.index("-f") + 1])
        assert target.name.endswith(".dump.partial") and not list(tmp_path.glob("*.dump"))  # nosec B101 -- an active or failed tool output must not be listed as a completed backup.
        target.write_bytes(b"PGDMP completed boundary")
        return SimpleNamespace(returncode=0, stderr="")
    monkeypatch.setattr(backups, "_run_postgres_tool", dump)
    backend = SimpleNamespace(backend_type=BackendType.POSTGRESQL, config=DatabaseConfig(backend_type=BackendType.POSTGRESQL))
    result = Path(backups.create_postgres_backup(backend, backup_dir=str(tmp_path), label="postgresql", timeout_seconds=30))
    assert result.suffix == ".dump" and result.read_bytes() == b"PGDMP completed boundary"  # nosec B101 -- successful callers receive only the atomically published complete file.
    assert not list(tmp_path.glob("*.partial"))  # nosec B101 -- completed temporary output is removed promptly.
