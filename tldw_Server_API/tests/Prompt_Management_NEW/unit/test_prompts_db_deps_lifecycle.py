"""
Lifecycle regression tests for Prompts DB dependency wiring.
"""

import os
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from tldw_Server_API.app.api.v1.API_Deps import Prompts_DB_Deps
from tldw_Server_API.app.core.DB_Management.Prompts_DB import DatabaseError, PromptsDatabase


@pytest.mark.unit
def test_prompts_db_deps_import_has_no_unawaited_coroutine_warning() -> None:
    script = "import gc\nimport tldw_Server_API.app.api.v1.API_Deps.Prompts_DB_Deps\ngc.collect()\n"
    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "always"
    proc = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    combined = f"{proc.stdout}\n{proc.stderr}"
    assert proc.returncode == 0, combined
    assert "was never awaited" not in combined
    assert "_process_pending_closes" not in combined


@pytest.mark.asyncio
@pytest.mark.unit
async def test_prompts_pending_close_worker_start_stop_cycle() -> None:
    started = Prompts_DB_Deps.start_prompts_pending_close_worker()
    assert started is True
    assert Prompts_DB_Deps._pending_close_task is not None
    await Prompts_DB_Deps.stop_prompts_pending_close_worker()
    assert Prompts_DB_Deps._pending_close_task is None


@pytest.mark.unit
@pytest.mark.parametrize("fail_probe", [False, True])
def test_liveness_probe_closes_its_worker_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_probe: bool
) -> None:
    """A successful or failed health probe must release its own SQLite handle."""
    database = PromptsDatabase(tmp_path / "probe.sqlite", "probe-test")
    database.close_connection()
    connections: list[sqlite3.Connection] = []
    original_get_connection = database.get_connection

    def record_connection() -> sqlite3.Connection:
        """Observe the real connection, optionally failing after it opens."""
        connection = original_get_connection()
        connections.append(connection)
        if fail_probe:
            raise DatabaseError("Probe failed after opening the connection")
        return connection

    monkeypatch.setattr(database, "get_connection", record_connection)
    with ThreadPoolExecutor(max_workers=1) as worker:
        try:
            assert worker.submit(Prompts_DB_Deps._is_db_instance_alive, database).result() is not fail_probe
            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                connections[0].execute("SELECT 1")
        finally:
            worker.submit(database.close_connection).result()


@pytest.mark.unit
def test_cached_instance_creation_releases_initialization_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Creating a cached database must not pin a handle to its setup worker."""
    connections: list[sqlite3.Connection] = []

    def create_database(db_path: str, client_id: str) -> PromptsDatabase:
        """Record the real constructor's initial SQLite handle."""
        database = PromptsDatabase(db_path, client_id)
        connections.append(database.get_connection())
        return database

    def database_path(user_id: int, salt: str | None) -> Path:
        """Keep database setup isolated to the test directory."""
        return tmp_path / f"{user_id}-{salt}.sqlite"

    monkeypatch.setattr(Prompts_DB_Deps, "PromptsDatabase", create_database)
    monkeypatch.setattr(Prompts_DB_Deps, "_get_prompts_db_path_for_user", database_path)
    with ThreadPoolExecutor(max_workers=1) as worker:
        database, _ = worker.submit(Prompts_DB_Deps._create_prompts_db_instance, 1, None, "setup-test").result()
        try:
            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                connections[0].execute("SELECT 1")
            # Releasing setup state must not make the cached instance unusable.
            assert worker.submit(Prompts_DB_Deps._is_db_instance_alive, database).result()
        finally:
            worker.submit(database.close_connection).result()
