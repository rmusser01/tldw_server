"""Every schema_once caller's verify callback must actually work.

``ensure_once`` treats a verify callback that raises as "schema absent" and
replays the setup. That is the safe direction, but it means a broken callback is
invisible: nothing fails, the memo simply never holds and the optimisation
quietly does nothing while the log fills with warnings.

Two of the four callbacks shipped broken for exactly this reason. They read
``row[0]``, but ``DatabaseBackend.execute()`` returns a ``QueryResult`` whose
rows are dicts, so every call raised ``KeyError: 0``:

    schema_once: verification failed for .../Media_DB_v2.db (0); re-running
    schema setup

These tests call each callback against a real database and require True, which
is the one thing the failing-open design cannot tell us on its own.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.DB_Management.RPG_DB import RPGRepository
from tldw_Server_API.app.core.DB_Management.Scheduled_Tasks_DB import (
    ScheduledTasksDatabase,
)
from tldw_Server_API.app.core.DB_Management.Watchlists_DB import WatchlistsDatabase


@pytest.fixture()
def _user_db_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point per-user database resolution at a scratch directory."""
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))
    return tmp_path


@pytest.mark.unit
def test_collections_verify_reports_its_own_schema_as_present(_user_db_dir: Path) -> None:
    """Construction sets the schema up, so the check must then agree."""
    db = CollectionsDatabase.for_user(user_id=1)

    assert db._collections_schema_present() is True


@pytest.mark.unit
def test_watchlists_verify_reports_its_own_schema_as_present(_user_db_dir: Path) -> None:
    db = WatchlistsDatabase.for_user(user_id=1)

    assert db._watchlists_schema_present() is True


@pytest.mark.unit
def test_scheduled_tasks_verify_reports_its_own_schema_as_present(
    _user_db_dir: Path,
) -> None:
    db = ScheduledTasksDatabase.for_user(user_id=1)
    db.ensure_schema()

    assert db.schema_present() is True


@pytest.mark.unit
def test_rpg_verify_reports_its_own_schema_as_present() -> None:
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

    repo = RPGRepository.initialized(CharactersRAGDB(":memory:", "schema-once-test"))

    assert repo._schema_present() is True


@pytest.mark.unit
def test_a_broken_verify_would_be_caught_here(_user_db_dir: Path) -> None:
    """Pin the failure mode these tests exist for.

    ``row[0]`` against a dict row is what shipped, and ensure_once swallowed it.
    """
    db = CollectionsDatabase.for_user(user_id=1)
    rows = db._backend.execute(
        "SELECT name FROM sqlite_master WHERE type='table' LIMIT 1"
    )
    row = next(iter(rows))

    assert isinstance(row, dict), (
        "backend rows are no longer dicts; the verify callbacks index them by "
        "column name and need revisiting"
    )
    with pytest.raises(KeyError):
        row[0]
