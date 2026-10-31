"""Every schema_once caller's verification must actually work.

``ensure_once`` treats a verify callback that raises as "schema absent" and
replays the setup. That is the safe direction, but it means a broken callback is
invisible: nothing fails, the memo simply never holds and the de-duplication
quietly does nothing while the log fills with warnings.

Two of the four callbacks shipped broken for exactly this reason. They read
``row[0]``, but ``DatabaseBackend.execute()`` returns a ``QueryResult`` whose
rows are dicts, so every call raised ``KeyError: 0``::

    schema_once: verification failed for .../Media_DB_v2.db (0); re-running
    schema setup

These tests drive each repository the way production does and assert that
warning never appears -- the observable symptom, rather than the callbacks'
return values, so a refactor that keeps the behaviour keeps the tests.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from loguru import logger

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.DB_Management.RPG_DB import RPGRepository
from tldw_Server_API.app.core.DB_Management.Scheduled_Tasks_DB import (
    ScheduledTasksDatabase,
)
from tldw_Server_API.app.core.DB_Management.Watchlists_DB import WatchlistsDatabase

VERIFICATION_FAILED = "schema_once: verification failed"


@contextlib.contextmanager
def _warnings() -> Iterator[list[str]]:
    """Collect WARNING-level loguru messages emitted inside the block.

    Yields:
        The list of messages, filled as they are emitted.
    """
    messages: list[str] = []
    sink_id = logger.add(messages.append, format="{message}", level="WARNING")
    try:
        yield messages
    finally:
        logger.remove(sink_id)


@pytest.fixture()
def _sqlite_user_databases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Resolve per-user databases to SQLite files under a scratch directory.

    The callbacks under test query ``sqlite_master``, so these repositories have
    to be on SQLite; an ambient PostgreSQL content backend would send that query
    to an engine that has no such table.

    Returns:
        The scratch directory holding the per-user databases.
    """
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))
    monkeypatch.delenv("TLDW_CONTENT_DB_BACKEND", raising=False)
    return tmp_path


# Three, not two. The first call for a database that does not exist yet cannot
# stat it, so ``ensure_once`` declines to memoize and runs the setup directly --
# the file only appears as a side effect of that first call. The second call
# memoizes. Verification therefore first runs on the third, and a driver that
# stops at two proves nothing.
_ROUNDS = 3


def _collections(tmp_path: Path) -> None:
    """Construct the repository the way a request does, repeatedly."""
    for _ in range(_ROUNDS):
        CollectionsDatabase.for_user(user_id=1)


def _watchlists(tmp_path: Path) -> None:
    """Ensure the schema through the entry point the FastAPI dependency uses."""
    db = WatchlistsDatabase.for_user(user_id=1)
    for _ in range(_ROUNDS):
        db.ensure_schema_once()


def _scheduled_tasks(tmp_path: Path) -> None:
    """Set the schema up, then check it the way the control-plane service does."""
    db = ScheduledTasksDatabase.for_user(user_id=1)
    db.ensure_schema()
    assert db.schema_present() is True


def _rpg(tmp_path: Path) -> None:
    """Build the repository repeatedly against one file-backed database.

    A file is required: ``ensure_once`` never memoizes an in-memory database, so
    an in-memory one would skip verification entirely and prove nothing.
    """
    db = CharactersRAGDB(str(tmp_path / "rpg.db"), "schema-once-test")
    for _ in range(_ROUNDS):
        RPGRepository.initialized(db)


@pytest.mark.unit
@pytest.mark.parametrize(
    "scope",
    ["collections", "watchlists", "scheduled_tasks", "rpg"],
)
def test_repeated_setup_never_reports_a_failed_verification(
    scope: str, tmp_path: Path, _sqlite_user_databases: Path
) -> None:
    """A warning here means the memo is not holding and the DDL is replaying.

    Returns:
        None.
    """
    drivers: dict[str, Callable[[Path], None]] = {
        "collections": _collections,
        "watchlists": _watchlists,
        "scheduled_tasks": _scheduled_tasks,
        "rpg": _rpg,
    }

    with _warnings() as messages:
        drivers[scope](tmp_path)

    failed = [message for message in messages if VERIFICATION_FAILED in message]
    assert not failed, (
        f"{scope} could not verify its own schema, so ensure_once falls back to "
        f"replaying the full setup on every call:\n  " + "\n  ".join(failed)
    )
