"""Contracts for the process-level schema setup memo.

``ensure_once`` decides whether a database's schema routine gets to be skipped.
Getting that wrong is not a slow request, it is a request against a database
with no tables, so the conditions under which it says "already done" are worth
pinning down.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management import schema_once
from tldw_Server_API.app.core.DB_Management.schema_once import ensure_once


@pytest.fixture(autouse=True)
def _isolate_memo():
    """Each test starts from an empty memo and leaves one behind."""
    schema_once.reset()
    yield
    schema_once.reset()


def _make_db(path: Path) -> None:
    """Create a database with the table our verify callback looks for."""
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS marker (id INTEGER PRIMARY KEY)")
        conn.commit()
    finally:
        conn.close()


def _has_marker(path: Path) -> bool:
    if not path.exists():
        return False
    conn = sqlite3.connect(path)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='marker'"
        ).fetchall()
    finally:
        conn.close()
    return bool(rows)


@pytest.mark.unit
def test_setup_runs_once_for_the_same_database(tmp_path: Path) -> None:
    db = tmp_path / "app.db"
    _make_db(db)
    calls = []

    for _ in range(5):
        ensure_once("s", db, lambda: calls.append(1), verify=lambda: _has_marker(db))

    assert len(calls) == 1


@pytest.mark.unit
def test_scopes_do_not_satisfy_each_other(tmp_path: Path) -> None:
    """Two callers sharing a file own different tables."""
    db = tmp_path / "app.db"
    _make_db(db)
    calls = []

    ensure_once("a", db, lambda: calls.append("a"), verify=lambda: True)
    ensure_once("b", db, lambda: calls.append("b"), verify=lambda: True)
    ensure_once("a", db, lambda: calls.append("a"), verify=lambda: True)

    assert calls == ["a", "b"]


@pytest.mark.unit
def test_failed_setup_is_retried(tmp_path: Path) -> None:
    """A crash halfway through must not be remembered as success."""
    db = tmp_path / "app.db"
    _make_db(db)
    attempts = []

    def ensure() -> None:
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        ensure_once("s", db, ensure, verify=lambda: True)
    ensure_once("s", db, ensure, verify=lambda: True)

    assert len(attempts) == 2


@pytest.mark.unit
def test_database_recreated_at_the_same_path_is_set_up_again(tmp_path: Path) -> None:
    """The reason verify exists: a filesystem may reuse the inode."""
    db = tmp_path / "app.db"
    _make_db(db)
    calls = []

    def ensure() -> None:
        calls.append(1)
        _make_db(db)

    ensure_once("s", db, ensure, verify=lambda: _has_marker(db))
    db.unlink()
    _make_db_without_marker = sqlite3.connect(db)
    _make_db_without_marker.close()
    ensure_once("s", db, ensure, verify=lambda: _has_marker(db))

    assert len(calls) == 2, "a database missing its tables was served from the memo"


@pytest.mark.unit
def test_a_verify_that_raises_is_treated_as_absent(tmp_path: Path) -> None:
    """We cannot trust a memo whose check blew up."""
    db = tmp_path / "app.db"
    _make_db(db)
    calls = []

    def exploding_verify() -> bool:
        raise sqlite3.DatabaseError("file is not a database")

    ensure_once("s", db, lambda: calls.append(1), verify=exploding_verify)
    ensure_once("s", db, lambda: calls.append(1), verify=exploding_verify)

    assert len(calls) == 2


@pytest.mark.unit
@pytest.mark.parametrize("path", [None, "", ":memory:", "file::memory:?cache=shared"])
def test_databases_without_a_real_file_are_never_memoized(path) -> None:
    """In-memory databases are distinct per connection despite sharing a name."""
    calls = []

    for _ in range(3):
        ensure_once("s", path, lambda: calls.append(1), verify=lambda: True)

    assert len(calls) == 3


@pytest.mark.unit
def test_sqlite_uris_are_memoized(tmp_path: Path) -> None:
    """A file-backed URI names a real file, so it can be identified."""
    db = tmp_path / "app.db"
    _make_db(db)
    calls = []
    uri = f"file:{db}?mode=rwc"

    for _ in range(3):
        ensure_once("s", uri, lambda: calls.append(1), verify=lambda: _has_marker(db))

    assert len(calls) == 1, "a file-backed URI fell through to setup every time"


@pytest.mark.unit
def test_concurrent_first_touches_run_setup_once(tmp_path: Path) -> None:
    """Two requests arriving together must not both replay the DDL."""
    db = tmp_path / "app.db"
    _make_db(db)
    calls = []
    calls_lock = threading.Lock()
    start = threading.Barrier(8)

    def ensure() -> None:
        with calls_lock:
            calls.append(1)

    def worker() -> None:
        start.wait()
        ensure_once("s", db, ensure, verify=lambda: _has_marker(db))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(calls) == 1


@pytest.mark.unit
def test_tracking_is_bounded(tmp_path: Path, monkeypatch) -> None:
    """A process opening a database per user must not grow this forever."""
    monkeypatch.setattr(schema_once, "_MAX_TRACKED_DATABASES", 4)

    for index in range(20):
        db = tmp_path / f"user{index}.db"
        _make_db(db)
        ensure_once("s", db, lambda: None, verify=lambda: True)

    assert len(schema_once._ENTRIES) <= 4


@pytest.mark.unit
def test_reset_targets_one_scope(tmp_path: Path) -> None:
    db = tmp_path / "app.db"
    _make_db(db)
    calls = []

    ensure_once("a", db, lambda: calls.append("a"), verify=lambda: True)
    ensure_once("b", db, lambda: calls.append("b"), verify=lambda: True)
    schema_once.reset("a")
    ensure_once("a", db, lambda: calls.append("a"), verify=lambda: True)
    ensure_once("b", db, lambda: calls.append("b"), verify=lambda: True)

    assert calls == ["a", "b", "a"]
