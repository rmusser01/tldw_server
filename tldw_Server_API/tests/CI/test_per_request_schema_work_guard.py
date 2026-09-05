"""Guard against schema verification running on every request.

A request-scoped ``MediaDatabase`` is constructed per request, and its schema
bootstrap used to replay the full ``CREATE ... IF NOT EXISTS`` set every time --
even on the path where the version check had already concluded the schema was
current. A warm ``GET /api/v1/media/`` issued 217 SQL statements, 177 of them
DDL, to do the work of two SELECTs. Nothing detected it, because it cost only a
couple of milliseconds and was invisible unless someone counted.

Why this asserts on a call count rather than on observed SQL: the DDL is issued
through ``executescript`` and helper routines that do not surface through
``sqlite3.set_trace_callback`` in this configuration. A SQL-sniffing version of
this test passed happily with the memo disabled -- it would have given false
confidence. Counting entries into the expensive routine is deterministic and
cannot silently under-report.

The invariant: after the first request has verified the schema,
``ensure_sqlite_post_core_structures`` must not run again for that database.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

WARMUP_REQUESTS = 3
MEDIA_LISTING = "/api/v1/media/"


@pytest.mark.integration
@pytest.mark.parametrize("prewarmed", [False, True], ids=["cold", "prewarmed"])
def test_warm_media_listing_does_not_reverify_schema(
    client_user_only: TestClient, monkeypatch: pytest.MonkeyPatch, prewarmed: bool
) -> None:
    """A warm request must not re-run the post-core schema structures."""
    from tldw_Server_API.app.core.DB_Management.media_db.schema.backends import (
        sqlite_helpers,
    )

    if prewarmed:
        # Reproduce a prior test/request populating the process-wide schema memo.
        assert client_user_only.get(MEDIA_LISTING).status_code == 200

    calls: list[str] = []
    original = sqlite_helpers.ensure_sqlite_post_core_structures

    def _spy(db, conn):
        calls.append(str(getattr(db, "db_path_str", db)))
        return original(db, conn)

    monkeypatch.setattr(
        sqlite_helpers, "ensure_sqlite_post_core_structures", _spy
    )
    # Startup or earlier requests may have verified this database before the
    # spy was installed. Start measurement cold; retain caching thereafter.
    sqlite_helpers.reset_schema_verification_cache()

    for _ in range(WARMUP_REQUESTS):
        assert client_user_only.get(MEDIA_LISTING).status_code == 200

    # The warm-up above must have exercised the schema path at least once,
    # otherwise this test would pass without proving anything.
    assert calls, (
        "schema verification never ran during warm-up; this guard is not "
        "exercising the code path it exists to protect"
    )

    calls.clear()
    assert client_user_only.get(MEDIA_LISTING).status_code == 200

    assert not calls, (
        f"schema verification ran {len(calls)} time(s) on a warm request. "
        "Verification belongs at startup or behind a per-database memo, not on "
        "the request path: it replays the full CREATE ... IF NOT EXISTS set on "
        f"every call. Databases re-verified: {sorted(set(calls))}"
    )


@pytest.mark.integration
def test_schema_verification_memo_is_keyed_per_database(tmp_path) -> None:
    """The memo must not let a different database skip verification.

    Keyed on path plus device/inode, so a database recreated at the same path
    (test teardown, a restored backup) is verified again rather than trusted.
    """
    from tldw_Server_API.app.core.DB_Management.media_db.schema.backends.sqlite_helpers import (
        _schema_verification_key,
    )

    first = tmp_path / "one.db"
    second = tmp_path / "two.db"
    first.write_bytes(b"")
    second.write_bytes(b"")

    class _Db:
        is_memory_db = False

        def __init__(self, path):
            self.db_path_str = str(path)
            self.db_path = str(path)

    key_first = _schema_verification_key(_Db(first), 1)
    key_second = _schema_verification_key(_Db(second), 1)
    assert key_first is not None and key_second is not None
    assert key_first != key_second, "two databases must not share a memo entry"

    # A different target schema version must also miss the memo.
    assert _schema_verification_key(_Db(first), 2) != key_first

    # Recreating the file at the same path changes the inode, so the memo misses.
    first.unlink()
    first.write_bytes(b"")
    assert _schema_verification_key(_Db(first), 1) != key_first, (
        "a recreated database must be verified again, not served from the memo"
    )


@pytest.mark.integration
def test_in_memory_databases_are_never_memoized() -> None:
    """Each ``:memory:`` connection is a distinct database; caching would lie."""
    from tldw_Server_API.app.core.DB_Management.media_db.schema.backends.sqlite_helpers import (
        _schema_verification_key,
    )

    class _MemoryDb:
        is_memory_db = True
        db_path_str = ":memory:"
        db_path = ":memory:"

    assert _schema_verification_key(_MemoryDb(), 1) is None
