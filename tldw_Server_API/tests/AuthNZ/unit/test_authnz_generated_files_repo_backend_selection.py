from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from tldw_Server_API.app.core.AuthNZ.repos.generated_files_repo import (
    FILE_CATEGORY_STT_AUDIO,
    FILE_CATEGORY_TTS_AUDIO,
    SOURCE_FEATURE_STT,
    SOURCE_FEATURE_TTS,
    VALID_FILE_CATEGORIES,
    VALID_SOURCE_FEATURES,
    AuthnzGeneratedFilesRepo,
)


class _Tx:
    def __init__(self, conn: Any) -> None:
        self._conn = conn

    async def __aenter__(self) -> Any:
        return self._conn

    async def __aexit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001, ARG002
        return False


class _PoolStub:
    def __init__(self, conn: Any, *, postgres: bool) -> None:
        self._conn = conn
        self.pool = object() if postgres else None

    def transaction(self) -> _Tx:
        return _Tx(self._conn)

    def acquire(self) -> _Tx:
        return _Tx(self._conn)


class _SqliteCursor:
    def __init__(
        self,
        *,
        lastrowid: int | None = None,
        row: Any = None,
        description: list[tuple[str]] | None = None,
        rowcount: int = 1,
    ) -> None:
        self.lastrowid = lastrowid
        self._row = row
        self.description = description or []
        self.rowcount = rowcount

    async def fetchone(self) -> Any:
        return self._row


class _SqliteConnWithFetchrowTrap:
    def __init__(self) -> None:
        self.execute_calls: list[tuple[str, Any]] = []

    async def fetchrow(self, *args, **kwargs):  # noqa: ANN001, ANN002, ARG002
        raise AssertionError("SQLite backend path should not call conn.fetchrow")

    async def execute(self, query: str, params: Any) -> _SqliteCursor:
        self.execute_calls.append((str(query), params))
        lower_q = str(query).lower()
        if "source_ref = ?" in lower_q:
            return _SqliteCursor(
                row=(11, 5, "vn_assets", "vn_asset_item:4", 0),
                description=[
                    ("id",), ("user_id",), ("source_feature",),
                    ("source_ref",), ("is_deleted",),
                ],
            )
        if "insert into generated_files" in lower_q:
            return _SqliteCursor(lastrowid=11)
        if "select * from generated_files where id = ?" in lower_q:
            row = (
                11,
                "uuid-1",
                5,
                '["alpha"]',
                1,
                0,
            )
            description = [
                ("id",),
                ("uuid",),
                ("user_id",),
                ("tags",),
                ("is_transient",),
                ("is_deleted",),
            ]
            return _SqliteCursor(row=row, description=description)
        return _SqliteCursor()


class _PostgresConnWithSqliteTrap:
    def __init__(self) -> None:
        self.fetchrow_calls: list[tuple[str, tuple[Any, ...]]] = []

    async def execute(self, *args, **kwargs):  # noqa: ANN001, ANN002, ARG002
        raise AssertionError("Postgres backend create_file should use conn.fetchrow")

    async def fetchrow(self, query: str, *params: Any) -> dict[str, Any]:
        lower_q = str(query).lower()
        if "?" in lower_q:
            raise AssertionError("Postgres backend path should not use SQLite placeholders")
        self.fetchrow_calls.append((str(query), tuple(params)))
        return {
            "id": "9",
            "uuid": "uuid-2",
            "user_id": "5",
            "tags": '["beta"]',
            "is_transient": False,
            "is_deleted": False,
        }


@pytest.mark.asyncio
async def test_create_file_sqlite_backend_selection_uses_execute_even_with_fetchrow():
    conn = _SqliteConnWithFetchrowTrap()
    repo = AuthnzGeneratedFilesRepo(db_pool=_PoolStub(conn, postgres=False))

    created = await repo.create_file(
        user_id=5,
        filename="clip.wav",
        storage_path="generated/clip.wav",
        file_category=FILE_CATEGORY_TTS_AUDIO,
        source_feature=SOURCE_FEATURE_TTS,
        tags=["alpha"],
        expires_at=datetime.now(timezone.utc),
    )

    assert created["id"] == 11
    assert created["tags"] == ["alpha"]
    assert conn.execute_calls
    assert "insert into generated_files" in conn.execute_calls[0][0].lower()
    assert "select * from generated_files where id = ?" in conn.execute_calls[1][0].lower()


@pytest.mark.asyncio
async def test_create_file_postgres_backend_selection_uses_fetchrow():
    conn = _PostgresConnWithSqliteTrap()
    repo = AuthnzGeneratedFilesRepo(db_pool=_PoolStub(conn, postgres=True))

    created = await repo.create_file(
        user_id=5,
        filename="clip.wav",
        storage_path="generated/clip.wav",
        file_category=FILE_CATEGORY_TTS_AUDIO,
        source_feature=SOURCE_FEATURE_TTS,
        tags=["beta"],
        expires_at=datetime.now(timezone.utc),
    )

    assert created["id"] == 9
    assert created["user_id"] == 5
    assert created["tags"] == ["beta"]
    assert conn.fetchrow_calls
    query, params = conn.fetchrow_calls[0]
    assert "returning *" in query.lower()
    assert "$1" in query
    assert len(params) >= 18


@pytest.mark.unit
@pytest.mark.asyncio
async def test_source_ref_lookup_scopes_sqlite_to_owner_and_feature() -> None:
    """Return the owned SQLite lookup result, never another requested scope.

    Args:
        None. The connection double supplies distinct owner/feature responses.

    Returns:
        None. Assert public row normalization, scope forwarding and misses;
        native companion coverage proves the database's filtering semantics.
    """
    conn = _ScopedSqliteLookup()
    repo = AuthnzGeneratedFilesRepo(db_pool=_PoolStub(conn, postgres=False))

    record = await repo.get_file_by_source_ref(
        user_id=5, source_feature="vn_assets", source_ref="vn_asset_item:4"
    )

    assert record == {"id": 11, "user_id": 5, "source_feature": "vn_assets",
                      "source_ref": "vn_asset_item:4", "is_deleted": False}
    assert await repo.get_file_by_source_ref(
        user_id=6, source_feature="vn_assets", source_ref="vn_asset_item:4",
    ) is None
    assert await repo.get_file_by_source_ref(
        user_id=5, source_feature="image_gen", source_ref="vn_asset_item:4",
    ) is None
    assert await repo.get_file_by_source_ref(
        user_id=5, source_feature="vn_assets", source_ref="vn_asset_item:5",
    ) is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_source_ref_lookup_scopes_postgres_to_owner_and_feature() -> None:
    """Normalize an owned PostgreSQL lookup and preserve requested scope misses.

    Args:
        None. The PostgreSQL double rejects SQLite connection operations.

    Returns:
        None. Assert public owner/feature/ref responses without SQL spelling;
        native companion coverage proves actual database isolation.
    """
    conn = _ScopedPostgresLookup()
    repo = AuthnzGeneratedFilesRepo(db_pool=_PoolStub(conn, postgres=True))

    record = await repo.get_file_by_source_ref(
        user_id=5, source_feature="vn_assets", source_ref="vn_asset_item:4"
    )

    assert record == {"id": 9, "user_id": 5, "source_feature": "vn_assets",
                      "source_ref": "vn_asset_item:4", "is_deleted": False}
    assert await repo.get_file_by_source_ref(
        user_id=6, source_feature="vn_assets", source_ref="vn_asset_item:4",
    ) is None
    assert await repo.get_file_by_source_ref(
        user_id=5, source_feature="image_gen", source_ref="vn_asset_item:4",
    ) is None
    assert await repo.get_file_by_source_ref(
        user_id=5, source_feature="vn_assets", source_ref="vn_asset_item:5",
    ) is None


def test_generated_files_repo_exposes_stt_audio_constants() -> None:
    assert FILE_CATEGORY_STT_AUDIO == "stt_audio"
    assert SOURCE_FEATURE_STT == "stt"
    assert FILE_CATEGORY_STT_AUDIO in VALID_FILE_CATEGORIES
    assert SOURCE_FEATURE_STT in VALID_SOURCE_FEATURES


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("postgres", [False, True])
async def test_vn_source_registration_reuses_live_record_in_transaction(postgres: bool) -> None:
    """Replay the live row's normalized identity rather than candidate metadata.

    Args:
        postgres: Select the PostgreSQL fetchrow or SQLite execute trap double.

    Returns:
        None. Assert the public replay result equals the current live lookup;
        native companion tests prove persistence and transaction exclusivity.
    """
    conn = _PostgresConnWithSqliteTrap() if postgres else _SqliteConnWithFetchrowTrap()
    repo = AuthnzGeneratedFilesRepo(db_pool=_PoolStub(conn, postgres=postgres))

    current = await repo.get_file_by_source_ref(
        user_id=5, source_feature="vn_assets", source_ref="vn_asset_item:4",
    )
    assert current is not None
    record = await repo.create_file(
        user_id=5, filename="loser.png", storage_path="vn_assets/loser.png",
        file_category="image", source_feature="vn_assets", source_ref="vn_asset_item:4",
    )

    assert record == {**current, "_idempotent_replay": True}
    assert record["id"] == (9 if postgres else 11)
    assert record["user_id"] == 5


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("postgres", [False, True])
@pytest.mark.parametrize("source_feature,source_ref", [
    ("vn_assets", "vn_asset_item:"),
    ("vn_assets", "vn_asset_item:abc"),
    ("vn_assets", "vn_asset_item:42:variant"),
    ("vn_assets", "vn_asset_item:0"),
    ("vn_assets", "vn_asset_item:-1"),
    ("vn_assets", "vn_asset_item:0042"),
    ("vn_assets", "vn_asset_item:42\n"),
    ("image_gen", "vn_asset_item:42"),
])
async def test_only_canonical_vn_item_refs_are_idempotent(
    postgres: bool, source_feature: str, source_ref: str,
) -> None:
    """Keep malformed or non-VN references on the ordinary registration path.

    Args:
        postgres: Select the PostgreSQL or SQLite backend connection double.
        source_feature: Feature owning the reference, including a non-VN control.
        source_ref: Empty, malformed, nonpositive, padded or suffixed item ref.

    Returns:
        None. Assert a normalized new-record result without a replay marker;
        native companion tests prove two distinct persisted registrations.
    """
    conn = _PostgresConnWithSqliteTrap() if postgres else _SqliteConnWithFetchrowTrap()
    repo = AuthnzGeneratedFilesRepo(db_pool=_PoolStub(conn, postgres=postgres))

    record = await repo.create_file(
        user_id=5, filename="new.png", storage_path="images/new.png",
        file_category="image", source_feature=source_feature, source_ref=source_ref,
    )

    assert not record.get("_idempotent_replay", False)
    assert record["id"] == (9 if postgres else 11)
    assert record["user_id"] == 5
    assert record["tags"] == (["beta"] if postgres else ["alpha"])


@pytest.mark.unit
@pytest.mark.asyncio
async def test_postgres_vn_admission_locks_all_quota_scopes_on_the_bound_connection() -> None:
    """Keep PostgreSQL admission and public reads usable on the bound repository.

    Args:
        None. The connection double rejects SQLite methods on PostgreSQL.

    Returns:
        None. Assert the normalized public row remains available after quota
        admission; native companion tests prove all three locks and release.
    """
    conn = _PostgresConnWithSqliteTrap()
    repo = AuthnzGeneratedFilesRepo(db_pool=_PoolStub(conn, postgres=True))

    async with repo.vn_item_transaction(user_id=5, source_ref="vn_asset_item:42") as bound_repo:
        await bound_repo.lock_quota_scopes(user_id=5, org_id=51, team_id=52)
        record = await bound_repo.get_file_by_id(9)

    assert record == {"id": 9, "uuid": "uuid-2", "user_id": 5, "tags": ["beta"],
                      "is_transient": False, "is_deleted": False}


class _ScopedSqliteLookup(_SqliteConnWithFetchrowTrap):
    """Supply scope-specific rows without interpreting SQL or weakening the trap."""

    async def execute(self, _query: str, params: Any) -> _SqliteCursor:
        """Return the requested scope response; native tests verify SQL filtering."""
        return _SqliteCursor(
            row=(11, 5, "vn_assets", "vn_asset_item:4", 0)
            if tuple(params) == (5, "vn_assets", "vn_asset_item:4") else None,
            description=[("id",), ("user_id",), ("source_feature",),
                         ("source_ref",), ("is_deleted",)],
        )


class _ScopedPostgresLookup:
    """Supply PostgreSQL scope responses while retaining wrong-backend traps."""

    def __init__(self) -> None:
        """Reuse the original PostgreSQL method and placeholder traps."""
        self._trap = _PostgresConnWithSqliteTrap()

    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Reject SQLite execution through the unchanged original trap."""
        return await self._trap.execute(*args, **kwargs)

    async def fetchrow(self, query: str, *params: Any) -> dict[str, Any] | None:
        """Return the requested scope's driver-shaped row, not a SQL-parser fake."""
        await self._trap.fetchrow(query, *params)
        if params != (5, "vn_assets", "vn_asset_item:4"):
            return None
        return {"id": "9", "user_id": "5", "source_feature": "vn_assets",
                "source_ref": "vn_asset_item:4", "is_deleted": False}
