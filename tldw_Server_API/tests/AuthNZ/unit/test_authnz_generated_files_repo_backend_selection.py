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


@pytest.mark.asyncio
async def test_source_ref_lookup_scopes_sqlite_to_owner_and_feature() -> None:
    conn = _SqliteConnWithFetchrowTrap()
    repo = AuthnzGeneratedFilesRepo(db_pool=_PoolStub(conn, postgres=False))

    record = await repo.get_file_by_source_ref(
        user_id=5, source_feature="vn_assets", source_ref="vn_asset_item:4"
    )

    assert record["id"] == 11
    query, params = conn.execute_calls[-1]
    assert "user_id = ? AND source_feature = ? AND source_ref = ?" in query
    assert params == (5, "vn_assets", "vn_asset_item:4")


@pytest.mark.asyncio
async def test_source_ref_lookup_scopes_postgres_to_owner_and_feature() -> None:
    conn = _PostgresConnWithSqliteTrap()
    repo = AuthnzGeneratedFilesRepo(db_pool=_PoolStub(conn, postgres=True))

    record = await repo.get_file_by_source_ref(
        user_id=5, source_feature="vn_assets", source_ref="vn_asset_item:4"
    )

    assert record["id"] == 9
    query, params = conn.fetchrow_calls[-1]
    assert "user_id = $1 AND source_feature = $2 AND source_ref = $3" in query
    assert params == (5, "vn_assets", "vn_asset_item:4")


def test_generated_files_repo_exposes_stt_audio_constants() -> None:
    assert FILE_CATEGORY_STT_AUDIO == "stt_audio"
    assert SOURCE_FEATURE_STT == "stt"
    assert FILE_CATEGORY_STT_AUDIO in VALID_FILE_CATEGORIES
    assert SOURCE_FEATURE_STT in VALID_SOURCE_FEATURES
