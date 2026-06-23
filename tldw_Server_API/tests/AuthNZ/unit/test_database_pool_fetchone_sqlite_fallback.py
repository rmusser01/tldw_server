from __future__ import annotations

from contextlib import asynccontextmanager
import types

import pytest

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.settings import Settings


def test_resolve_sqlite_paths_normalizes_file_uri_windows_drive_path():
    db_path, is_uri, fs_path = DatabasePool._resolve_sqlite_paths(
        "file:///C:/Users/runneradmin/AppData/Local/Temp/users.db"
    )

    assert db_path == "file:///C:/Users/runneradmin/AppData/Local/Temp/users.db"
    assert is_uri is True
    assert fs_path == "C:/Users/runneradmin/AppData/Local/Temp/users.db"


class _FakeCursor:
    def __init__(self, row):
        self._row = row

    async def fetchone(self):
        return self._row


class _FakeConn:
    def __init__(self, row):
        self._row = row

    async def execute(self, _query, _params):
        return _FakeCursor(self._row)


def _stub_acquire_with_row(row):
    @asynccontextmanager
    async def _ctx():
        yield _FakeConn(row)

    return _ctx()


class _DictConvertibleRow:
    def keys(self):  # pragma: no cover - exercised by fetchone implementation
        return ["id", "name"]

    def __getitem__(self, _key):
        raise IndexError("No item with that key")

    def __iter__(self):
        yield ("id", 10)
        yield ("name", "alice")


class _TupleOnlyRow:
    def keys(self):
        raise RuntimeError("keys unavailable")

    def __iter__(self):
        yield 9
        yield "legacy-user"


@pytest.mark.asyncio
async def test_fetchone_sqlite_handles_dict_row(monkeypatch):
    pool = DatabasePool(Settings(AUTH_MODE="single_user", DATABASE_URL="sqlite:///:memory:"))
    monkeypatch.setattr(pool, "acquire", types.MethodType(lambda _self: _stub_acquire_with_row({"id": 1}), pool))

    row = await pool.fetchone("SELECT id FROM users WHERE id = ?", 1)
    assert row == {"id": 1}


@pytest.mark.asyncio
async def test_fetchone_sqlite_falls_back_to_dict_row_conversion(monkeypatch):
    pool = DatabasePool(Settings(AUTH_MODE="single_user", DATABASE_URL="sqlite:///:memory:"))
    monkeypatch.setattr(
        pool,
        "acquire",
        types.MethodType(lambda _self: _stub_acquire_with_row(_DictConvertibleRow()), pool),
    )

    row = await pool.fetchone("SELECT id, username FROM users WHERE id = ?", 10)
    assert row == {"id": 10, "name": "alice"}


@pytest.mark.asyncio
async def test_fetchone_sqlite_falls_back_to_positional_mapping(monkeypatch):
    pool = DatabasePool(Settings(AUTH_MODE="single_user", DATABASE_URL="sqlite:///:memory:"))
    monkeypatch.setattr(
        pool,
        "acquire",
        types.MethodType(lambda _self: _stub_acquire_with_row(_TupleOnlyRow()), pool),
    )

    row = await pool.fetchone("SELECT id, username FROM users WHERE id = ?", 9)
    assert row == {"0": 9, "1": "legacy-user"}
