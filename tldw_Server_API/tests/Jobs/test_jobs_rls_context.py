from __future__ import annotations

from contextlib import contextmanager

import pytest

from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.playlist_ingest_store import (
    PlaylistIngestStore,
)
from tldw_Server_API.app.core.Jobs.manager import JobManager


def _current_rls_context() -> tuple[bool, str | None, str | None]:
    return (
        JobManager._RLS_IS_ADMIN.get(),
        JobManager._RLS_DOMAIN_ALLOWLIST.get(),
        JobManager._RLS_OWNER_USER_ID.get(),
    )


def test_scoped_rls_context_restores_nested_and_exception_state():
    JobManager.clear_rls_context()
    JobManager.set_rls_context(
        is_admin=True,
        domain_allowlist="outer",
        owner_user_id="outer-owner",
    )
    try:
        with JobManager.rls_context(
            is_admin=False,
            domain_allowlist="media_ingest",
            owner_user_id="u1",
        ):
            assert _current_rls_context() == (False, "media_ingest", "u1")
            with JobManager.rls_context(
                is_admin=False,
                domain_allowlist="nested",
                owner_user_id="u2",
            ):
                assert _current_rls_context() == (False, "nested", "u2")
            assert _current_rls_context() == (False, "media_ingest", "u1")

            with pytest.raises(RuntimeError, match="boom"):
                with JobManager.rls_context(
                    is_admin=True,
                    domain_allowlist=None,
                    owner_user_id=None,
                ):
                    raise RuntimeError("boom")
            assert _current_rls_context() == (False, "media_ingest", "u1")

        assert _current_rls_context() == (True, "outer", "outer-owner")
    finally:
        JobManager.clear_rls_context()


def test_playlist_store_sets_owner_before_postgres_cursor_and_restores(monkeypatch):
    manager = object.__new__(JobManager)
    manager.backend = "postgres"
    observed: list[tuple[str, tuple[bool, str | None, str | None]]] = []

    class FakeConnection:
        def commit(self):
            return None

        def rollback(self):
            return None

        def close(self):
            return None

    def fake_connect():
        observed.append(("connect", _current_rls_context()))
        return FakeConnection()

    @contextmanager
    def fake_pg_cursor(_connection):
        observed.append(("cursor", _current_rls_context()))
        yield object()

    monkeypatch.setattr(manager, "_connect", fake_connect)
    monkeypatch.setattr(manager, "_pg_cursor", fake_pg_cursor)
    store = PlaylistIngestStore(manager)

    JobManager.clear_rls_context()
    JobManager.set_rls_context(
        is_admin=True,
        domain_allowlist="outer",
        owner_user_id="outer-owner",
    )
    try:
        with store._connection(owner_user_id="u1", write=False):
            assert _current_rls_context() == (False, "media_ingest", "u1")
        assert _current_rls_context() == (True, "outer", "outer-owner")

        with pytest.raises(RuntimeError, match="store failure"):
            with store._connection(owner_user_id="u2", write=True):
                assert _current_rls_context() == (False, "media_ingest", "u2")
                raise RuntimeError("store failure")
        assert _current_rls_context() == (True, "outer", "outer-owner")

        with store._connection(owner_user_id="u3", write=False, rls_admin=True):
            assert _current_rls_context() == (True, "media_ingest", "u3")
        assert _current_rls_context() == (True, "outer", "outer-owner")
    finally:
        JobManager.clear_rls_context()

    assert observed == [
        ("connect", (False, "media_ingest", "u1")),
        ("cursor", (False, "media_ingest", "u1")),
        ("connect", (False, "media_ingest", "u2")),
        ("cursor", (False, "media_ingest", "u2")),
        ("connect", (True, "media_ingest", "u3")),
        ("cursor", (True, "media_ingest", "u3")),
    ]
