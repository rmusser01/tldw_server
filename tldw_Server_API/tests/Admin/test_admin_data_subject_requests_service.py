from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_list_data_subject_requests_passes_org_scope_to_repo(monkeypatch) -> None:
    from tldw_Server_API.app.services import admin_data_subject_requests_service as service

    principal = SimpleNamespace(user_id=17, roles=["admin"])

    class _StubRepo:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        async def ensure_schema(self) -> None:
            return None

        async def list_requests(
            self,
            *,
            limit: int,
            offset: int,
            org_ids: list[int] | None = None,
            resolved_user_ids: list[int] | None = None,
        ):
            self.calls.append(
                {
                    "limit": limit,
                    "offset": offset,
                    "org_ids": list(org_ids or []),
                    "resolved_user_ids": list(resolved_user_ids or []),
                }
            )
            return [
                {
                    "id": 1,
                    "client_request_id": "dsr-1",
                    "requester_identifier": "subject@example.com",
                    "resolved_user_id": 1005,
                    "request_type": "access",
                    "status": "recorded",
                    "selected_categories": ["media_records"],
                    "preview_summary": [],
                    "coverage_metadata": {},
                    "requested_by_user_id": 17,
                    "requested_at": "2026-03-10T12:00:00+00:00",
                    "notes": None,
                }
            ], 1

    stub_repo = _StubRepo()

    monkeypatch.setattr(service.admin_scope_service, "is_platform_admin", lambda current_principal: False)

    async def _fake_get_admin_org_ids(current_principal):
        assert current_principal is principal
        return [99]

    monkeypatch.setattr(service.admin_scope_service, "get_admin_org_ids", _fake_get_admin_org_ids)

    items, total = await service.list_data_subject_requests(
        principal,
        limit=50,
        offset=0,
        requests_repo=stub_repo,
    )

    assert total == 1
    assert len(items) == 1
    assert stub_repo.calls == [
        {
            "limit": 50,
            "offset": 0,
            "org_ids": [99],
            "resolved_user_ids": [],
        }
    ]


@pytest.mark.asyncio
async def test_erase_notes_returns_transactional_finalizer_count(monkeypatch) -> None:
    from tldw_Server_API.app.services import admin_data_subject_requests_service as service

    class _Coordinator:
        async def erase(self):
            return SimpleNamespace(deleted_notes=3)

    monkeypatch.setattr(
        service,
        "_build_notes_semantic_erasure_coordinator",
        lambda _user_id: _Coordinator(),
    )
    monkeypatch.setattr(service, "_notes_content_backend_is_postgres", lambda: True)

    assert await service._erase_notes(7) == 3


@pytest.mark.asyncio
async def test_erase_notes_does_not_delete_canonical_notes_after_semantic_failure(
    monkeypatch,
) -> None:
    from tldw_Server_API.app.services import admin_data_subject_requests_service as service

    calls: list[str] = []

    class _Coordinator:
        async def erase(self):
            calls.append("semantic")
            raise RuntimeError("notes_semantic_erasure_timeout")

    monkeypatch.setattr(
        service,
        "_build_notes_semantic_erasure_coordinator",
        lambda _user_id: _Coordinator(),
    )
    monkeypatch.setattr(service, "_notes_content_backend_is_postgres", lambda: True)
    with pytest.raises(RuntimeError, match="notes_semantic_erasure_timeout"):
        await service._erase_notes(7)

    assert calls == ["semantic"]


@pytest.mark.asyncio
async def test_erase_notes_does_not_create_missing_sqlite_database(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from tldw_Server_API.app.services import admin_data_subject_requests_service as service

    missing = tmp_path / "missing" / "ChaChaNotes.db"
    monkeypatch.setattr(
        service.DatabasePaths,
        "get_chacha_db_path",
        lambda _user_id: missing,
    )
    monkeypatch.setattr(service, "_notes_content_backend_is_postgres", lambda: False)
    monkeypatch.setattr(
        service,
        "_build_notes_semantic_erasure_coordinator",
        lambda _user_id: pytest.fail("missing SQLite database must not be opened"),
    )

    assert await service._erase_notes(7) == 0
    assert not missing.exists()


def test_existing_only_notes_database_does_not_recreate_unlinked_file(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from configparser import ConfigParser

    from tldw_Server_API.app.core.DB_Management.backends.sqlite_backend import (
        SQLiteConnectionPool,
    )
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
        CharactersRAGDB,
        CharactersRAGDBError,
    )

    path = tmp_path / "ChaChaNotes.db"
    path.touch()
    original = SQLiteConnectionPool._create_connection

    def _unlink_then_connect(pool):
        path.unlink(missing_ok=True)
        return original(pool)

    monkeypatch.setattr(SQLiteConnectionPool, "_create_connection", _unlink_then_connect)

    with pytest.raises(CharactersRAGDBError):
        CharactersRAGDB(
            path,
            client_id="7",
            config=ConfigParser(),
            require_existing_sqlite=True,
        )

    assert not path.exists()


@pytest.mark.asyncio
async def test_notes_erasure_builder_releases_constructor_thread_connection(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from configparser import ConfigParser

    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
    from tldw_Server_API.app.services import admin_data_subject_requests_service as service

    path = tmp_path / "ChaChaNotes.db"
    seed = CharactersRAGDB(path, client_id="7", config=ConfigParser())
    seed.close_all_connections()
    monkeypatch.setattr(
        service.DatabasePaths,
        "get_chacha_db_path",
        lambda _user_id: path,
    )
    original_close = CharactersRAGDB.close_connection
    closed: list[CharactersRAGDB] = []

    def _close(db: CharactersRAGDB) -> None:
        closed.append(db)
        original_close(db)

    monkeypatch.setattr(CharactersRAGDB, "close_connection", _close)

    coordinator = await asyncio.to_thread(
        service._build_notes_semantic_erasure_coordinator,
        7,
    )

    assert closed == [coordinator._db]
    await coordinator.erase()


@pytest.mark.asyncio
async def test_erase_notes_maps_coordinator_construction_failure_to_stable_code(
    monkeypatch,
) -> None:
    from tldw_Server_API.app.core.Notes_Graph.semantic_erasure import SemanticErasureError
    from tldw_Server_API.app.services import admin_data_subject_requests_service as service

    def _fail(_user_id: int):
        raise RuntimeError("secret database connection detail")

    monkeypatch.setattr(service, "_notes_content_backend_is_postgres", lambda: True)
    monkeypatch.setattr(service, "_build_notes_semantic_erasure_coordinator", _fail)

    with pytest.raises(SemanticErasureError) as exc_info:
        await service._erase_notes(7)

    assert exc_info.value.code == "notes_semantic_erasure_backend_unavailable"
    assert "secret" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_notes_erasure_runs_semantic_path_without_generic_embeddings(monkeypatch) -> None:
    from tldw_Server_API.app.services import admin_data_subject_requests_service as service

    calls: list[str] = []

    async def _notes(_user_id: int) -> int:
        calls.append("notes")
        return 2

    async def _embeddings(_user_id: int) -> int:
        calls.append("embeddings")
        return 99

    class _Repo:
        def __init__(self) -> None:
            self.statuses: list[str] = []

        async def update_request_status(self, _request_id, status, notes=None):
            del notes
            self.statuses.append(status)

    monkeypatch.setitem(service._ERASURE_HANDLERS, "notes", _notes)
    monkeypatch.setitem(service._ERASURE_HANDLERS, "embeddings", _embeddings)
    repo = _Repo()

    result = await service.execute_dsr_erasure(
        request_id=41,
        user_id=7,
        selected_categories=["notes"],
        dsr_repo=repo,
    )

    assert result["status"] == "completed"
    assert calls == ["notes"]
    assert repo.statuses == ["executing", "completed"]
