from __future__ import annotations

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
async def test_erase_notes_removes_graph_edges_before_notes(monkeypatch) -> None:
    from tldw_Server_API.app.services import admin_data_subject_requests_service as service

    events: list[str] = []

    class _Coordinator:
        async def erase(self):
            events.append("semantic")
            return SimpleNamespace(deleted_notes=3)

    def _hard_delete(_path, statements) -> int:
        for sql, _params in statements:
            if sql.startswith("DELETE FROM note_edges"):
                events.append("note_edges")
            elif sql.startswith("DELETE FROM note_wikilink_edges"):
                events.append("note_wikilink_edges")
            elif sql == "DELETE FROM notes":
                events.append("notes")
        return 6

    monkeypatch.setattr(
        service,
        "_build_notes_semantic_erasure_coordinator",
        lambda _user_id: _Coordinator(),
    )
    monkeypatch.setattr(service, "_sqlite_hard_delete_sync", _hard_delete)

    assert await service._erase_notes(7) == 6
    assert events == ["semantic", "note_edges", "note_wikilink_edges", "notes"]


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
    monkeypatch.setattr(
        service,
        "_sqlite_hard_delete_sync",
        lambda _path, _statements: calls.append("canonical"),
    )

    with pytest.raises(RuntimeError, match="notes_semantic_erasure_timeout"):
        await service._erase_notes(7)

    assert calls == ["semantic"]


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
