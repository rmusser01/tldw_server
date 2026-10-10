from __future__ import annotations

from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import try_get_media_db_for_user
from tldw_Server_API.app.api.v1.endpoints import workspaces as workspaces_endpoint
from tldw_Server_API.app.api.v1.endpoints.workspaces_rate_limit_policy import WORKSPACES_READ_RATE_LIMIT
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


class _MediaPreviewDB:
    def __init__(
        self,
        rows: dict[int, dict[str, Any]],
        chunks: dict[int, list[dict[str, Any]]] | None = None,
    ) -> None:
        self.rows = rows
        self.chunks = chunks or {}

    def get_media_by_id(
        self,
        media_id: int,
        *,
        include_deleted: bool = False,
        include_trash: bool = False,
    ) -> dict[str, Any] | None:
        _ = (include_deleted, include_trash)
        row = self.rows.get(media_id)
        return dict(row) if row else None

    def get_media_status_by_id(
        self,
        media_id: int,
        *,
        include_deleted: bool = False,
        include_trash: bool = False,
    ) -> dict[str, Any] | None:
        row = self.get_media_by_id(
            media_id,
            include_deleted=include_deleted,
            include_trash=include_trash,
        )
        if row is None:
            return None
        content = str(row.pop("content", "") or "")
        row.setdefault("has_content", bool(content.strip()))
        return row

    def has_unvectorized_chunks(self, media_id: int) -> bool:
        return bool(self.chunks.get(media_id))

    def get_unvectorized_chunks_in_range(
        self,
        media_id: int,
        start_index: int,
        end_index: int,
    ) -> list[dict[str, Any]]:
        return [
            dict(chunk)
            for chunk in self.chunks.get(media_id, [])
            if start_index <= int(chunk.get("chunk_index", -1)) <= end_index
        ]


class _JobManagerDouble:
    def __init__(self, jobs: list[dict[str, Any]] | None = None) -> None:
        self.jobs = jobs or []

    def list_jobs(self, **kwargs: Any) -> list[dict[str, Any]]:
        _ = kwargs
        return list(self.jobs)


@pytest.fixture
def workspace_preview_db(tmp_path):
    db = CharactersRAGDB(db_path=str(tmp_path / "chacha.db"), client_id="user-1")
    db.upsert_workspace("ws-preview", "Preview Workspace")
    db.add_workspace_source(
        "ws-preview",
        {
            "id": "src-ready",
            "media_id": 1,
            "title": "Ready source",
            "source_type": "pdf",
            "url": "https://example.test/ready.pdf",
            "position": 0,
            "selected": True,
        },
    )
    db.add_workspace_source(
        "ws-preview",
        {
            "id": "src-pending",
            "media_id": 2,
            "title": "Pending source",
            "source_type": "web",
            "position": 1,
            "selected": True,
        },
    )
    db.add_workspace_source(
        "ws-preview",
        {
            "id": "src-missing",
            "media_id": 99,
            "title": "Missing source",
            "source_type": "docx",
            "position": 2,
            "selected": False,
        },
    )
    return db


@pytest.fixture
def workspace_preview_app():
    app = FastAPI()
    app.include_router(workspaces_endpoint.router, prefix="/api/v1/workspaces")
    return app


async def _allow_rate_limit() -> None:
    return None


async def _user() -> SimpleNamespace:
    return SimpleNamespace(id=1, username="testuser", email="test@example.com", roles=["admin"], is_admin=True)


async def _provider_ready_service_capabilities(
    *,
    workspace_id: str,
    user_id: int | str | None,
) -> dict[str, Any]:
    _ = (workspace_id, user_id)
    return {
        "workspace_services": {
            "provider": {
                "state": "available",
                "reason_code": None,
                "management_surface": "model_settings",
            }
        }
    }


def _install_overrides(
    app: Any,
    db: CharactersRAGDB,
    media_db: _MediaPreviewDB | None,
    jobs: list[dict[str, Any]] | None = None,
) -> None:
    async def _media_db() -> AsyncGenerator[_MediaPreviewDB | None, None]:
        yield media_db

    app.dependency_overrides[get_request_user] = _user
    app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    app.dependency_overrides[try_get_media_db_for_user] = _media_db
    app.dependency_overrides[workspaces_endpoint.try_get_workspace_job_manager] = lambda: _JobManagerDouble(jobs)
    app.dependency_overrides[WORKSPACES_READ_RATE_LIMIT] = _allow_rate_limit


def _clear_overrides(app: Any) -> None:
    app.dependency_overrides.pop(get_request_user, None)
    app.dependency_overrides.pop(get_chacha_db_for_user, None)
    app.dependency_overrides.pop(try_get_media_db_for_user, None)
    app.dependency_overrides.pop(workspaces_endpoint.try_get_workspace_job_manager, None)
    app.dependency_overrides.pop(WORKSPACES_READ_RATE_LIMIT, None)


def _media_db() -> _MediaPreviewDB:
    large_text = (
        "Research workspace captured content starts here. "
        "This text is intentionally long enough to prove preview truncation. "
        "The endpoint must not return unbounded document bodies in page shell responses."
    )
    return _MediaPreviewDB(
        {
            1: {
                "id": 1,
                "title": "Ready source",
                "type": "pdf",
                "content": large_text,
                "chunking_status": "completed",
                "vector_processing": 1,
            },
            2: {
                "id": 2,
                "title": "Pending source",
                "type": "web",
                "content": "",
                "chunking_status": "",
                "vector_processing": 0,
            },
        },
        chunks={
            1: [
                {
                    "chunk_index": 0,
                    "uuid": "chunk-ready-0",
                    "chunk_text": "First chunk with citeable evidence.",
                    "start_char": 0,
                    "end_char": 35,
                    "chunk_type": "text",
                },
                {
                    "chunk_index": 1,
                    "uuid": "chunk-ready-1",
                    "chunk_text": "Second chunk with another evidence point.",
                    "start_char": 36,
                    "end_char": 78,
                    "chunk_type": "text",
                },
            ]
        },
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    "suffix",
    ["", "/sources", "/artifacts", "/notes", "/sources/src-ready/preview"],
)
@pytest.mark.parametrize("expected_user_id", ["2", " "])
def test_workspace_content_reads_reject_stale_expected_user(
    workspace_preview_app,
    workspace_preview_db,
    suffix: str,
    expected_user_id: str,
):
    _install_overrides(workspace_preview_app, workspace_preview_db, _media_db())
    try:
        with (
            patch.object(
                workspace_preview_db,
                "get_workspace",
                wraps=workspace_preview_db.get_workspace,
            ) as workspace_read,
            TestClient(workspace_preview_app) as client,
        ):
            response = client.get(
                f"/api/v1/workspaces/ws-preview{suffix}",
                headers={"X-TLDW-Expected-User-ID": expected_user_id},
            )
        assert response.status_code == 412
        assert response.headers["cache-control"] == "no-store"
        assert response.json()["detail"]["code"] == "request_config_scope_changed"
        workspace_read.assert_not_called()
    finally:
        _clear_overrides(workspace_preview_app)


@pytest.mark.integration
@pytest.mark.parametrize(
    "suffix",
    ["", "/sources", "/artifacts", "/notes", "/sources/src-ready/preview"],
)
@pytest.mark.parametrize("expected_user_id", [None, "1"])
def test_workspace_content_reads_accept_current_or_unasserted_user(
    workspace_preview_app,
    workspace_preview_db,
    suffix: str,
    expected_user_id: str | None,
):
    _install_overrides(workspace_preview_app, workspace_preview_db, _media_db())
    try:
        with TestClient(workspace_preview_app) as client:
            response = client.get(
                f"/api/v1/workspaces/ws-preview{suffix}",
                headers={"X-TLDW-Expected-User-ID": expected_user_id}
                if expected_user_id is not None else {},
            )
        assert response.status_code == 200, response.text
    finally:
        _clear_overrides(workspace_preview_app)


@pytest.mark.integration
def test_workspace_context_combines_sources_readiness_capabilities_and_preview_refs(
    workspace_preview_app,
    workspace_preview_db,
    monkeypatch,
):
    monkeypatch.setattr(
        workspaces_endpoint,
        "collect_workspace_service_capabilities",
        _provider_ready_service_capabilities,
        raising=True,
    )
    _install_overrides(workspace_preview_app, workspace_preview_db, _media_db())
    try:
        with TestClient(workspace_preview_app, raise_server_exceptions=False) as client:
            response = client.get("/api/v1/workspaces/ws-preview/context")
    finally:
        _clear_overrides(workspace_preview_app)

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["workspace_id"] == "ws-preview"
    assert payload["workspace_kind"] == "research_workspace"
    assert payload["workspace"]["id"] == "ws-preview"
    assert payload["sources"]["summary"]["total"] == 3
    assert payload["partial_errors"] == []

    sources = {source["id"]: source for source in payload["sources"]["items"]}
    assert sources["src-ready"]["state"] == "queryable"
    assert sources["src-ready"]["readiness"]["citation_ready"] is True
    assert sources["src-ready"]["preview"] == {
        "available": True,
        "detail_href": "/api/v1/workspaces/ws-preview/sources/src-ready/preview",
        "snippet_count": None,
        "total_chars": None,
        "unavailable_reason": None,
    }

    assert sources["src-pending"]["state"] == "extracting"
    assert sources["src-pending"]["preview"]["available"] is False
    assert sources["src-pending"]["preview"]["unavailable_reason"] == "extraction_pending"
    assert payload["capabilities"]["allowed_actions"]["ask_grounded_questions"]["allowed"] is True
    assert payload["services"]["mcp"]["management_surface"] == "mcp_hub"


@pytest.mark.integration
def test_workspace_context_includes_attached_primary_root(
    workspace_preview_app,
    tmp_path,
    monkeypatch,
):
    db = CharactersRAGDB(db_path=str(tmp_path / "project-context-chacha.db"), client_id="user-1")
    db.upsert_workspace("ws-project-context", "Project Workspace")
    db.upsert_workspace_primary_root(
        "ws-project-context",
        {
            "root_id": "root-1",
            "backend": "host_local",
            "display_name": "Local project",
            "absolute_root": "/Users/researcher/local-project",
            "root_state": "attached",
            "indexing_state": "ready",
            "mcp_trust_state": "trusted",
        },
    )

    async def _service_capabilities(*, workspace_id: str, user_id: int | str | None) -> dict[str, Any]:
        _ = (workspace_id, user_id)
        return {
            "workspace_services": {
                "preview": {"state": "available", "reason_code": None},
            },
        }

    monkeypatch.setattr(
        workspaces_endpoint,
        "collect_workspace_service_capabilities",
        _service_capabilities,
        raising=True,
    )
    _install_overrides(workspace_preview_app, db, _MediaPreviewDB({}))
    try:
        with TestClient(workspace_preview_app, raise_server_exceptions=False) as client:
            response = client.get("/api/v1/workspaces/ws-project-context/context")
    finally:
        _clear_overrides(workspace_preview_app)

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["workspace_profile"] == "project"
    assert payload["workspace_kind"] == "project_workspace"
    assert payload["project_root"]["state"] == "attached"
    assert payload["project_root"]["backend"] == "host_local"
    assert payload["project_root"]["display_name"] == "Local project"
    assert payload["project_root"]["path_hint"] == "Local project"
    assert payload["allowed_actions"]["write_files"]["allowed"] is True


@pytest.mark.integration
def test_workspace_context_reports_jobs_partial_error_when_jobs_manager_unavailable(
    workspace_preview_app,
    workspace_preview_db,
):
    _install_overrides(workspace_preview_app, workspace_preview_db, _media_db())
    workspace_preview_app.dependency_overrides[
        workspaces_endpoint.try_get_workspace_job_manager
    ] = lambda: None
    try:
        with TestClient(workspace_preview_app, raise_server_exceptions=False) as client:
            response = client.get("/api/v1/workspaces/ws-preview/context")
    finally:
        _clear_overrides(workspace_preview_app)

    assert response.status_code == 200, response.text
    payload = response.json()
    assert {
        "scope": "jobs",
        "code": "jobs_unavailable",
        "message": "Jobs service is unavailable; in-flight ingestion progress may be incomplete.",
    } in payload["partial_errors"]


@pytest.mark.integration
def test_workspace_context_filters_active_jobs_to_workspace_sources(
    workspace_preview_app,
    workspace_preview_db,
):
    jobs = [
        {
            "id": 101,
            "uuid": "uuid-matched",
            "status": "processing",
            "job_type": "workspace_source_ingest",
            "progress_percent": 55,
            "progress_message": "Chunking ready source",
            "error_code": "unsafe/error code with spaces",
            "payload": {"workspace_id": "ws-preview", "media_id": 1},
        },
        {
            "id": 102,
            "uuid": "uuid-wrong-workspace",
            "status": "processing",
            "job_type": "workspace_source_ingest",
            "progress_percent": 60,
            "progress_message": "Processing matching media in another workspace",
            "payload": {"workspace_id": "ws-other", "media_id": 1},
        },
        {
            "id": 202,
            "uuid": "uuid-unrelated",
            "status": "processing",
            "job_type": "workspace_source_ingest",
            "progress_percent": 20,
            "progress_message": "Processing another workspace source",
            "payload": {"media_id": 777},
        },
        {
            "id": 203,
            "uuid": "uuid-missing-workspace-id",
            "status": "processing",
            "job_type": "workspace_source_ingest",
            "progress_percent": 35,
            "progress_message": "Processing matching media without workspace scope",
            "payload": {"media_id": 1},
        },
        {
            "id": 303,
            "uuid": "uuid-finished",
            "status": "completed",
            "job_type": "workspace_source_ingest",
            "progress_percent": 100,
            "payload": {"media_id": 1},
        },
    ]
    _install_overrides(workspace_preview_app, workspace_preview_db, _media_db(), jobs)
    try:
        with TestClient(workspace_preview_app, raise_server_exceptions=False) as client:
            response = client.get("/api/v1/workspaces/ws-preview/context")
    finally:
        _clear_overrides(workspace_preview_app)

    assert response.status_code == 200, response.text
    payload = response.json()
    assert [job["uuid"] for job in payload["active_jobs"]] == ["uuid-matched"]
    assert payload["active_jobs"][0]["error_code"] is None


@pytest.mark.unit
def test_workspace_job_status_payload_truncates_safe_error_code() -> None:
    safe_code = "a" * 200

    payload = workspaces_endpoint._job_status_payload({"error_code": safe_code})

    assert payload["error_code"] == "a" * 128


@pytest.mark.integration
def test_workspace_context_reports_partial_error_when_media_db_unavailable(
    workspace_preview_app,
    workspace_preview_db,
):
    _install_overrides(workspace_preview_app, workspace_preview_db, None)
    try:
        with TestClient(workspace_preview_app, raise_server_exceptions=False) as client:
            response = client.get("/api/v1/workspaces/ws-preview/context")
    finally:
        _clear_overrides(workspace_preview_app)

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["sources"]["summary"]["missing"] == 3
    assert payload["partial_errors"] == [
        {
            "scope": "sources",
            "code": "media_db_unavailable",
            "message": "Media database is unavailable; source readiness is conservative.",
        }
    ]


@pytest.mark.integration
def test_workspace_source_preview_returns_bounded_content_and_chunk_evidence(
    workspace_preview_app,
    workspace_preview_db,
):
    _install_overrides(workspace_preview_app, workspace_preview_db, _media_db())
    try:
        with TestClient(workspace_preview_app, raise_server_exceptions=False) as client:
            response = client.get(
                "/api/v1/workspaces/ws-preview/sources/src-ready/preview?max_chars=60&chunk_limit=2"
            )
    finally:
        _clear_overrides(workspace_preview_app)

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["source_id"] == "src-ready"
    assert payload["media_id"] == 1
    assert payload["preview_mode"] == "available"
    assert payload["content_available"] is True
    assert payload["text_total_chars"] > 60
    assert len(payload["text_preview"]) == 60
    assert payload["text_truncated"] is True
    assert payload["readiness"]["citation_ready"] is True

    snippets = payload["snippets"]
    assert snippets[0]["kind"] == "content_excerpt"
    assert snippets[0]["text"] == payload["text_preview"]
    assert snippets[1] == {
        "id": "chunk-ready-0",
        "source_id": "src-ready",
        "media_id": 1,
        "kind": "chunk",
        "text": "First chunk with citeable evidence.",
        "start_char": 0,
        "end_char": 35,
        "chunk_index": 0,
        "chunk_uuid": "chunk-ready-0",
        "chunk_type": "text",
    }


@pytest.mark.integration
def test_workspace_source_preview_reports_pending_extraction(
    workspace_preview_app,
    workspace_preview_db,
):
    _install_overrides(workspace_preview_app, workspace_preview_db, _media_db())
    try:
        with TestClient(workspace_preview_app, raise_server_exceptions=False) as client:
            response = client.get("/api/v1/workspaces/ws-preview/sources/src-pending/preview")
    finally:
        _clear_overrides(workspace_preview_app)

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["preview_mode"] == "pending"
    assert payload["content_available"] is False
    assert payload["unavailable_reason"] == "extraction_pending"
    assert payload["text_preview"] is None
    assert payload["snippets"] == []


@pytest.mark.integration
def test_workspace_source_preview_reports_missing_media(
    workspace_preview_app,
    workspace_preview_db,
):
    _install_overrides(workspace_preview_app, workspace_preview_db, _media_db())
    try:
        with TestClient(workspace_preview_app, raise_server_exceptions=False) as client:
            response = client.get("/api/v1/workspaces/ws-preview/sources/src-missing/preview")
    finally:
        _clear_overrides(workspace_preview_app)

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["preview_mode"] == "missing_media"
    assert payload["content_available"] is False
    assert payload["unavailable_reason"] == "media_not_found"
    assert payload["readiness"]["text_extracted"] is False


@pytest.mark.integration
@pytest.mark.parametrize(
    ("workspace", "source", "version", "expected"),
    [
        ("ws-preview", "src-ready", 1, 200),
        ("ws-preview", "src-ready", 7, 404),
        ("ws-preview", "src-ready", 0, 422),
        ("ws-preview", "foreign-source", 1, 404),
        ("foreign-workspace", "src-ready", 1, 404),
        ("removed-workspace", "src-ready", 1, 404),
    ],
)
def test_workspace_pinned_preview_checks_membership_before_version_read(
    workspace_preview_app,
    workspace_preview_db,
    workspace,
    source,
    version,
    expected,
):
    if workspace == "removed-workspace":
        workspace_preview_db.delete_workspace("ws-preview", workspace_preview_db.get_workspace("ws-preview")["version"])
        workspace = "ws-preview"
    db = _media_db()
    calls = []

    def get_version(media_id, version_number=None, include_content=True):
        calls.append((media_id, version_number, include_content))
        return {"version_number": 1, "content": "old snapshot"} if version_number == 1 else None

    db.get_document_version = get_version
    _install_overrides(workspace_preview_app, workspace_preview_db, db)
    try:
        with TestClient(workspace_preview_app, raise_server_exceptions=False) as client:
            response = client.get(f"/api/v1/workspaces/{workspace}/sources/{source}/preview?version_number={version}")
    finally:
        _clear_overrides(workspace_preview_app)
    assert response.status_code == expected, response.text
    if expected == 200:
        payload = response.json()
        assert payload["document_version_number"] == 1
        assert payload["text_preview"] == "old snapshot"
        assert all(item["kind"] != "chunk" for item in payload["snippets"])
    if (
        expected in {200, 404}
        and workspace == "ws-preview"
        and source == "src-ready"
        and workspace_preview_db.get_workspace(workspace)
    ):
        assert calls == [(1, version, True)]
    else:
        assert calls == []
