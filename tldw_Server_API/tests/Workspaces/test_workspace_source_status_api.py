from __future__ import annotations

from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import try_get_media_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user
from tldw_Server_API.app.api.v1.endpoints import workspaces as workspaces_endpoint
from tldw_Server_API.app.api.v1.endpoints.workspaces_rate_limit_policy import WORKSPACES_READ_RATE_LIMIT
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


class _MediaStatusDB:
    def __init__(self, rows: dict[int, dict[str, Any]], unvectorized: set[int] | None = None) -> None:
        self.rows = rows
        self.unvectorized = unvectorized or set()

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
        return media_id in self.unvectorized


class _JobManagerDouble:
    def __init__(self, jobs: list[dict[str, Any]] | None = None) -> None:
        self.jobs = jobs or []

    def list_jobs(self, **kwargs: Any) -> list[dict[str, Any]]:
        _ = kwargs
        return list(self.jobs)


@pytest.fixture
def workspace_status_db(tmp_path):
    db = CharactersRAGDB(db_path=str(tmp_path / "chacha.db"), client_id="user-1")
    db.upsert_workspace("ws-status", "Source Status")
    db.add_workspace_source(
        "ws-status",
        {
            "id": "src-ready",
            "media_id": 1,
            "title": "Ready paper",
            "source_type": "pdf",
            "position": 0,
            "selected": True,
        },
    )
    db.add_workspace_source(
        "ws-status",
        {
            "id": "src-indexing",
            "media_id": 2,
            "title": "Indexing article",
            "source_type": "web",
            "position": 1,
            "selected": True,
        },
    )
    db.add_workspace_source(
        "ws-status",
        {
            "id": "src-missing",
            "media_id": 99,
            "title": "Missing upload",
            "source_type": "docx",
            "position": 2,
            "selected": True,
        },
    )
    return db


@pytest.fixture
def workspace_status_app():
    app = FastAPI()
    app.include_router(workspaces_endpoint.router, prefix="/api/v1/workspaces")
    return app


async def _allow_rate_limit() -> None:
    return None


async def _user() -> SimpleNamespace:
    return SimpleNamespace(id=1, username="testuser", email="test@example.com", roles=["admin"], is_admin=True)


async def _empty_service_capabilities(
    *,
    workspace_id: str,
    user_id: int | str | None,
) -> dict[str, Any]:
    _ = (workspace_id, user_id)
    return {}


def _install_overrides(
    app: Any,
    db: CharactersRAGDB,
    media_db: _MediaStatusDB,
    jobs: list[dict[str, Any]] | None = None,
) -> None:
    async def _media_db() -> AsyncGenerator[_MediaStatusDB, None]:
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


@pytest.mark.integration
def test_workspace_sources_status_reports_readiness_and_missing_media(
    workspace_status_app,
    workspace_status_db,
):
    media_db = _MediaStatusDB(
        {
            1: {
                "id": 1,
                "title": "Ready paper",
                "type": "pdf",
                "content": "Grounded evidence text.",
                "chunking_status": "completed",
                "vector_processing": 1,
            },
            2: {
                "id": 2,
                "title": "Indexing article",
                "type": "web",
                "content": "Extracted but still being indexed.",
                "chunking_status": "completed",
                "vector_processing": 0,
            },
        },
        unvectorized={1, 2},
    )
    _install_overrides(workspace_status_app, workspace_status_db, media_db)
    try:
        with TestClient(workspace_status_app, raise_server_exceptions=False) as client:
            response = client.get("/api/v1/workspaces/ws-status/sources/status")
    finally:
        _clear_overrides(workspace_status_app)

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["workspace_id"] == "ws-status"
    assert payload["summary"] == {
        "total": 3,
        "selected": 3,
        "queryable": 1,
        "partially_queryable": 1,
        "processing": 1,
        "failed": 0,
        "missing": 1,
    }

    sources = {source["id"]: source for source in payload["sources"]}
    assert sources["src-ready"]["state"] == "queryable"
    assert sources["src-ready"]["readiness"]["text_extracted"] is True
    assert sources["src-ready"]["readiness"]["fts_ready"] is True
    assert sources["src-ready"]["readiness"]["vector_ready"] is True
    assert sources["src-ready"]["progress_percent"] == 100

    assert sources["src-indexing"]["state"] == "partially_queryable"
    assert sources["src-indexing"]["readiness"]["fts_ready"] is True
    assert sources["src-indexing"]["readiness"]["vector_ready"] is False
    assert sources["src-indexing"]["status_reason"] == "vector_index_pending"

    assert sources["src-missing"]["state"] == "missing_media"
    assert sources["src-missing"]["readiness"]["text_extracted"] is False
    assert sources["src-missing"]["status_reason"] == "media_not_found"


@pytest.mark.integration
def test_workspace_source_status_treats_vector_ready_media_as_queryable_when_chunking_marker_is_stale(
    workspace_status_app,
    tmp_path,
):
    db = CharactersRAGDB(db_path=str(tmp_path / "stale-chunking.db"), client_id="user-1")
    db.upsert_workspace("ws-stale", "Stale Chunking")
    db.add_workspace_source(
        "ws-stale",
        {
            "id": "src-stale",
            "media_id": 10,
            "title": "Vector-ready source",
            "source_type": "document",
            "position": 0,
            "selected": True,
        },
    )
    media_db = _MediaStatusDB(
        {
            10: {
                "id": 10,
                "title": "Vector-ready source",
                "type": "document",
                "content": "Indexed evidence text.",
                "chunking_status": "pending",
                "vector_processing": 1,
            },
        }
    )
    _install_overrides(workspace_status_app, db, media_db)
    try:
        with TestClient(workspace_status_app, raise_server_exceptions=False) as client:
            response = client.get("/api/v1/workspaces/ws-stale/sources/status")
    finally:
        _clear_overrides(workspace_status_app)

    assert response.status_code == 200, response.text
    source = response.json()["sources"][0]
    assert source["state"] == "queryable"
    assert source["status_reason"] == "source_queryable"
    assert source["readiness"]["vector_ready"] is True


@pytest.mark.integration
def test_workspace_capabilities_fail_closed_without_queryable_sources(
    workspace_status_app,
    tmp_path,
    monkeypatch,
):
    db = CharactersRAGDB(db_path=str(tmp_path / "empty-chacha.db"), client_id="user-1")
    db.upsert_workspace("ws-empty", "Empty Workspace")
    monkeypatch.setattr(
        workspaces_endpoint,
        "collect_workspace_service_capabilities",
        _empty_service_capabilities,
        raising=False,
    )
    _install_overrides(workspace_status_app, db, _MediaStatusDB({}))
    try:
        with TestClient(workspace_status_app, raise_server_exceptions=False) as client:
            response = client.get("/api/v1/workspaces/ws-empty/capabilities")
    finally:
        _clear_overrides(workspace_status_app)

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["workspace_id"] == "ws-empty"
    assert payload["workspace_kind"] == "research_workspace"
    assert payload["access_level"] == "owner"
    assert payload["source_summary"]["total"] == 0
    assert payload["allowed_actions"]["add_sources"]["allowed"] is True
    assert payload["allowed_actions"]["ask_grounded_questions"] == {
        "allowed": False,
        "reason_code": "no_queryable_sources",
    }
    assert payload["allowed_actions"]["run_mcp_tools"] == {
        "allowed": False,
        "reason_code": "mcp_not_configured",
    }
    assert payload["workspace_services"]["mcp"]["management_surface"] == "mcp_hub"
    assert payload["workspace_services"]["acp"]["state"] == "not_configured"
    assert payload["workspace_services"]["sandbox"]["state"] == "not_configured"


@pytest.mark.integration
def test_workspace_capabilities_accept_project_workspace_profile(
    workspace_status_app,
    tmp_path,
    monkeypatch,
):
    db = CharactersRAGDB(db_path=str(tmp_path / "project-chacha.db"), client_id="user-1")
    db.upsert_workspace("ws-project", "Project Workspace", workspace_profile="project")
    monkeypatch.setattr(
        workspaces_endpoint,
        "collect_workspace_service_capabilities",
        _empty_service_capabilities,
        raising=False,
    )
    _install_overrides(workspace_status_app, db, _MediaStatusDB({}))
    try:
        with TestClient(workspace_status_app, raise_server_exceptions=False) as client:
            response = client.get("/api/v1/workspaces/ws-project/capabilities")
    finally:
        _clear_overrides(workspace_status_app)

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["workspace_profile"] == "project"
    assert payload["workspace_kind"] == "project_workspace"
    assert payload["project_root"]["state"] == "not_configured"
    assert payload["resolution"]["status"] == "complete"


@pytest.mark.integration
def test_workspace_capabilities_include_attached_primary_root(
    workspace_status_app,
    tmp_path,
    monkeypatch,
):
    db = CharactersRAGDB(db_path=str(tmp_path / "project-root-chacha.db"), client_id="user-1")
    db.upsert_workspace("ws-project-root", "Project Workspace")
    db.upsert_workspace_primary_root(
        "ws-project-root",
        {
            "root_id": "root-1",
            "backend": "sandbox_volume",
            "display_name": "Sandbox project",
            "sandbox_volume_id": "volume-1",
            "root_state": "attached",
            "indexing_state": "ready",
            "sandbox_mount_state": "mounted",
            "mcp_trust_state": "trusted",
        },
    )

    async def _service_capabilities(*, workspace_id: str, user_id: int | str | None) -> dict[str, Any]:
        _ = (workspace_id, user_id)
        return {
            "workspace_services": {
                "sandbox": {"state": "available", "reason_code": None},
                "preview": {"state": "available", "reason_code": None},
            },
            "allowed_actions": {
                "use_sandbox": {"allowed": True, "reason_code": None},
            },
        }

    monkeypatch.setattr(
        workspaces_endpoint,
        "collect_workspace_service_capabilities",
        _service_capabilities,
        raising=False,
    )
    _install_overrides(workspace_status_app, db, _MediaStatusDB({}))
    try:
        with TestClient(workspace_status_app, raise_server_exceptions=False) as client:
            response = client.get("/api/v1/workspaces/ws-project-root/capabilities")
    finally:
        _clear_overrides(workspace_status_app)

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["workspace_profile"] == "project"
    assert payload["workspace_kind"] == "project_workspace"
    assert payload["project_root"]["state"] == "attached"
    assert payload["project_root"]["backend"] == "sandbox_volume"
    assert payload["project_root"]["path_hint"] == "volume-1"
    assert payload["project_root"]["sandbox_mount_state"] == "mounted"
    assert payload["allowed_actions"]["write_files"]["allowed"] is True
    assert payload["allowed_actions"]["run_sandbox"]["allowed"] is True


@pytest.mark.integration
def test_workspace_capabilities_reflect_service_capability_projection(
    workspace_status_app,
    workspace_status_db,
    monkeypatch,
):
    media_db = _MediaStatusDB(
        {
            1: {
                "id": 1,
                "title": "Ready paper",
                "type": "pdf",
                "content": "Grounded evidence text.",
                "chunking_status": "completed",
                "vector_processing": 1,
            },
            2: {
                "id": 2,
                "title": "Indexing article",
                "type": "web",
                "content": "Extracted but still being indexed.",
                "chunking_status": "completed",
                "vector_processing": 0,
            },
        },
        unvectorized={1, 2},
    )

    async def _service_capabilities(*, workspace_id: str, user_id: int | str | None) -> dict[str, Any]:
        assert workspace_id == "ws-status"
        assert user_id == 1
        return {
            "workspace_services": {
                "mcp": {
                    "state": "available",
                    "reason_code": None,
                    "management_surface": "mcp_hub",
                },
                "acp": {
                    "state": "needs_approval",
                    "reason_code": "acp_approval_required",
                    "management_surface": "acp_workspace",
                },
                "sandbox": {
                    "state": "blocked",
                    "reason_code": "sandbox_runtime_unavailable",
                    "management_surface": "sandbox_settings",
                },
                "provider": {
                    "state": "degraded",
                    "reason_code": "external_provider_only",
                    "management_surface": "model_settings",
                },
            },
            "allowed_actions": {
                "run_mcp_tools": {"allowed": True, "reason_code": None},
                "use_acp_agents": {
                    "allowed": False,
                    "reason_code": "acp_approval_required",
                },
                "use_sandbox": {
                    "allowed": False,
                    "reason_code": "sandbox_runtime_unavailable",
                },
            },
        }

    monkeypatch.setattr(
        workspaces_endpoint,
        "collect_workspace_service_capabilities",
        _service_capabilities,
        raising=False,
    )
    _install_overrides(workspace_status_app, workspace_status_db, media_db)
    try:
        with TestClient(workspace_status_app, raise_server_exceptions=False) as client:
            response = client.get("/api/v1/workspaces/ws-status/capabilities")
    finally:
        _clear_overrides(workspace_status_app)

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["workspace_services"]["mcp"] == {
        "state": "available",
        "reason_code": None,
        "management_surface": "mcp_hub",
    }
    assert payload["allowed_actions"]["run_mcp_tools"] == {
        "allowed": True,
        "reason_code": None,
    }
    assert payload["workspace_services"]["acp"] == {
        "state": "needs_approval",
        "reason_code": "acp_approval_required",
        "management_surface": "acp_workspace",
    }
    assert payload["allowed_actions"]["use_acp_agents"] == {
        "allowed": False,
        "reason_code": "project_root_not_configured",
    }
    assert payload["workspace_services"]["sandbox"] == {
        "state": "blocked",
        "reason_code": "sandbox_runtime_unavailable",
        "management_surface": "sandbox_settings",
    }
    assert payload["workspace_services"]["provider"] == {
        "state": "degraded",
        "reason_code": "external_provider_only",
        "management_surface": "model_settings",
    }


@pytest.mark.integration
def test_workspace_sources_status_uses_active_media_ingest_job_progress(
    workspace_status_app,
    tmp_path,
):
    db = CharactersRAGDB(db_path=str(tmp_path / "jobs-chacha.db"), client_id="user-1")
    db.upsert_workspace("ws-jobs", "Jobs Workspace")
    db.add_workspace_source(
        "ws-jobs",
        {
            "id": "src-running",
            "media_id": 7,
            "title": "Running import",
            "source_type": "web",
            "url": "https://example.test/running",
            "position": 0,
            "selected": True,
        },
    )
    jobs = [
        {
            "id": 42,
            "uuid": "job-42",
            "domain": "media_ingest",
            "job_type": "media_ingest_item",
            "status": "processing",
            "payload": {"source": "https://example.test/running"},
            "result": None,
            "progress_percent": 82.5,
            "progress_message": "vector indexing",
            "created_at": "2026-05-23T12:00:00Z",
        },
    ]
    _install_overrides(workspace_status_app, db, _MediaStatusDB({}), jobs=jobs)
    try:
        with TestClient(workspace_status_app, raise_server_exceptions=False) as client:
            response = client.get("/api/v1/workspaces/ws-jobs/sources/status")
    finally:
        _clear_overrides(workspace_status_app)

    assert response.status_code == 200, response.text
    source = response.json()["sources"][0]
    assert source["state"] == "indexing"
    assert source["progress_percent"] == 82.5
    assert source["job"]["id"] == 42
    assert source["job"]["progress_message"] == "vector indexing"


@pytest.mark.integration
def test_workspace_sources_status_prefers_extracted_media_over_workspace_lifecycle_job(
    workspace_status_app,
    tmp_path,
):
    db = CharactersRAGDB(db_path=str(tmp_path / "workspace-job-chacha.db"), client_id="user-1")
    db.upsert_workspace("ws-workspace-job", "Workspace Job")
    db.add_workspace_source(
        "ws-workspace-job",
        {
            "id": "src-extracted",
            "media_id": 8,
            "title": "Extracted upload",
            "source_type": "document",
            "position": 0,
            "selected": True,
        },
    )
    jobs = [
        {
            "id": 84,
            "uuid": "job-84",
            "domain": "media_ingest",
            "job_type": "workspace_source_ingest",
            "status": "processing",
            "payload": {
                "workspace_id": "ws-workspace-job",
                "workspace_source_id": "src-extracted",
                "media_id": 8,
            },
            "result": None,
            "progress_percent": 10,
            "progress_message": "validate source",
            "created_at": "2026-05-23T12:00:00Z",
        },
    ]
    media_db = _MediaStatusDB(
        {
            8: {
                "id": 8,
                "title": "Extracted upload",
                "type": "document",
                "content": "Extracted source text is already available.",
                "chunking_status": "completed",
                "vector_processing": 0,
            }
        }
    )
    _install_overrides(workspace_status_app, db, media_db, jobs=jobs)
    try:
        with TestClient(workspace_status_app, raise_server_exceptions=False) as client:
            response = client.get("/api/v1/workspaces/ws-workspace-job/sources/status")
    finally:
        _clear_overrides(workspace_status_app)

    assert response.status_code == 200, response.text
    source = response.json()["sources"][0]
    assert source["state"] == "partially_queryable"
    assert source["status_reason"] == "vector_index_pending"
    assert source["readiness"]["text_extracted"] is True
    assert source["job"] is None


@pytest.mark.integration
def test_workspace_sources_status_uses_lightweight_media_status_without_content_load(
    workspace_status_app,
    tmp_path,
):
    class _LightweightMediaStatusDB:
        def get_media_status_by_id(
            self,
            media_id: int,
            *,
            include_deleted: bool = False,
            include_trash: bool = False,
        ) -> dict[str, Any] | None:
            _ = (include_deleted, include_trash)
            if media_id != 12:
                return None
            return {
                "id": 12,
                "title": "Lightweight source",
                "type": "document",
                "has_content": True,
                "chunking_status": "completed",
                "vector_processing": 1,
            }

        def get_media_by_id(self, *_args: Any, **_kwargs: Any) -> dict[str, Any] | None:
            raise RuntimeError("full media content should not be loaded for status")

    db = CharactersRAGDB(db_path=str(tmp_path / "lightweight-status.db"), client_id="user-1")
    db.upsert_workspace("ws-lightweight", "Lightweight Status")
    db.add_workspace_source(
        "ws-lightweight",
        {
            "id": "src-lightweight",
            "media_id": 12,
            "title": "Lightweight source",
            "source_type": "document",
            "position": 0,
            "selected": True,
        },
    )
    _install_overrides(workspace_status_app, db, _LightweightMediaStatusDB())
    try:
        with TestClient(workspace_status_app, raise_server_exceptions=False) as client:
            response = client.get("/api/v1/workspaces/ws-lightweight/sources/status")
    finally:
        _clear_overrides(workspace_status_app)

    assert response.status_code == 200, response.text
    source = response.json()["sources"][0]
    assert source["state"] == "queryable"
    assert source["readiness"]["text_extracted"] is True
    assert source["readiness"]["vector_ready"] is True


def test_recent_media_ingest_jobs_filters_to_supported_media_ingest_jobs() -> None:
    class _MediaIngestJobManager:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def list_jobs(self, **kwargs: Any) -> list[dict[str, Any]]:
            self.calls.append(kwargs)
            if kwargs.get("job_type") == "workspace_source_ingest":
                return [
                    {
                        "id": 1,
                        "domain": "media_ingest",
                        "queue": "default",
                        "job_type": "workspace_source_ingest",
                    },
                ]
            return [
                {
                    "id": 2,
                    "domain": "media_ingest",
                    "queue": "default",
                    "job_type": "media_ingest_item",
                },
            ]

    jm = _MediaIngestJobManager()
    jobs = workspaces_endpoint._list_recent_media_ingest_jobs(jm, SimpleNamespace(id=1))

    assert [job["id"] for job in jobs] == [1, 2]
    assert jm.calls[0]["domain"] == "media_ingest"
    assert jm.calls[0]["queue"] == "default"
    assert jm.calls[0]["job_type"] == "workspace_source_ingest"
    assert jm.calls[1]["domain"] == "media_ingest"
    assert "job_type" not in jm.calls[1]


def test_workspace_job_manager_resolver_fails_open(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_job_manager() -> None:
        raise RuntimeError("jobs manager unavailable")

    monkeypatch.setattr(workspaces_endpoint, "try_get_job_manager", _raise_job_manager)

    assert workspaces_endpoint.try_get_workspace_job_manager() is None
