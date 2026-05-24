from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import try_get_media_db_for_user
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
from tldw_Server_API.app.api.v1.endpoints import workspaces as workspaces_endpoint
from tldw_Server_API.app.api.v1.endpoints.workspaces_rate_limit_policy import WORKSPACES_READ_RATE_LIMIT
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, InputError
from tldw_Server_API.app.core.Workspaces import status_projection
from tldw_Server_API.app.core.Workspaces.status_projection import (
    build_source_status_projection,
    build_workspace_capability_projection,
)


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

    def has_unvectorized_chunks(self, media_id: int) -> bool:
        return media_id in self.unvectorized


class _JobManagerDouble:
    def __init__(self, jobs: list[dict[str, Any]] | None = None) -> None:
        self.jobs = jobs or []

    def list_jobs(self, **kwargs: Any) -> list[dict[str, Any]]:
        _ = kwargs
        return list(self.jobs)


@pytest.fixture
def workspace_status_db(tmp_path: Path) -> CharactersRAGDB:
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
def workspace_status_app() -> Any:
    from tldw_Server_API.app.main import app

    return app


async def _allow_rate_limit() -> None:
    return None


async def _user() -> SimpleNamespace:
    return SimpleNamespace(
        id=1,
        username="testuser",
        email="test@example.com",
        roles=["admin"],
        is_admin=True,
    )


def _install_overrides(
    app: Any,
    db: Any,
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
    workspace_status_app: Any,
    workspace_status_db: CharactersRAGDB,
) -> None:
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
        unvectorized={2},
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
def test_workspace_capabilities_fail_closed_without_queryable_sources(
    workspace_status_app: Any,
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(db_path=str(tmp_path / "empty-chacha.db"), client_id="user-1")
    db.upsert_workspace("ws-empty", "Empty Workspace")
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
def test_workspace_sources_status_uses_active_media_ingest_job_progress(
    workspace_status_app: Any,
    tmp_path: Path,
) -> None:
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


def test_recent_media_ingest_jobs_prioritizes_workspace_source_jobs() -> None:
    class _PrioritizedJobManager:
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
                    }
                ]
            return [
                {
                    "id": 2,
                    "domain": "media_ingest",
                    "queue": "default",
                    "job_type": "media_ingest_item",
                },
                {
                    "id": 1,
                    "domain": "media_ingest",
                    "queue": "default",
                    "job_type": "workspace_source_ingest",
                },
            ]

    jm = _PrioritizedJobManager()
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

    monkeypatch.setattr(workspaces_endpoint, "get_job_manager", _raise_job_manager)

    assert workspaces_endpoint.try_get_workspace_job_manager() is None


def test_workspace_capabilities_honor_viewer_access_level() -> None:
    payload = build_workspace_capability_projection(
        workspace={"id": "ws-viewer", "access_level": "viewer"},
        status_projection={
            "summary": {
                "total": 1,
                "selected": 1,
                "queryable": 1,
                "partially_queryable": 0,
                "processing": 0,
                "failed": 0,
                "missing": 0,
            }
        },
    )

    assert payload["access_level"] == "viewer"
    assert payload["allowed_actions"]["inspect_sources"]["allowed"] is True
    assert payload["allowed_actions"]["ask_grounded_questions"]["allowed"] is True
    assert payload["allowed_actions"]["add_sources"] == {
        "allowed": False,
        "reason_code": "insufficient_access",
    }
    assert payload["allowed_actions"]["export_workspace"] == {
        "allowed": False,
        "reason_code": "insufficient_access",
    }
    assert payload["allowed_actions"]["manage_tools"] == {
        "allowed": False,
        "reason_code": "insufficient_access",
    }


def test_failed_job_status_does_not_expose_raw_error_text() -> None:
    payload = build_source_status_projection(
        workspace_id="ws-errors",
        sources=[
            {
                "id": "src-failed",
                "workspace_id": "ws-errors",
                "media_id": 71,
                "title": "Failed import",
                "source_type": "pdf",
                "selected": True,
            }
        ],
        jobs=[
            {
                "id": 12,
                "uuid": "job-12",
                "status": "failed",
                "job_type": "workspace_source_ingest",
                "payload": {"workspace_source_id": "src-failed", "media_id": 71},
                "result": {"error": "sqlite stack trace with /private/path"},
                "error_message": "provider secret leaked",
                "progress_percent": 25,
                "created_at": "2026-05-23T12:00:00Z",
            }
        ],
    )

    source = payload["sources"][0]
    assert source["state"] == "failed"
    assert source["progress_message"] == "Ingestion job failed."
    assert source["job"]["error_message"] is None


def test_source_job_matching_uses_specificity_order() -> None:
    source = {
        "id": "src-specific",
        "workspace_id": "ws-match",
        "media_id": 88,
        "title": "Shared Title",
        "source_type": "web",
        "url": "https://example.test/shared",
        "selected": True,
    }

    assert status_projection._source_match_keys(source) == [
        "id:src-specific",
        "media:88",
        "url:https://example.test/shared",
        "title:Shared Title",
    ]

    payload = build_source_status_projection(
        workspace_id="ws-match",
        sources=[source],
        jobs=[
            {
                "id": 90,
                "status": "failed",
                "job_type": "media_ingest_item",
                "payload": {"source": "https://example.test/shared"},
                "created_at": "2026-05-23T13:00:00Z",
            },
            {
                "id": 91,
                "status": "queued",
                "job_type": "workspace_source_ingest",
                "payload": {"workspace_source_id": "src-specific"},
                "created_at": "2026-05-23T12:00:00Z",
            },
        ],
    )

    assert payload["sources"][0]["state"] == "queued"
    assert payload["sources"][0]["job"]["id"] == 91


@pytest.mark.integration
def test_workspace_sources_status_maps_chacha_input_errors_to_400(
    workspace_status_app: Any,
) -> None:
    class _InputErrorDB:
        def get_workspace(self, workspace_id: str) -> dict[str, Any]:
            return {"id": workspace_id}

        def list_workspace_sources(self, workspace_id: str) -> list[dict[str, Any]]:
            _ = workspace_id
            raise InputError("invalid workspace source filter")

    _install_overrides(workspace_status_app, _InputErrorDB(), _MediaStatusDB({}))
    try:
        with TestClient(workspace_status_app, raise_server_exceptions=False) as client:
            response = client.get("/api/v1/workspaces/ws-error/sources/status")
    finally:
        _clear_overrides(workspace_status_app)

    assert response.status_code == 400, response.text
    assert response.json()["detail"] == "invalid workspace source filter"
