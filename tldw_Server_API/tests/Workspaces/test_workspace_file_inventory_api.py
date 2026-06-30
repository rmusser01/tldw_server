from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user
from tldw_Server_API.app.api.v1.endpoints import workspaces as workspaces_endpoint
from tldw_Server_API.app.api.v1.endpoints.workspaces_rate_limit_policy import (
    WORKSPACES_READ_RATE_LIMIT,
    WORKSPACES_WRITE_RATE_LIMIT,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Workspaces.file_inventory_ignore import build_inventory_ignore_policy


pytestmark = pytest.mark.integration


class _RecordingJobs:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.created_jobs: list[dict[str, Any]] = []

    def create_job(self, **kwargs: Any) -> dict[str, Any]:
        if self.fail:
            raise RuntimeError("jobs unavailable")
        idempotency_key = kwargs.get("idempotency_key")
        if idempotency_key:
            for row in self.created_jobs:
                if row.get("idempotency_key") == idempotency_key:
                    return row
        job_id = len(self.created_jobs) + 700
        row = {
            "id": job_id,
            "uuid": f"job-{job_id}",
            "status": "queued",
            "progress_percent": 0,
            "progress_message": "queued",
            **kwargs,
        }
        self.created_jobs.append(row)
        return row

    def get_job(self, job_id: int) -> dict[str, Any] | None:
        for row in self.created_jobs:
            if int(row["id"]) == int(job_id):
                return row
        return None


@pytest.fixture
def db(tmp_path: Path) -> Iterator[CharactersRAGDB]:
    chacha = CharactersRAGDB(str(tmp_path / "workspace-file-inventory-api.db"), client_id="workspace-api-tests")
    try:
        yield chacha
    finally:
        chacha.close_connection()


@pytest.fixture
def workspace_app() -> FastAPI:
    app = FastAPI()
    app.include_router(workspaces_endpoint.router, prefix="/api/v1/workspaces")
    return app


async def _allow_rate_limit() -> None:
    return None


async def _user() -> SimpleNamespace:
    return SimpleNamespace(id=1, username="testuser", email="test@example.com", roles=["admin"], is_admin=True)


@contextlib.contextmanager
def _inventory_client(
    app: FastAPI,
    db: CharactersRAGDB,
    *,
    jobs: _RecordingJobs | None,
) -> Iterator[TestClient]:
    app.dependency_overrides[get_request_user] = _user
    app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    app.dependency_overrides[workspaces_endpoint.try_get_workspace_job_manager] = lambda: jobs
    app.dependency_overrides[WORKSPACES_READ_RATE_LIMIT] = _allow_rate_limit
    app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client
    finally:
        app.dependency_overrides.pop(get_request_user, None)
        app.dependency_overrides.pop(get_chacha_db_for_user, None)
        app.dependency_overrides.pop(workspaces_endpoint.try_get_workspace_job_manager, None)
        app.dependency_overrides.pop(WORKSPACES_READ_RATE_LIMIT, None)
        app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)


def _workspace_with_root(db: CharactersRAGDB, root_path: Path) -> dict[str, Any]:
    db.upsert_workspace("ws-inventory", "Inventory Workspace", workspace_profile="project")
    root_path.mkdir(parents=True, exist_ok=True)
    return db.upsert_workspace_primary_root(
        "ws-inventory",
        {
            "root_id": "root-1",
            "backend": "host_local",
            "absolute_root": str(root_path),
            "root_state": "attached",
            "display_name": "Project",
        },
    )


def _completed_scan(
    db: CharactersRAGDB,
    root: dict[str, Any],
    *,
    state: str = "current",
    policy_fingerprint: str | None = None,
    diagnostics: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    policy = build_inventory_ignore_policy()
    scan = db.begin_workspace_file_inventory_scan(
        "ws-inventory",
        root["root_id"],
        int(root["version"]),
        policy_fingerprint or policy.fingerprint,
        requested_by="user-1",
    )
    db.attach_workspace_file_inventory_job(scan["scan_id"], {"id": 701, "uuid": "job-701"})
    return db.complete_workspace_file_inventory_scan(
        scan["scan_id"],
        state,
        {"files": 2, "directories": 1, "ignored": 1, "diagnostics": len(diagnostics or [])},
        diagnostics or [],
        root_snapshot_token="root-snapshot",
    )


def test_scan_endpoint_enqueues_identifier_only_job_and_returns_status(
    workspace_app: FastAPI,
    db: CharactersRAGDB,
    tmp_path: Path,
) -> None:
    root_path = tmp_path / "project"
    root = _workspace_with_root(db, root_path)
    jobs = _RecordingJobs()

    with _inventory_client(workspace_app, db, jobs=jobs) as client:
        response = client.post(
            "/api/v1/workspaces/ws-inventory/file-inventory/scan",
            json={"expected_root_version": root["version"]},
        )

    assert response.status_code == 202, response.text
    payload = response.json()
    assert payload["workspace_id"] == "ws-inventory"
    assert payload["root_id"] == "root-1"
    assert payload["state"] == "queued"
    assert payload["last_scan_id"]
    assert payload["job"]["id"] == jobs.created_jobs[0]["id"]
    assert payload["job"]["progress_message"] == "queued"
    assert payload["counts"]["files"] == 0
    assert str(root_path) not in response.text
    assert str(root_path) not in str(jobs.created_jobs[0]["payload"])


def test_scan_endpoint_maps_no_root_version_conflict_and_jobs_unavailable(
    workspace_app: FastAPI,
    db: CharactersRAGDB,
    tmp_path: Path,
) -> None:
    db.upsert_workspace("ws-inventory", "Inventory Workspace", workspace_profile="project")
    with _inventory_client(workspace_app, db, jobs=_RecordingJobs()) as client:
        no_root = client.post("/api/v1/workspaces/ws-inventory/file-inventory/scan", json={})
    assert no_root.status_code == 409

    root = _workspace_with_root(db, tmp_path / "project")
    with _inventory_client(workspace_app, db, jobs=_RecordingJobs()) as client:
        mismatch = client.post(
            "/api/v1/workspaces/ws-inventory/file-inventory/scan",
            json={"expected_root_version": int(root["version"]) + 1},
        )
    assert mismatch.status_code == 409

    with _inventory_client(workspace_app, db, jobs=None) as client:
        unavailable = client.post(
            "/api/v1/workspaces/ws-inventory/file-inventory/scan",
            json={"expected_root_version": root["version"]},
        )
    assert unavailable.status_code == 503
    assert "jobs unavailable" in unavailable.text.lower()
    assert str(tmp_path) not in unavailable.text


def test_scan_force_semantics_return_current_or_create_new_scan(
    workspace_app: FastAPI,
    db: CharactersRAGDB,
    tmp_path: Path,
) -> None:
    root = _workspace_with_root(db, tmp_path / "project")
    completed = _completed_scan(db, root)
    jobs = _RecordingJobs()

    with _inventory_client(workspace_app, db, jobs=jobs) as client:
        current = client.post("/api/v1/workspaces/ws-inventory/file-inventory/scan", json={"force": False})

    assert current.status_code == 200, current.text
    current_payload = current.json()
    assert current_payload["state"] == "current"
    assert current_payload["last_scan_id"] == completed["scan_id"]
    assert current_payload["job"] is None
    assert jobs.created_jobs == []

    with _inventory_client(workspace_app, db, jobs=jobs) as client:
        forced = client.post("/api/v1/workspaces/ws-inventory/file-inventory/scan", json={"force": True})

    assert forced.status_code == 202, forced.text
    forced_payload = forced.json()
    assert forced_payload["state"] == "queued"
    assert forced_payload["last_scan_id"] != completed["scan_id"]
    assert len(jobs.created_jobs) == 1


def test_scan_reuses_active_scan_only_after_job_is_attached(
    workspace_app: FastAPI,
    db: CharactersRAGDB,
    tmp_path: Path,
) -> None:
    root = _workspace_with_root(db, tmp_path / "project")
    policy = build_inventory_ignore_policy()
    orphaned = db.begin_workspace_file_inventory_scan(
        "ws-inventory",
        root["root_id"],
        int(root["version"]),
        policy.fingerprint,
        requested_by="user-1",
    )
    jobs = _RecordingJobs()

    with _inventory_client(workspace_app, db, jobs=jobs) as client:
        first = client.post("/api/v1/workspaces/ws-inventory/file-inventory/scan", json={})
        second = client.post("/api/v1/workspaces/ws-inventory/file-inventory/scan", json={})

    assert first.status_code == 202, first.text
    assert second.status_code == 202, second.text
    first_payload = first.json()
    second_payload = second.json()
    assert first_payload["last_scan_id"] != orphaned["scan_id"]
    assert second_payload["last_scan_id"] == first_payload["last_scan_id"]
    assert len(jobs.created_jobs) == 1


def test_status_endpoint_returns_bounded_redacted_failed_diagnostics(
    workspace_app: FastAPI,
    db: CharactersRAGDB,
    tmp_path: Path,
) -> None:
    root = _workspace_with_root(db, tmp_path / "project")
    diagnostics = [
        {
            "code": "scan_warning",
            "message": "Cannot inspect path",
            "path_hint": str(tmp_path / "project" / f"private-{index}.txt"),
        }
        for index in range(60)
    ]
    failed = _completed_scan(db, root, state="failed", diagnostics=diagnostics)
    jobs = _RecordingJobs()
    jobs.created_jobs.append(
        {
            "id": 701,
            "uuid": "job-701",
            "status": "failed",
            "job_type": "workspace_file_inventory_scan",
            "progress_percent": 100,
            "progress_message": "failed",
            "error_message": f"Traceback reading {tmp_path / 'project' / 'private.txt'}",
        }
    )

    with _inventory_client(workspace_app, db, jobs=jobs) as client:
        response = client.get("/api/v1/workspaces/ws-inventory/file-inventory/status")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["workspace_id"] == "ws-inventory"
    assert payload["state"] == "failed"
    assert payload["last_scan_id"] == failed["scan_id"]
    assert payload["job"]["error_message"] is None
    assert len(payload["diagnostics"]) == 50
    assert str(tmp_path) not in response.text


def test_items_endpoint_paginates_filters_and_returns_relative_paths_only(
    workspace_app: FastAPI,
    db: CharactersRAGDB,
    tmp_path: Path,
) -> None:
    root = _workspace_with_root(db, tmp_path / "project")
    completed = _completed_scan(db, root)
    db.replace_workspace_file_inventory_items(
        "ws-inventory",
        root["root_id"],
        completed["scan_id"],
        [
            {"relative_path": "src", "entry_kind": "directory", "indexing_candidate": False},
            {"relative_path": "src/b.py", "entry_kind": "file", "size_bytes": 2, "ignored": True},
            {"relative_path": "src/a.py", "entry_kind": "file", "size_bytes": 1, "indexing_candidate": True},
            {"relative_path": "src/c.py", "entry_kind": "file", "size_bytes": 3, "indexing_candidate": True},
        ],
        scan_coverage_complete=True,
    )

    with _inventory_client(workspace_app, db, jobs=_RecordingJobs()) as client:
        first = client.get(
            "/api/v1/workspaces/ws-inventory/file-inventory/items",
            params={"prefix": "src/", "entry_kind": "file", "limit": 1},
        )
        second = client.get(
            "/api/v1/workspaces/ws-inventory/file-inventory/items",
            params={"prefix": "src/", "entry_kind": "file", "cursor": first.json()["next_cursor"], "limit": 5},
        )
        invalid = client.get(
            "/api/v1/workspaces/ws-inventory/file-inventory/items",
            params={"cursor": "not-base64"},
        )

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert invalid.status_code == 422
    assert [item["relative_path"] for item in first.json()["items"]] == ["src/a.py"]
    assert [item["relative_path"] for item in second.json()["items"]] == ["src/c.py"]
    assert all(item["entry_kind"] == "file" for item in first.json()["items"] + second.json()["items"])
    assert str(tmp_path) not in first.text
    assert str(tmp_path) not in second.text
