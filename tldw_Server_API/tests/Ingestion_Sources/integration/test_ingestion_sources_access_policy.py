from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


class _FakeTx:
    async def __aenter__(self):
        return object()

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakePool:
    def transaction(self):
        return _FakeTx()


@pytest.fixture()
def ingestion_sources_policy_client(monkeypatch):
    import tldw_Server_API.app.api.v1.endpoints.ingestion_sources as ep

    app = FastAPI()
    app.include_router(ep.router, prefix="/api/v1")
    current_user = SimpleNamespace(id=7, active_org_id=42, org_ids=[41, 42])
    app.dependency_overrides[ep.get_request_user] = lambda: current_user

    created_payloads: list[dict[str, Any]] = []
    updated_patches: list[dict[str, Any]] = []
    source_state: dict[int, dict[str, Any]] = {
        11: _source_row(
            id=11,
            source_type="archive_snapshot",
            config={"archive": "staged.zip"},
        ),
        12: _source_row(
            id=12,
            source_type="local_directory",
            config={"path": "/allowed/docs"},
        ),
    }

    async def _fake_get_db_pool():
        return _FakePool()

    async def _fake_ensure_schema(_db):
        return None

    async def _fake_create_source(_db, *, user_id, payload):
        created_payloads.append(dict(payload))
        return _source_row(id=21 + len(created_payloads), user_id=user_id, **payload)

    async def _fake_get_source_by_id(_db, *, source_id, user_id=None):
        row = source_state.get(source_id)
        if row is None:
            return None
        if user_id is not None and row.get("user_id") != user_id:
            return None
        return dict(row)

    async def _fake_update_source(_db, *, source_id, user_id, patch):
        updated_patches.append(dict(patch))
        row = dict(source_state[source_id])
        row.update(patch)
        source_state[source_id] = row
        return dict(row)

    monkeypatch.setattr(ep, "get_db_pool", _fake_get_db_pool)
    monkeypatch.setattr(ep, "ensure_ingestion_sources_schema", _fake_ensure_schema)
    monkeypatch.setattr(ep, "create_source", _fake_create_source)
    monkeypatch.setattr(ep, "get_source_by_id", _fake_get_source_by_id)
    monkeypatch.setattr(ep, "update_source", _fake_update_source)
    monkeypatch.setattr(ep, "validate_local_directory_source", lambda config: config["path"])
    monkeypatch.setattr(ep, "validate_git_repository_source", lambda config: dict(config))

    return {
        "client": TestClient(app),
        "endpoint_module": ep,
        "created_payloads": created_payloads,
        "updated_patches": updated_patches,
        "source_state": source_state,
    }


def _source_row(
    *,
    id: int,
    source_type: str,
    sink_type: str = "notes",
    policy: str = "canonical",
    enabled: bool = True,
    user_id: int = 7,
    config: dict[str, Any] | None = None,
    **overrides,
) -> dict[str, Any]:
    row = {
        "id": id,
        "user_id": user_id,
        "source_type": source_type,
        "sink_type": sink_type,
        "policy": policy,
        "enabled": enabled,
        "schedule_enabled": False,
        "schedule_config": {},
        "config": config or {},
        "active_job_id": None,
        "last_successful_snapshot_id": None,
        "last_sync_started_at": None,
        "last_sync_completed_at": None,
        "last_sync_status": None,
        "last_error": None,
        "last_successful_sync_summary": {},
        "created_at": None,
        "updated_at": None,
    }
    row.update(overrides)
    return row


def _create_payload(source_type: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "source_type": source_type,
        "sink_type": "notes",
        "policy": "canonical",
        "enabled": True,
        "config": config or {},
    }


@pytest.mark.integration
def test_create_local_directory_source_returns_403_without_entitlement(
    ingestion_sources_policy_client,
    monkeypatch,
):
    ep = ingestion_sources_policy_client["endpoint_module"]
    monkeypatch.setattr(
        ep,
        "can_create_local_directory_ingestion_source",
        lambda _current_user: False,
        raising=False,
    )

    response = ingestion_sources_policy_client["client"].post(
        "/api/v1/ingestion-sources/",
        json=_create_payload("local_directory", {"path": "/allowed/docs"}),
    )

    assert response.status_code == 403, response.text
    assert response.json()["detail"] == "Local directory ingestion sources are not enabled for this user"
    assert ingestion_sources_policy_client["created_payloads"] == []


@pytest.mark.integration
def test_create_local_directory_source_succeeds_with_entitlement(
    ingestion_sources_policy_client,
    monkeypatch,
):
    ep = ingestion_sources_policy_client["endpoint_module"]
    monkeypatch.setattr(
        ep,
        "can_create_local_directory_ingestion_source",
        lambda _current_user: True,
        raising=False,
    )

    response = ingestion_sources_policy_client["client"].post(
        "/api/v1/ingestion-sources/",
        json=_create_payload("local_directory", {"path": "/allowed/docs"}),
    )

    assert response.status_code == 201, response.text
    assert response.json()["source_type"] == "local_directory"
    assert ingestion_sources_policy_client["created_payloads"][0]["config"]["path"] == "/allowed/docs"


@pytest.mark.integration
@pytest.mark.parametrize(
    ("source_type", "config"),
    [
        ("archive_snapshot", {}),
        (
            "git_repository",
            {
                "mode": "remote_github_repo",
                "repo_url": "https://github.com/example/repo",
            },
        ),
    ],
)
def test_create_non_local_directory_sources_are_not_blocked_by_entitlement(
    ingestion_sources_policy_client,
    monkeypatch,
    source_type,
    config,
):
    ep = ingestion_sources_policy_client["endpoint_module"]
    monkeypatch.setattr(
        ep,
        "can_create_local_directory_ingestion_source",
        lambda _current_user: False,
        raising=False,
    )

    response = ingestion_sources_policy_client["client"].post(
        "/api/v1/ingestion-sources/",
        json=_create_payload(source_type, config),
    )

    assert response.status_code == 201, response.text
    assert response.json()["source_type"] == source_type


@pytest.mark.integration
def test_patch_cannot_retarget_source_to_local_directory_without_entitlement(
    ingestion_sources_policy_client,
    monkeypatch,
):
    ep = ingestion_sources_policy_client["endpoint_module"]
    monkeypatch.setattr(
        ep,
        "can_create_local_directory_ingestion_source",
        lambda _current_user: False,
        raising=False,
    )

    response = ingestion_sources_policy_client["client"].patch(
        "/api/v1/ingestion-sources/11",
        json={"source_type": "local_directory", "config": {"path": "/allowed/docs"}},
    )

    assert response.status_code == 403, response.text
    assert ingestion_sources_policy_client["updated_patches"] == []


@pytest.mark.integration
def test_patch_cannot_change_local_directory_config_without_entitlement(
    ingestion_sources_policy_client,
    monkeypatch,
):
    ep = ingestion_sources_policy_client["endpoint_module"]
    monkeypatch.setattr(
        ep,
        "can_create_local_directory_ingestion_source",
        lambda _current_user: False,
        raising=False,
    )

    response = ingestion_sources_policy_client["client"].patch(
        "/api/v1/ingestion-sources/12",
        json={"config": {"path": "/allowed/other-docs"}},
    )

    assert response.status_code == 403, response.text
    assert ingestion_sources_policy_client["updated_patches"] == []


@pytest.mark.integration
def test_patch_existing_local_directory_non_identity_settings_without_entitlement(
    ingestion_sources_policy_client,
    monkeypatch,
):
    ep = ingestion_sources_policy_client["endpoint_module"]
    monkeypatch.setattr(
        ep,
        "can_create_local_directory_ingestion_source",
        lambda _current_user: False,
        raising=False,
    )

    response = ingestion_sources_policy_client["client"].patch(
        "/api/v1/ingestion-sources/12",
        json={"enabled": False, "schedule_enabled": True},
    )

    assert response.status_code == 200, response.text
    assert response.json()["enabled"] is False
    assert response.json()["schedule_enabled"] is True
    assert ingestion_sources_policy_client["updated_patches"] == [
        {"enabled": False, "schedule_enabled": True}
    ]


@pytest.mark.integration
@pytest.mark.parametrize("allowed", [False, True])
def test_capabilities_reports_current_user_local_directory_entitlement(
    ingestion_sources_policy_client,
    monkeypatch,
    allowed,
):
    ep = ingestion_sources_policy_client["endpoint_module"]
    monkeypatch.setattr(
        ep,
        "can_create_local_directory_ingestion_source",
        lambda _current_user: allowed,
        raising=False,
    )

    response = ingestion_sources_policy_client["client"].get(
        "/api/v1/ingestion-sources/capabilities",
    )

    assert response.status_code == 200, response.text
    assert response.json() == {"can_create_local_directory": allowed}
