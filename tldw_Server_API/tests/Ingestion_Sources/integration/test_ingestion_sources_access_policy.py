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
    import tldw_Server_API.app.core.Ingestion_Sources.access_policy as access_policy

    app = FastAPI()

    @app.middleware("http")
    async def _inject_scope(request, call_next):
        request.state.org_ids = [41, 42]
        request.state.active_org_id = 42
        return await call_next(request)

    app.include_router(ep.router, prefix="/api/v1")
    current_user = SimpleNamespace(id=7)
    app.dependency_overrides[ep.get_request_user] = lambda: current_user

    feature_flags: list[dict[str, Any]] = []
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
    monkeypatch.setattr(access_policy, "is_single_user_mode", lambda: False)
    monkeypatch.setattr(access_policy, "list_feature_flags", lambda: list(feature_flags))

    return {
        "client": TestClient(app),
        "app": app,
        "endpoint_module": ep,
        "feature_flags": feature_flags,
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


def _flag(**overrides) -> dict[str, Any]:
    payload = {
        "key": "ingestion_sources.local_directory",
        "scope": "global",
        "enabled": True,
        "org_id": None,
        "user_id": None,
        "target_user_ids": [],
        "rollout_percent": 100,
    }
    payload.update(overrides)
    return payload


@pytest.mark.integration
def test_create_local_directory_source_returns_403_without_entitlement(
    ingestion_sources_policy_client,
):
    response = ingestion_sources_policy_client["client"].post(
        "/api/v1/ingestion-sources/",
        json=_create_payload("local_directory", {"path": "/allowed/docs"}),
    )

    assert response.status_code == 403, response.text
    assert response.json()["detail"] == "Local directory ingestion sources are not enabled for this user"
    assert ingestion_sources_policy_client["created_payloads"] == []


@pytest.mark.integration
def test_denied_local_directory_create_does_not_validate_path(
    ingestion_sources_policy_client,
    monkeypatch,
):
    ep = ingestion_sources_policy_client["endpoint_module"]
    validated_paths: list[dict[str, Any]] = []

    def _spy_validate(config):
        validated_paths.append(dict(config))
        return config["path"]

    monkeypatch.setattr(ep, "validate_local_directory_source", _spy_validate)

    response = ingestion_sources_policy_client["client"].post(
        "/api/v1/ingestion-sources/",
        json=_create_payload("local_directory", {"path": "/outside/secret"}),
    )

    assert response.status_code == 403, response.text
    assert validated_paths == []
    assert ingestion_sources_policy_client["created_payloads"] == []


@pytest.mark.integration
def test_create_local_directory_source_succeeds_with_entitlement(
    ingestion_sources_policy_client,
):
    ingestion_sources_policy_client["feature_flags"].append(_flag(scope="user", user_id=7))

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
    source_type,
    config,
):
    response = ingestion_sources_policy_client["client"].post(
        "/api/v1/ingestion-sources/",
        json=_create_payload(source_type, config),
    )

    assert response.status_code == 201, response.text
    assert response.json()["source_type"] == source_type


@pytest.mark.integration
def test_patch_cannot_retarget_source_to_local_directory_without_entitlement(
    ingestion_sources_policy_client,
):
    response = ingestion_sources_policy_client["client"].patch(
        "/api/v1/ingestion-sources/11",
        json={"source_type": "local_directory", "config": {"path": "/allowed/docs"}},
    )

    assert response.status_code == 403, response.text
    assert ingestion_sources_policy_client["updated_patches"] == []


@pytest.mark.integration
def test_patch_cannot_change_local_directory_config_without_entitlement(
    ingestion_sources_policy_client,
):
    response = ingestion_sources_policy_client["client"].patch(
        "/api/v1/ingestion-sources/12",
        json={"config": {"path": "/allowed/other-docs"}},
    )

    assert response.status_code == 403, response.text
    assert ingestion_sources_policy_client["updated_patches"] == []


@pytest.mark.integration
def test_denied_local_directory_patch_does_not_validate_changed_path(
    ingestion_sources_policy_client,
    monkeypatch,
):
    ep = ingestion_sources_policy_client["endpoint_module"]
    validated_paths: list[dict[str, Any]] = []

    def _spy_validate(config):
        validated_paths.append(dict(config))
        return config["path"]

    monkeypatch.setattr(ep, "validate_local_directory_source", _spy_validate)

    response = ingestion_sources_policy_client["client"].patch(
        "/api/v1/ingestion-sources/12",
        json={"config": {"path": "/outside/secret"}},
    )

    assert response.status_code == 403, response.text
    assert validated_paths == []
    assert ingestion_sources_policy_client["updated_patches"] == []


@pytest.mark.integration
def test_patch_existing_local_directory_normalized_equivalent_config_without_entitlement(
    ingestion_sources_policy_client,
    monkeypatch,
):
    ep = ingestion_sources_policy_client["endpoint_module"]
    validated_paths: list[dict[str, Any]] = []

    def _normalize_trailing_slash(config):
        validated_paths.append(dict(config))
        return str(config["path"]).rstrip("/")

    monkeypatch.setattr(ep, "validate_local_directory_source", _normalize_trailing_slash)

    response = ingestion_sources_policy_client["client"].patch(
        "/api/v1/ingestion-sources/12",
        json={"config": {"path": "/allowed/docs/"}},
    )

    assert response.status_code == 200, response.text
    assert validated_paths == [{"path": "/allowed/docs/"}]
    assert response.json()["config"] == {"path": "/allowed/docs"}
    assert ingestion_sources_policy_client["updated_patches"] == [
        {"config": {"path": "/allowed/docs"}}
    ]


@pytest.mark.integration
def test_patch_null_source_type_still_checks_changed_local_directory_config_without_entitlement(
    ingestion_sources_policy_client,
):
    response = ingestion_sources_policy_client["client"].patch(
        "/api/v1/ingestion-sources/12",
        json={"source_type": None, "config": {"path": "/allowed/other-docs"}},
    )

    assert response.status_code == 403, response.text
    assert ingestion_sources_policy_client["updated_patches"] == []


@pytest.mark.integration
def test_patch_cannot_change_local_directory_sink_type_without_entitlement(
    ingestion_sources_policy_client,
):
    response = ingestion_sources_policy_client["client"].patch(
        "/api/v1/ingestion-sources/12",
        json={"sink_type": "media"},
    )

    assert response.status_code == 403, response.text
    assert ingestion_sources_policy_client["updated_patches"] == []


@pytest.mark.integration
def test_patch_existing_local_directory_non_identity_settings_without_entitlement(
    ingestion_sources_policy_client,
):
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
def test_patch_existing_local_directory_same_source_type_without_entitlement(
    ingestion_sources_policy_client,
):
    response = ingestion_sources_policy_client["client"].patch(
        "/api/v1/ingestion-sources/12",
        json={"source_type": "local_directory"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["source_type"] == "local_directory"
    assert ingestion_sources_policy_client["updated_patches"] == [
        {"source_type": "local_directory", "config": {"path": "/allowed/docs"}}
    ]


@pytest.mark.integration
def test_patch_existing_local_directory_same_normalized_config_without_entitlement(
    ingestion_sources_policy_client,
):
    response = ingestion_sources_policy_client["client"].patch(
        "/api/v1/ingestion-sources/12",
        json={"config": {"path": "/allowed/docs"}},
    )

    assert response.status_code == 200, response.text
    assert response.json()["config"] == {"path": "/allowed/docs"}
    assert ingestion_sources_policy_client["updated_patches"] == [
        {"config": {"path": "/allowed/docs"}}
    ]


@pytest.mark.integration
@pytest.mark.parametrize(
    "feature_flags",
    [
        [_flag(scope="global")],
        [_flag(scope="user", user_id=7)],
        [_flag(scope="org", org_id=42)],
    ],
)
def test_capabilities_reports_applicable_user_org_and_global_flags(
    ingestion_sources_policy_client,
    feature_flags,
):
    ingestion_sources_policy_client["feature_flags"].extend(feature_flags)

    response = ingestion_sources_policy_client["client"].get("/api/v1/ingestion-sources/capabilities")

    assert response.status_code == 200, response.text
    assert response.json() == {"can_create_local_directory": True}


@pytest.mark.integration
def test_capabilities_reports_false_without_applicable_flag(
    ingestion_sources_policy_client,
):
    response = ingestion_sources_policy_client["client"].get("/api/v1/ingestion-sources/capabilities")

    assert response.status_code == 200, response.text
    assert response.json() == {"can_create_local_directory": False}


@pytest.mark.integration
def test_capabilities_endpoint_uses_explicit_response_model(ingestion_sources_policy_client):
    from tldw_Server_API.app.api.v1.schemas.ingestion_sources import (
        IngestionSourceCapabilitiesResponse,
    )

    route = next(
        route
        for route in ingestion_sources_policy_client["app"].routes
        if getattr(route, "path", None) == "/api/v1/ingestion-sources/capabilities"
    )

    assert route.response_model is IngestionSourceCapabilitiesResponse
