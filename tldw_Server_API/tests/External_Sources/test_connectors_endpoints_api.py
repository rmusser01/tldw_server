import os
from typing import Tuple

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.External_Sources.sync_adapter import FileSyncWebhookSubscription


@pytest.fixture()
def connectors_client() -> Tuple[TestClient, dict]:
    """Provide a TestClient with connectors routes enabled and X-API-KEY headers.

    We enable the experimental 'connectors' route via ROUTES_ENABLE before importing the app.
    """
    # Ensure single-user auth and enable connectors router before importing the app
    # so the route gating logic sees the toggles during app construction.
    os.environ.setdefault("TEST_MODE", "true")
    os.environ.setdefault("ROUTES_STABLE_ONLY", "false")
    # Explicitly allow connectors (experimental) in case stable_only is true
    os.environ["ROUTES_ENABLE"] = "connectors"
    os.environ.setdefault("AUTH_MODE", "single_user")
    os.environ.setdefault("TESTING", "true")

    from tldw_Server_API.app.core.AuthNZ.settings import get_settings
    from tldw_Server_API.app.main import app
    from tldw_Server_API.app.api.v1.endpoints import connectors as connectors_router

    # Ensure connectors router is mounted even if a prior import cached the app without it.
    paths = {route.path for route in app.routes}
    if "/api/v1/connectors/sources" not in paths:
        app.include_router(connectors_router.router, prefix="/api/v1", tags=["connectors"])

    api_key = get_settings().SINGLE_USER_API_KEY
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    client = TestClient(app)
    return client, headers


@pytest.mark.integration
def test_list_providers_hides_gmail_when_feature_flag_disabled(connectors_client, monkeypatch):
    client, headers = connectors_client
    import tldw_Server_API.app.api.v1.endpoints.connectors as ep

    monkeypatch.setitem(ep.settings, "EMAIL_GMAIL_CONNECTOR_ENABLED", False)
    response = client.get("/api/v1/connectors/providers", headers=headers)
    assert response.status_code == 200, response.text
    names = {str(item.get("name")) for item in response.json()}
    assert "drive" in names
    assert "notion" in names
    assert "gmail" not in names


@pytest.mark.integration
def test_list_providers_includes_gmail_when_feature_flag_enabled(connectors_client, monkeypatch):
    client, headers = connectors_client
    import tldw_Server_API.app.api.v1.endpoints.connectors as ep

    monkeypatch.setitem(ep.settings, "EMAIL_GMAIL_CONNECTOR_ENABLED", True)
    response = client.get("/api/v1/connectors/providers", headers=headers)
    assert response.status_code == 200, response.text
    names = {str(item.get("name")) for item in response.json()}
    assert "gmail" in names


@pytest.mark.integration
def test_list_providers_includes_zotero(connectors_client):
    client, headers = connectors_client

    response = client.get("/api/v1/connectors/providers", headers=headers)
    assert response.status_code == 200, response.text
    providers = {str(item.get("name")): item for item in response.json()}
    assert "zotero" in providers
    assert providers["zotero"]["auth_type"] == "oauth1"


@pytest.mark.integration
def test_add_source_blocks_gmail_when_feature_flag_disabled(connectors_client, monkeypatch):
    client, headers = connectors_client
    import tldw_Server_API.app.api.v1.endpoints.connectors as ep

    monkeypatch.setitem(ep.settings, "EMAIL_GMAIL_CONNECTOR_ENABLED", False)
    payload = {
        "account_id": 1,
        "provider": "gmail",
        "remote_id": "INBOX",
        "type": "folder",
    }
    response = client.post("/api/v1/connectors/sources", json=payload, headers=headers)
    assert response.status_code == 404


@pytest.mark.integration
def test_add_source_success(connectors_client, monkeypatch):
    client, headers = connectors_client

    # Patch the service call used by the endpoint to avoid DB work
    async def _fake_create_source(db, *, account_id, provider, remote_id, type_, path, options, enabled=True):
        return {
            "id": 101,
            "account_id": account_id,
            "provider": provider,
            "remote_id": remote_id,
            "type": type_,
            "path": path,
            "options": options or {},
            "enabled": enabled,
            "last_synced_at": None,
        }

    import tldw_Server_API.app.api.v1.endpoints.connectors as ep
    monkeypatch.setattr(ep, "create_source", _fake_create_source)
    async def _fake_get_account_for_user(db, user_id, account_id):
        return {"id": account_id, "user_id": user_id, "provider": "drive"}
    monkeypatch.setattr(ep, "get_account_for_user", _fake_get_account_for_user)

    payload = {
        "account_id": 1,
        "provider": "drive",
        "remote_id": "root",
        "type": "folder",
        "path": "/",
        "options": {"recursive": True},
    }
    r = client.post("/api/v1/connectors/sources", json=payload, headers=headers)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["id"] == 101
    assert body["provider"] == "drive"
    assert body["options"]["recursive"] is True


@pytest.mark.integration
def test_add_onedrive_source_provisions_webhook_subscription(connectors_client, monkeypatch):
    client, headers = connectors_client

    import tldw_Server_API.app.api.v1.endpoints.connectors as ep

    subscribe_calls: list[dict] = []
    sync_state_updates: list[dict] = []

    async def _fake_create_source(db, *, account_id, provider, remote_id, type_, path, options, enabled=True):
        return {
            "id": 501,
            "account_id": account_id,
            "provider": provider,
            "remote_id": remote_id,
            "type": type_,
            "path": path,
            "options": options or {},
            "enabled": enabled,
            "last_synced_at": None,
        }

    async def _fake_get_account_for_user(db, user_id, account_id):
        return {"id": account_id, "user_id": user_id, "provider": "onedrive"}

    async def _fake_get_account_tokens(db, user_id, account_id):
        return {"access_token": "tok"}

    async def _fake_upsert_source_sync_state(db, *, source_id, **updates):
        payload = {"source_id": source_id, **updates}
        sync_state_updates.append(payload)
        return payload

    class _FakeConn:
        name = "onedrive"

        def authorize_url(self, *a, **kw):
            return ""

        async def exchange_code(self, *a, **kw):
            return {}

        async def subscribe_webhook(self, account, *, resource, callback_url):
            subscribe_calls.append(
                {
                    "account": dict(account),
                    "resource": dict(resource),
                    "callback_url": callback_url,
                }
            )
            return FileSyncWebhookSubscription(
                subscription_id="sub-501",
                expires_at="2026-03-10T00:00:00Z",
                metadata={"expirationDateTime": "2026-03-10T00:00:00Z"},
            )

    monkeypatch.setattr(ep, "create_source", _fake_create_source)
    monkeypatch.setattr(ep, "get_account_for_user", _fake_get_account_for_user)
    monkeypatch.setattr(ep, "get_account_tokens", _fake_get_account_tokens)
    monkeypatch.setattr(ep, "get_connector_by_name", lambda provider: _FakeConn())
    monkeypatch.setattr(ep, "evaluate_policy_constraints", lambda *a, **kw: (True, None))
    monkeypatch.setattr(ep, "upsert_source_sync_state", _fake_upsert_source_sync_state, raising=False)

    payload = {
        "account_id": 12,
        "provider": "onedrive",
        "remote_id": "root",
        "type": "folder",
        "path": "/",
        "options": {"recursive": True},
    }
    r = client.post("/api/v1/connectors/sources", json=payload, headers=headers)

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["id"] == 501
    assert body["provider"] == "onedrive"
    assert len(subscribe_calls) == 1
    assert subscribe_calls[0]["callback_url"] == "http://testserver/api/v1/connectors/providers/onedrive/webhook"
    assert subscribe_calls[0]["resource"]["resource"] == "me/drive/root"
    assert subscribe_calls[0]["account"]["tokens"]["access_token"] == "tok"
    assert sync_state_updates[0]["source_id"] == 501
    assert sync_state_updates[0]["sync_mode"] == "hybrid"
    assert sync_state_updates[0]["webhook_status"] == "active"
    assert sync_state_updates[0]["webhook_subscription_id"] == "sub-501"
    assert sync_state_updates[0]["webhook_expires_at"] == "2026-03-10T00:00:00Z"


@pytest.mark.integration
def test_add_drive_source_provisions_webhook_subscription_and_cursor(connectors_client, monkeypatch):
    client, headers = connectors_client

    import tldw_Server_API.app.api.v1.endpoints.connectors as ep

    subscribe_calls: list[dict] = []
    sync_state_updates: list[dict] = []

    async def _fake_create_source(db, *, account_id, provider, remote_id, type_, path, options, enabled=True):
        return {
            "id": 601,
            "account_id": account_id,
            "provider": provider,
            "remote_id": remote_id,
            "type": type_,
            "path": path,
            "options": options or {},
            "enabled": enabled,
            "last_synced_at": None,
        }

    async def _fake_get_account_for_user(db, user_id, account_id):
        return {"id": account_id, "user_id": user_id, "provider": "drive"}

    async def _fake_get_account_tokens(db, user_id, account_id):
        return {"access_token": "tok"}

    async def _fake_upsert_source_sync_state(db, *, source_id, **updates):
        payload = {"source_id": source_id, **updates}
        sync_state_updates.append(payload)
        return payload

    class _FakeConn:
        name = "drive"

        def authorize_url(self, *a, **kw):
            return ""

        async def exchange_code(self, *a, **kw):
            return {}

        async def subscribe_webhook(self, account, *, resource, callback_url):
            subscribe_calls.append(
                {
                    "account": dict(account),
                    "resource": dict(resource),
                    "callback_url": callback_url,
                }
            )
            return FileSyncWebhookSubscription(
                subscription_id="drive-chan-1",
                expires_at="2026-03-10T00:00:00Z",
                metadata={
                    "resourceId": "drive-resource-1",
                    "pageToken": "drive-cursor-1",
                    "callback_url": callback_url,
                    "clientState": "drive-state-1",
                },
            )

    monkeypatch.setattr(ep, "create_source", _fake_create_source)
    monkeypatch.setattr(ep, "get_account_for_user", _fake_get_account_for_user)
    monkeypatch.setattr(ep, "get_account_tokens", _fake_get_account_tokens)
    monkeypatch.setattr(ep, "get_connector_by_name", lambda provider: _FakeConn())
    monkeypatch.setattr(ep, "evaluate_policy_constraints", lambda *a, **kw: (True, None))
    monkeypatch.setattr(ep, "upsert_source_sync_state", _fake_upsert_source_sync_state, raising=False)

    payload = {
        "account_id": 13,
        "provider": "drive",
        "remote_id": "root",
        "type": "folder",
        "path": "/",
        "options": {"recursive": True},
    }
    r = client.post("/api/v1/connectors/sources", json=payload, headers=headers)

    assert r.status_code == 200, r.text
    assert len(subscribe_calls) == 1
    assert subscribe_calls[0]["callback_url"] == "http://testserver/api/v1/connectors/providers/drive/webhook"
    assert sync_state_updates[0]["source_id"] == 601
    assert sync_state_updates[0]["sync_mode"] == "hybrid"
    assert sync_state_updates[0]["cursor"] == "drive-cursor-1"
    assert sync_state_updates[0]["cursor_kind"] == "drive_start_page_token"
    assert sync_state_updates[0]["webhook_status"] == "active"
    assert sync_state_updates[0]["webhook_subscription_id"] == "drive-chan-1"
    assert sync_state_updates[0]["webhook_expires_at"] == "2026-03-10T00:00:00Z"
    assert sync_state_updates[0]["webhook_metadata"]["resourceId"] == "drive-resource-1"


@pytest.mark.integration
def test_add_source_rejects_missing_account(connectors_client, monkeypatch):
    client, headers = connectors_client

    async def _fake_get_account_for_user(db, user_id, account_id):
        return None

    import tldw_Server_API.app.api.v1.endpoints.connectors as ep
    monkeypatch.setattr(ep, "get_account_for_user", _fake_get_account_for_user)

    payload = {
        "account_id": 999,
        "provider": "drive",
        "remote_id": "root",
        "type": "folder",
    }
    r = client.post("/api/v1/connectors/sources", json=payload, headers=headers)
    assert r.status_code == 404


@pytest.mark.integration
def test_add_source_rejects_provider_mismatch(connectors_client, monkeypatch):
    client, headers = connectors_client

    async def _fake_get_account_for_user(db, user_id, account_id):
        return {"id": account_id, "user_id": user_id, "provider": "notion"}

    import tldw_Server_API.app.api.v1.endpoints.connectors as ep
    monkeypatch.setattr(ep, "get_account_for_user", _fake_get_account_for_user)

    payload = {
        "account_id": 1,
        "provider": "drive",
        "remote_id": "root",
        "type": "folder",
    }
    r = client.post("/api/v1/connectors/sources", json=payload, headers=headers)
    assert r.status_code == 400


@pytest.mark.integration
def test_add_source_forbid_extra_fields(connectors_client, monkeypatch):
    client, headers = connectors_client

    # No patch needed; we expect validation to fail before hitting service
    payload = {
        "account_id": 1,
        "provider": "drive",
        "remote_id": "root",
        "type": "folder",
        "unexpected": "nope",  # extra field should be rejected by extra='forbid'
    }
    r = client.post("/api/v1/connectors/sources", json=payload, headers=headers)
    assert r.status_code == 422


@pytest.mark.integration
def test_patch_source_success(connectors_client, monkeypatch):
    client, headers = connectors_client

    async def _fake_get_source_by_id(db, user_id, source_id):
        return {
            "id": source_id,
            "account_id": 1,
            "provider": "notion",
            "remote_id": "abc",
            "type": "page",
            "path": None,
            "options": {"recursive": False},
            "enabled": True,
            "last_synced_at": None,
        }

    async def _fake_update_source(db, user_id, source_id, *, enabled=None, options=None):
        return {
            "id": source_id,
            "account_id": 1,
            "provider": "notion",
            "remote_id": "abc",
            "type": "page",
            "path": None,
            "options": options or {"recursive": False},
            "enabled": bool(enabled) if enabled is not None else True,
            "last_synced_at": None,
        }

    import tldw_Server_API.app.api.v1.endpoints.connectors as ep
    monkeypatch.setattr(ep, "get_source_by_id", _fake_get_source_by_id)
    monkeypatch.setattr(ep, "update_source", _fake_update_source)

    r = client.patch(
        "/api/v1/connectors/sources/55",
        json={"enabled": False, "options": {"recursive": False}},
        headers=headers,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["id"] == 55
    assert body["enabled"] is False
    assert body["options"]["recursive"] is False


@pytest.mark.integration
def test_patch_source_404(connectors_client, monkeypatch):
    client, headers = connectors_client

    async def _fake_update_source_none(db, user_id, source_id, *, enabled=None, options=None):
        return None

    import tldw_Server_API.app.api.v1.endpoints.connectors as ep
    monkeypatch.setattr(ep, "update_source", _fake_update_source_none)

    r = client.patch(
        "/api/v1/connectors/sources/9999",
        json={"enabled": True},
        headers=headers,
    )
    assert r.status_code == 404


@pytest.mark.integration
def test_patch_source_forbid_extra_fields(connectors_client):
    client, headers = connectors_client
    r = client.patch(
        "/api/v1/connectors/sources/55",
        json={"enabled": True, "extra": "bad"},
        headers=headers,
    )
    assert r.status_code == 422


@pytest.mark.integration
def test_get_sources_includes_sync_summary(connectors_client, monkeypatch):
    client, headers = connectors_client

    import tldw_Server_API.app.api.v1.endpoints.connectors as ep
    import tldw_Server_API.app.core.Jobs.manager as jobs_manager

    async def _fake_list_sources(db, user_id):
        return [
            {
                "id": 22,
                "account_id": 5,
                "provider": "onedrive",
                "remote_id": "root",
                "type": "folder",
                "path": "/",
                "options": {"recursive": True},
                "enabled": True,
                "last_synced_at": None,
            }
        ]

    async def _fake_get_source_sync_state(db, *, source_id):
        assert source_id == 22
        return {
            "source_id": source_id,
            "sync_mode": "hybrid",
            "last_sync_failed_at": "2026-03-06 10:02:00",
            "last_error": "delta cursor invalid",
            "webhook_status": "active",
            "needs_full_rescan": True,
            "active_job_id": "88",
        }

    async def _fake_get_source_binding_health(db, *, source_id):
        assert source_id == 22
        return {
            "tracked_item_count": 4,
            "degraded_item_count": 2,
            "duplicate_count": 3,
            "metadata_only_count": 1,
        }

    class _FakeJobManager:
        def get_job(self, job_id: int):
            assert job_id == 88
            return None

    monkeypatch.setattr(ep, "list_sources", _fake_list_sources)
    monkeypatch.setattr(ep, "get_source_sync_state", _fake_get_source_sync_state)
    monkeypatch.setattr(ep, "get_source_binding_health", _fake_get_source_binding_health)
    monkeypatch.setattr(jobs_manager, "JobManager", _FakeJobManager)

    response = client.get("/api/v1/connectors/sources", headers=headers)
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body) == 1
    assert body[0]["id"] == 22
    assert body[0]["sync"]["state"] == "needs_full_rescan"
    assert body[0]["sync"]["sync_mode"] == "hybrid"
    assert body[0]["sync"]["last_error"] == "delta cursor invalid"
    assert body[0]["sync"]["webhook_status"] == "active"
    assert body[0]["sync"]["needs_full_rescan"] is True
    assert body[0]["sync"]["active_job_id"] == "88"
    assert body[0]["sync"]["tracked_item_count"] == 4
    assert body[0]["sync"]["degraded_item_count"] == 2
    assert body[0]["sync"]["duplicate_count"] == 3
    assert body[0]["sync"]["metadata_only_count"] == 1


@pytest.mark.integration
def test_oauth_callback_rejects_invalid_state(connectors_client, monkeypatch):
    client, headers = connectors_client

    import tldw_Server_API.app.api.v1.endpoints.connectors as ep

    async def _fake_consume_oauth_state(db, *, user_id, provider, state, max_age_minutes=10):
        return False

    class _FakeConn:
        name = "notion"
        def authorize_url(self, *a, **kw):
            return ""
        async def exchange_code(self, *a, **kw):
            raise AssertionError("exchange_code should not be called on invalid state")

    monkeypatch.setattr(ep, "consume_oauth_state", _fake_consume_oauth_state)
    monkeypatch.setattr(ep, "get_connector_by_name", lambda provider: _FakeConn())

    r = client.get(
        "/api/v1/connectors/providers/notion/callback",
        params={"code": "abc", "state": "bad"},
        headers=headers,
    )
    assert r.status_code == 403


@pytest.mark.integration
def test_oauth_callback_accepts_valid_state(connectors_client, monkeypatch):
    client, headers = connectors_client

    import tldw_Server_API.app.api.v1.endpoints.connectors as ep

    async def _fake_consume_oauth_state(db, *, user_id, provider, state, max_age_minutes=10):
        return True

    class _FakeConn:
        name = "notion"
        def authorize_url(self, *a, **kw):
            return ""
        async def exchange_code(self, code, redirect_uri):
            return {
                "access_token": "tok",
                "refresh_token": "rtok",
                "provider": "notion",
                "display_name": "Notion Account",
                "workspace_id": "ws1",
                "workspace_name": "Workspace",
            }

    async def _fake_create_account(db, user_id, provider, display_name, email, tokens):
        return {"id": 123, "display_name": display_name, "email": email, "created_at": "now"}

    monkeypatch.setattr(ep, "consume_oauth_state", _fake_consume_oauth_state)
    monkeypatch.setattr(ep, "get_connector_by_name", lambda provider: _FakeConn())
    monkeypatch.setattr(ep, "create_account", _fake_create_account)
    monkeypatch.setenv("ORG_CONNECTORS_ACCOUNT_LINKING_ROLE", "member")

    r = client.get(
        "/api/v1/connectors/providers/notion/callback",
        params={"code": "abc", "state": "good"},
        headers=headers,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["id"] == 123
    assert body["provider"] == "notion"


@pytest.mark.integration
def test_start_authorize_zotero_requests_temporary_credential_and_persists_state_metadata(
    connectors_client,
    monkeypatch,
):
    client, headers = connectors_client

    import tldw_Server_API.app.api.v1.endpoints.connectors as ep

    oauth_state_calls: list[dict] = []
    authorize_calls: list[dict] = []

    async def _fake_create_oauth_state(db, user_id, provider, state, metadata=None):
        oauth_state_calls.append(
            {
                "user_id": user_id,
                "provider": provider,
                "state": state,
                "metadata": dict(metadata or {}),
            }
        )

    class _FakeConn:
        name = "zotero"
        redirect_base = "http://ignored.example"

        async def request_temporary_credential(self, callback_url):
            authorize_calls.append({"callback_url": callback_url})
            return {
                "oauth_token": "temp-token",
                "oauth_token_secret": "temp-secret",
            }

        def authorize_url(self, state=None, scopes=None, redirect_path=None):
            authorize_calls.append(
                {
                    "state": state,
                    "scopes": list(scopes or []),
                    "redirect_path": redirect_path,
                }
            )
            return "https://www.zotero.org/oauth/authorize?oauth_token=temp-token"

    monkeypatch.setattr(ep, "create_oauth_state", _fake_create_oauth_state)
    monkeypatch.setattr(ep, "get_connector_by_name", lambda provider: _FakeConn())

    response = client.post("/api/v1/connectors/providers/zotero/authorize", headers=headers)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["auth_url"] == "https://www.zotero.org/oauth/authorize?oauth_token=temp-token"
    assert oauth_state_calls[0]["provider"] == "zotero"
    assert oauth_state_calls[0]["metadata"] == {
        "oauth_token": "temp-token",
        "oauth_token_secret": "temp-secret",
    }
    assert authorize_calls[0]["callback_url"].startswith(
        "http://testserver/api/v1/connectors/providers/zotero/callback?state="
    )
    assert authorize_calls[1]["scopes"] == ["oauth_token=temp-token"]


@pytest.mark.integration
def test_start_authorize_zotero_ignores_untrusted_request_base_when_connector_base_is_configured(
    connectors_client,
    monkeypatch,
):
    base_client, headers = connectors_client

    import tldw_Server_API.app.api.v1.endpoints.connectors as ep

    oauth_state_calls: list[dict] = []
    authorize_calls: list[dict] = []

    async def _fake_create_oauth_state(db, user_id, provider, state, metadata=None):
        oauth_state_calls.append(
            {
                "user_id": user_id,
                "provider": provider,
                "state": state,
                "metadata": dict(metadata or {}),
            }
        )

    class _FakeConn:
        name = "zotero"
        redirect_base = "https://trusted.example"

        async def request_temporary_credential(self, callback_url):
            authorize_calls.append({"callback_url": callback_url})
            return {
                "oauth_token": "temp-token",
                "oauth_token_secret": "temp-secret",
            }

        def authorize_url(self, state=None, scopes=None, redirect_path=None):
            authorize_calls.append(
                {
                    "state": state,
                    "scopes": list(scopes or []),
                    "redirect_path": redirect_path,
                    "redirect_base": self.redirect_base,
                }
            )
            return "https://www.zotero.org/oauth/authorize?oauth_token=temp-token"

    client = TestClient(base_client.app, base_url="https://evil.example")
    monkeypatch.setattr(ep, "create_oauth_state", _fake_create_oauth_state)
    monkeypatch.setattr(ep, "get_connector_by_name", lambda provider: _FakeConn())

    response = client.post("/api/v1/connectors/providers/zotero/authorize", headers=headers)
    assert response.status_code == 200, response.text
    assert oauth_state_calls[0]["provider"] == "zotero"
    assert authorize_calls[0]["callback_url"].startswith(
        "https://trusted.example/api/v1/connectors/providers/zotero/callback?state="
    )
    assert authorize_calls[1]["redirect_base"] == "https://trusted.example"


@pytest.mark.integration
def test_start_authorize_zotero_rejects_request_derived_localhost_base_in_production_mode(
    connectors_client,
    monkeypatch,
):
    base_client, headers = connectors_client

    import tldw_Server_API.app.api.v1.endpoints.connectors as ep

    class _FakeConn:
        name = "zotero"
        redirect_base = "http://ignored.example"

        async def request_temporary_credential(self, callback_url):
            raise AssertionError("temporary credentials must not be requested when redirect base is unconfigured")

        def authorize_url(self, *a, **kw):
            raise AssertionError("authorize_url must not be reached when redirect base is unconfigured")

    monkeypatch.delenv("CONNECTOR_REDIRECT_BASE_URL", raising=False)
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.delenv("TESTING", raising=False)
    monkeypatch.setattr(ep, "get_connector_by_name", lambda provider: _FakeConn())

    client = TestClient(base_client.app, base_url="http://localhost")
    response = client.post("/api/v1/connectors/providers/zotero/authorize", headers=headers)

    assert response.status_code == 500
    assert response.json()["detail"] == "CONNECTOR_REDIRECT_BASE_URL must be configured before enabling OAuth connectors"


@pytest.mark.integration
def test_start_authorize_zotero_rejects_insecure_non_https_redirect_base(
    connectors_client,
    monkeypatch,
):
    client, headers = connectors_client

    import tldw_Server_API.app.api.v1.endpoints.connectors as ep

    class _FakeConn:
        name = "zotero"
        redirect_base = "http://ignored.example"

        async def request_temporary_credential(self, callback_url):
            raise AssertionError("temporary credentials must not be requested for insecure redirect bases")

        def authorize_url(self, *a, **kw):
            raise AssertionError("authorize_url must not be reached for insecure redirect bases")

    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.delenv("TESTING", raising=False)
    monkeypatch.setenv("CONNECTOR_REDIRECT_BASE_URL", "http://example.com")
    monkeypatch.setattr(ep, "get_connector_by_name", lambda provider: _FakeConn())

    response = client.post("/api/v1/connectors/providers/zotero/authorize", headers=headers)

    assert response.status_code == 500
    assert response.json()["detail"] == "CONNECTOR_REDIRECT_BASE_URL must use https unless it targets localhost"


@pytest.mark.integration
def test_start_authorize_zotero_treats_whitespace_redirect_base_as_unconfigured(
    connectors_client,
    monkeypatch,
):
    base_client, headers = connectors_client

    import tldw_Server_API.app.api.v1.endpoints.connectors as ep

    class _FakeConn:
        name = "zotero"
        redirect_base = "http://ignored.example"

        async def request_temporary_credential(self, callback_url):
            raise AssertionError("temporary credentials must not be requested when redirect base is blank")

        def authorize_url(self, *a, **kw):
            raise AssertionError("authorize_url must not be reached when redirect base is blank")

    monkeypatch.setenv("CONNECTOR_REDIRECT_BASE_URL", "   ")
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.delenv("TESTING", raising=False)
    monkeypatch.setattr(ep, "get_connector_by_name", lambda provider: _FakeConn())

    client = TestClient(base_client.app, base_url="http://localhost")
    response = client.post("/api/v1/connectors/providers/zotero/authorize", headers=headers)

    assert response.status_code == 500
    assert response.json()["detail"] == "CONNECTOR_REDIRECT_BASE_URL must be configured before enabling OAuth connectors"


def test_resolve_webhook_callback_base_treats_whitespace_redirect_base_as_missing(monkeypatch):
    import tldw_Server_API.app.api.v1.endpoints.connectors as ep

    monkeypatch.setenv("CONNECTOR_REDIRECT_BASE_URL", "   ")
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.delenv("TESTING", raising=False)

    with pytest.raises(HTTPException) as exc:
        ep._resolve_webhook_callback_base(None, type("Conn", (), {"redirect_base": ""})())

    assert exc.value.status_code == 500
    assert exc.value.detail == "CONNECTOR_REDIRECT_BASE_URL must be configured before enabling webhook subscriptions"


@pytest.mark.integration
def test_oauth_callback_accepts_zotero_oauth1_payload_and_persists_provider_identity(
    connectors_client,
    monkeypatch,
):
    client, headers = connectors_client

    import tldw_Server_API.app.api.v1.endpoints.connectors as ep

    exchange_calls: list[dict] = []
    account_calls: list[dict] = []

    async def _fake_consume_oauth_state(db, *, user_id, provider, state, max_age_minutes=10):
        return {
            "state": state,
            "provider": provider,
            "oauth_token": "temp-token",
            "oauth_token_secret": "temp-secret",
        }

    class _FakeConn:
        name = "zotero"

        def authorize_url(self, *a, **kw):
            return ""

        async def exchange_code(self, code, redirect_uri):
            exchange_calls.append({"code": code, "redirect_uri": redirect_uri})
            return {
                "access_token": "api-key",
                "provider": "zotero",
                "display_name": "Zotero Account",
                "provider_user_id": "123456",
                "username": "alice",
            }

    async def _fake_create_account(db, user_id, provider, display_name, email, tokens):
        account_calls.append(
            {
                "user_id": user_id,
                "provider": provider,
                "display_name": display_name,
                "email": email,
                "tokens": dict(tokens),
            }
        )
        return {"id": 456, "display_name": display_name, "email": email, "created_at": "now"}

    monkeypatch.setattr(ep, "consume_oauth_state", _fake_consume_oauth_state)
    monkeypatch.setattr(ep, "get_connector_by_name", lambda provider: _FakeConn())
    monkeypatch.setattr(ep, "create_account", _fake_create_account)
    monkeypatch.setenv("ORG_CONNECTORS_ACCOUNT_LINKING_ROLE", "member")

    response = client.get(
        "/api/v1/connectors/providers/zotero/callback",
        params={"oauth_token": "temp-token", "oauth_verifier": "verifier-1", "state": "good"},
        headers=headers,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["id"] == 456
    assert body["provider"] == "zotero"
    assert exchange_calls[0]["code"] == (
        '{"oauth_token": "temp-token", "oauth_token_secret": "temp-secret", "oauth_verifier": "verifier-1"}'
    )
    assert account_calls[0]["tokens"]["provider_user_id"] == "123456"


@pytest.mark.integration
def test_browse_provider_sources_returns_zotero_collection_rows(connectors_client, monkeypatch):
    client, headers = connectors_client

    import tldw_Server_API.app.api.v1.endpoints.connectors as ep

    class _FakeConn:
        name = "zotero"

        def authorize_url(self, *a, **kw):
            return ""

        async def exchange_code(self, *a, **kw):
            return {}

        async def list_collections(self, account, *, cursor=None, page_size=100):
            assert account["provider_user_id"] == "123456"
            assert account["tokens"]["access_token"] == "api-key"
            return (
                [
                    {
                        "provider": "zotero",
                        "collection_key": "COLL1234",
                        "collection_name": "Language Models",
                        "source_url": "https://www.zotero.org/users/123456/collections/COLL1234",
                    }
                ],
                "next-zotero-cursor",
            )

    async def _fake_get_account_tokens(db, user_id, account_id):
        assert account_id == 15
        return {"access_token": "api-key", "provider_user_id": "123456"}

    async def _fake_get_account_email(db, user_id, account_id):
        return None

    async def _fake_get_account_for_user(db, user_id, account_id):
        assert account_id == 15
        return {"id": account_id, "user_id": user_id, "provider": "zotero"}

    monkeypatch.setattr(ep, "get_connector_by_name", lambda provider: _FakeConn())
    monkeypatch.setattr(ep, "get_account_for_user", _fake_get_account_for_user)
    monkeypatch.setattr(ep, "get_account_tokens", _fake_get_account_tokens)
    monkeypatch.setattr(ep, "get_account_email", _fake_get_account_email)

    response = client.get(
        "/api/v1/connectors/providers/zotero/sources/browse",
        params={"account_id": 15, "page_size": 25},
        headers=headers,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["next_cursor"] == "next-zotero-cursor"
    assert body["items"][0]["collection_key"] == "COLL1234"
    assert body["items"][0]["collection_name"] == "Language Models"


@pytest.mark.integration
def test_browse_provider_sources_includes_canonical_cursor_pagination(connectors_client, monkeypatch) -> None:
    client, headers = connectors_client

    import tldw_Server_API.app.api.v1.endpoints.connectors as ep

    class _FakeConn:
        async def list_collections(
            self,
            account: object,
            *,
            cursor: str | None = None,
            page_size: int = 100,
        ) -> tuple[list[dict[str, str]], str]:
            assert cursor == "zotero-cursor"
            assert page_size == 25
            return ([{"provider": "zotero", "collection_key": "COLL1234"}], "next-zotero-cursor")

    async def _fake_get_account_tokens(db: object, user_id: object, account_id: int) -> dict[str, str]:
        assert account_id == 15
        return {"access_token": "api-key", "provider_user_id": "123456"}

    async def _fake_get_account_email(db: object, user_id: object, account_id: int) -> None:
        return None

    async def _fake_get_account_for_user(db: object, user_id: object, account_id: int) -> dict[str, object]:
        return {"id": account_id, "user_id": user_id, "provider": "zotero"}

    monkeypatch.setattr(ep, "get_connector_by_name", lambda provider: _FakeConn())
    monkeypatch.setattr(ep, "get_account_for_user", _fake_get_account_for_user)
    monkeypatch.setattr(ep, "get_account_tokens", _fake_get_account_tokens)
    monkeypatch.setattr(ep, "get_account_email", _fake_get_account_email)

    response = client.get(
        "/api/v1/connectors/providers/zotero/sources/browse",
        params={"account_id": 15, "page_size": 25, "cursor": "zotero-cursor"},
        headers=headers,
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["next_cursor"] == "next-zotero-cursor"
    assert body["limit"] == 25
    assert body["cursor"] == "zotero-cursor"
    assert body["has_more"] is True
    assert body["pagination"] == {
        "mode": "cursor",
        "limit": 25,
        "cursor": "zotero-cursor",
        "next_cursor": "next-zotero-cursor",
        "has_more": True,
    }


@pytest.mark.integration
def test_browse_provider_sources_backfills_blank_zotero_provider_user_id(connectors_client, monkeypatch):
    client, headers = connectors_client

    import tldw_Server_API.app.api.v1.endpoints.connectors as ep

    class _FakeConn:
        name = "zotero"

        def authorize_url(self, *a, **kw):
            return ""

        async def exchange_code(self, *a, **kw):
            return {}

        async def list_collections(self, account, *, cursor=None, page_size=100):
            assert account["provider_user_id"] == "123456"
            return ([], None)

    async def _fake_get_account_tokens(db, user_id, account_id):
        assert account_id == 15
        return {"access_token": "api-key", "provider_user_id": "123456"}

    async def _fake_get_account_email(db, user_id, account_id):
        return None

    async def _fake_get_account_for_user(db, user_id, account_id):
        return {
            "id": account_id,
            "user_id": user_id,
            "provider": "zotero",
            "provider_user_id": "",
        }

    monkeypatch.setattr(ep, "get_connector_by_name", lambda provider: _FakeConn())
    monkeypatch.setattr(ep, "get_account_tokens", _fake_get_account_tokens)
    monkeypatch.setattr(ep, "get_account_email", _fake_get_account_email)
    monkeypatch.setattr(ep, "get_account_for_user", _fake_get_account_for_user)

    response = client.get(
        "/api/v1/connectors/providers/zotero/sources/browse",
        params={"account_id": 15, "page_size": 25},
        headers=headers,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["items"] == []
    assert body["next_cursor"] is None


@pytest.mark.integration
def test_browse_provider_sources_sanitizes_connector_error(connectors_client, monkeypatch):
    client, headers = connectors_client

    import tldw_Server_API.app.api.v1.endpoints.connectors as ep

    class _FailingConn:
        async def list_collections(self, account, *, cursor=None, page_size=100):
            raise RuntimeError("connector backend exploded")

    async def _fake_get_account_tokens(db, user_id, account_id):
        assert account_id == 15
        return {"access_token": "api-key", "provider_user_id": "123456"}

    async def _fake_get_account_email(db, user_id, account_id):
        return None

    async def _fake_get_account_for_user(db, user_id, account_id):
        return {"id": account_id, "user_id": user_id, "provider": "zotero"}

    monkeypatch.setattr(ep, "get_connector_by_name", lambda provider: _FailingConn())
    monkeypatch.setattr(ep, "get_account_tokens", _fake_get_account_tokens)
    monkeypatch.setattr(ep, "get_account_email", _fake_get_account_email)
    monkeypatch.setattr(ep, "get_account_for_user", _fake_get_account_for_user)

    response = client.get(
        "/api/v1/connectors/providers/zotero/sources/browse",
        params={"account_id": 15, "page_size": 25},
        headers=headers,
    )

    assert response.status_code == 502
    assert response.json()["detail"] == "Browse failed"


@pytest.mark.integration
def test_browse_provider_sources_rejects_account_provider_mismatch(connectors_client, monkeypatch):
    client, headers = connectors_client

    import tldw_Server_API.app.api.v1.endpoints.connectors as ep

    async def _fake_get_account_tokens(db, user_id, account_id):
        return {"access_token": "api-key", "provider_user_id": "123456"}

    async def _fake_get_account_for_user(db, user_id, account_id):
        return {"id": account_id, "user_id": user_id, "provider": "drive"}

    monkeypatch.setattr(ep, "get_account_tokens", _fake_get_account_tokens)
    monkeypatch.setattr(ep, "get_account_for_user", _fake_get_account_for_user)

    response = client.get(
        "/api/v1/connectors/providers/zotero/sources/browse",
        params={"account_id": 15, "page_size": 25},
        headers=headers,
    )
    assert response.status_code == 400, response.text
    assert response.json()["detail"] == "Account provider mismatch"


@pytest.mark.integration
def test_add_zotero_source_accepts_collection_type(connectors_client, monkeypatch):
    client, headers = connectors_client

    import tldw_Server_API.app.api.v1.endpoints.connectors as ep

    async def _fake_create_source(db, *, account_id, provider, remote_id, type_, path, options, enabled=True):
        return {
            "id": 202,
            "account_id": account_id,
            "provider": provider,
            "remote_id": remote_id,
            "type": type_,
            "path": path,
            "options": options or {},
            "enabled": enabled,
            "last_synced_at": None,
        }

    async def _fake_get_account_for_user(db, user_id, account_id):
        return {"id": account_id, "user_id": user_id, "provider": "zotero"}

    monkeypatch.setattr(ep, "create_source", _fake_create_source)
    monkeypatch.setattr(ep, "get_account_for_user", _fake_get_account_for_user)

    payload = {
        "account_id": 19,
        "provider": "zotero",
        "remote_id": "COLL1234",
        "type": "collection",
        "options": {},
    }
    response = client.post("/api/v1/connectors/sources", json=payload, headers=headers)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["provider"] == "zotero"
    assert body["type"] == "collection"


@pytest.mark.integration
def test_add_zotero_source_rejects_recursive_option(connectors_client, monkeypatch):
    client, headers = connectors_client

    import tldw_Server_API.app.api.v1.endpoints.connectors as ep

    async def _fake_get_account_for_user(db, user_id, account_id):
        return {"id": account_id, "user_id": user_id, "provider": "zotero"}

    monkeypatch.setattr(ep, "get_account_for_user", _fake_get_account_for_user)

    payload = {
        "account_id": 19,
        "provider": "zotero",
        "remote_id": "COLL1234",
        "type": "collection",
        "options": {"recursive": True},
    }
    response = client.post("/api/v1/connectors/sources", json=payload, headers=headers)
    assert response.status_code == 422, response.text
    assert response.json()["detail"] == "Zotero collection sync is flat in v1; link child collections separately."


@pytest.mark.integration
def test_patch_zotero_source_rejects_recursive_option(connectors_client, monkeypatch):
    client, headers = connectors_client

    import tldw_Server_API.app.api.v1.endpoints.connectors as ep

    async def _fake_get_source_by_id(db, user_id, source_id):
        return {
            "id": source_id,
            "account_id": 19,
            "provider": "zotero",
            "remote_id": "COLL1234",
            "type": "collection",
            "path": None,
            "options": {"recursive": False},
            "enabled": True,
            "last_synced_at": None,
        }

    monkeypatch.setattr(ep, "get_source_by_id", _fake_get_source_by_id)

    response = client.patch(
        "/api/v1/connectors/sources/19",
        json={"options": {"recursive": True}},
        headers=headers,
    )
    assert response.status_code == 422, response.text
    assert response.json()["detail"] == "Zotero collection sync is flat in v1; link child collections separately."


@pytest.mark.integration
def test_patch_zotero_source_preserves_flat_recursive_false(connectors_client, monkeypatch):
    client, headers = connectors_client

    import tldw_Server_API.app.api.v1.endpoints.connectors as ep

    update_calls: list[dict] = []

    async def _fake_get_source_by_id(db, user_id, source_id):
        return {
            "id": source_id,
            "account_id": 19,
            "provider": "zotero",
            "remote_id": "COLL1234",
            "type": "collection",
            "path": None,
            "options": {"recursive": False},
            "enabled": True,
            "last_synced_at": None,
        }

    async def _fake_update_source(db, user_id, source_id, *, enabled=None, options=None):
        update_calls.append({"source_id": source_id, "enabled": enabled, "options": dict(options or {})})
        return {
            "id": source_id,
            "account_id": 19,
            "provider": "zotero",
            "remote_id": "COLL1234",
            "type": "collection",
            "path": None,
            "options": dict(options or {}),
            "enabled": True,
            "last_synced_at": None,
        }

    monkeypatch.setattr(ep, "get_source_by_id", _fake_get_source_by_id)
    monkeypatch.setattr(ep, "update_source", _fake_update_source)

    response = client.patch(
        "/api/v1/connectors/sources/19",
        json={"options": {}},
        headers=headers,
    )
    assert response.status_code == 200, response.text
    assert update_calls[0]["options"]["recursive"] is False
    assert response.json()["options"]["recursive"] is False


@pytest.mark.integration
def test_get_source_sync_status_returns_state_and_active_job(connectors_client, monkeypatch):
    client, headers = connectors_client

    import tldw_Server_API.app.api.v1.endpoints.connectors as ep
    import tldw_Server_API.app.core.Jobs.manager as jobs_manager

    async def _fake_get_source_by_id(db, user_id, source_id):
        return {
            "id": source_id,
            "provider": "drive",
            "enabled": True,
        }

    async def _fake_get_source_sync_state(db, *, source_id):
        return {
            "source_id": source_id,
            "sync_mode": "hybrid",
            "cursor": "delta-1",
            "cursor_kind": "drive_start_page_token",
            "last_sync_started_at": "2026-03-06 10:00:00",
            "last_sync_succeeded_at": "2026-03-06 10:01:00",
            "last_error": None,
            "retry_backoff_count": 0,
            "webhook_status": "active",
            "webhook_expires_at": "2026-03-07 10:00:00",
            "needs_full_rescan": False,
            "active_job_id": "77",
            "active_job_started_at": "2026-03-06 10:00:00",
        }

    class _FakeJobManager:
        def get_job(self, job_id: int):
            assert job_id == 77
            return {
                "id": job_id,
                "job_type": "import",
                "status": "processing",
                "progress_percent": 35,
                "result": {"processed": 7, "failed": 0, "skipped": 0},
            }

    async def _fake_get_source_binding_health(db, *, source_id):
        assert source_id == 22
        return {
            "tracked_item_count": 9,
            "degraded_item_count": 3,
            "duplicate_count": 2,
            "metadata_only_count": 5,
        }

    monkeypatch.setattr(ep, "get_source_by_id", _fake_get_source_by_id)
    monkeypatch.setattr(ep, "get_source_sync_state", _fake_get_source_sync_state)
    monkeypatch.setattr(ep, "get_source_binding_health", _fake_get_source_binding_health)
    monkeypatch.setattr(jobs_manager, "JobManager", _FakeJobManager)

    response = client.get("/api/v1/connectors/sources/22/sync", headers=headers)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["source_id"] == 22
    assert body["provider"] == "drive"
    assert body["state"] == "running"
    assert body["sync_mode"] == "hybrid"
    assert body["cursor"] == "delta-1"
    assert body["active_job_id"] == "77"
    assert body["active_job"]["id"] == "77"
    assert body["active_job"]["status"] == "processing"
    assert body["active_job"]["progress_pct"] == 35
    assert body["tracked_item_count"] == 9
    assert body["degraded_item_count"] == 3
    assert body["duplicate_count"] == 2
    assert body["metadata_only_count"] == 5


@pytest.mark.integration
def test_get_source_sync_status_prefers_active_job_state_over_needs_full_rescan(connectors_client, monkeypatch):
    client, headers = connectors_client

    import tldw_Server_API.app.api.v1.endpoints.connectors as ep
    import tldw_Server_API.app.core.Jobs.manager as jobs_manager

    async def _fake_get_source_by_id(db, user_id, source_id):
        return {
            "id": source_id,
            "provider": "zotero",
            "enabled": True,
        }

    async def _fake_get_source_sync_state(db, *, source_id):
        return {
            "source_id": source_id,
            "sync_mode": "poll",
            "needs_full_rescan": True,
            "active_job_id": "88",
            "active_job_started_at": "2026-03-30 20:00:00",
        }

    async def _fake_get_source_binding_health(db, *, source_id):
        return {
            "tracked_item_count": 0,
            "degraded_item_count": 0,
            "duplicate_count": 0,
            "metadata_only_count": 0,
        }

    class _FakeJobManager:
        def get_job(self, job_id: int):
            assert job_id == 88
            return {
                "id": job_id,
                "job_type": "incremental_sync",
                "status": "processing",
                "progress_percent": 10,
                "result": {"processed": 1, "failed": 0, "skipped": 0},
            }

    monkeypatch.setattr(ep, "get_source_by_id", _fake_get_source_by_id)
    monkeypatch.setattr(ep, "get_source_sync_state", _fake_get_source_sync_state)
    monkeypatch.setattr(ep, "get_source_binding_health", _fake_get_source_binding_health)
    monkeypatch.setattr(jobs_manager, "JobManager", _FakeJobManager)

    response = client.get("/api/v1/connectors/sources/88/sync", headers=headers)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["state"] == "running"
    assert body["needs_full_rescan"] is True
    assert body["active_job"]["id"] == "88"


@pytest.mark.integration
def test_trigger_source_sync_endpoint_queues_job(connectors_client, monkeypatch):
    client, headers = connectors_client

    import tldw_Server_API.app.api.v1.endpoints.connectors as ep

    queued_requests: list[dict] = []

    async def _fake_get_source_by_id(db, user_id, source_id):
        return {
            "id": source_id,
            "provider": "onedrive",
            "enabled": True,
        }

    async def _fake_get_source_sync_state(db, *, source_id):
        return {}

    async def _fake_create_import_job(user_id, source_id, *, request_id=None, job_type="import"):
        queued_requests.append(
            {
                "user_id": user_id,
                "source_id": source_id,
                "request_id": request_id,
                "job_type": job_type,
            }
        )
        return {
            "id": "job-123",
            "source_id": source_id,
            "type": job_type,
            "status": "queued",
            "progress_pct": 0,
            "counts": {"processed": 0, "skipped": 0, "failed": 0},
        }

    monkeypatch.setattr(ep, "get_source_by_id", _fake_get_source_by_id)
    monkeypatch.setattr(ep, "get_source_sync_state", _fake_get_source_sync_state)
    monkeypatch.setattr(ep, "create_import_job", _fake_create_import_job)

    response = client.post("/api/v1/connectors/sources/22/sync", headers=headers)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["source_id"] == 22
    assert body["provider"] == "onedrive"
    assert body["status"] == "queued"
    assert body["job"]["id"] == "job-123"
    assert body["job"]["type"] == "incremental_sync"
    assert queued_requests[0]["job_type"] == "incremental_sync"


@pytest.mark.integration
def test_trigger_source_sync_endpoint_falls_back_to_import_for_non_file_provider(connectors_client, monkeypatch):
    client, headers = connectors_client

    import tldw_Server_API.app.api.v1.endpoints.connectors as ep

    queued_requests: list[dict] = []

    async def _fake_get_source_by_id(db, user_id, source_id):
        return {
            "id": source_id,
            "provider": "notion",
            "enabled": True,
        }

    async def _fake_get_source_sync_state(db, *, source_id):
        return {}

    async def _fake_create_import_job(user_id, source_id, *, request_id=None, job_type="import"):
        queued_requests.append(
            {
                "user_id": user_id,
                "source_id": source_id,
                "request_id": request_id,
                "job_type": job_type,
            }
        )
        return {
            "id": "job-456",
            "source_id": source_id,
            "type": job_type,
            "status": "queued",
            "progress_pct": 0,
            "counts": {"processed": 0, "skipped": 0, "failed": 0},
        }

    monkeypatch.setattr(ep, "get_source_by_id", _fake_get_source_by_id)
    monkeypatch.setattr(ep, "get_source_sync_state", _fake_get_source_sync_state)
    monkeypatch.setattr(ep, "create_import_job", _fake_create_import_job)

    response = client.post("/api/v1/connectors/sources/41/sync", headers=headers)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["provider"] == "notion"
    assert body["job"]["type"] == "import"
    assert queued_requests[0]["job_type"] == "import"


@pytest.mark.integration
def test_trigger_source_sync_endpoint_uses_repair_rescan_when_source_requires_it(connectors_client, monkeypatch):
    client, headers = connectors_client

    import tldw_Server_API.app.api.v1.endpoints.connectors as ep

    queued_requests: list[dict] = []

    async def _fake_get_source_by_id(db, user_id, source_id):
        return {
            "id": source_id,
            "provider": "drive",
            "enabled": True,
        }

    async def _fake_get_source_sync_state(db, *, source_id):
        return {"needs_full_rescan": True}

    async def _fake_create_import_job(user_id, source_id, *, request_id=None, job_type="import"):
        queued_requests.append(
            {
                "user_id": user_id,
                "source_id": source_id,
                "request_id": request_id,
                "job_type": job_type,
            }
        )
        return {
            "id": "job-789",
            "source_id": source_id,
            "type": job_type,
            "status": "queued",
            "progress_pct": 0,
            "counts": {"processed": 0, "skipped": 0, "failed": 0},
        }

    monkeypatch.setattr(ep, "get_source_by_id", _fake_get_source_by_id)
    monkeypatch.setattr(ep, "get_source_sync_state", _fake_get_source_sync_state)
    monkeypatch.setattr(ep, "create_import_job", _fake_create_import_job)

    response = client.post("/api/v1/connectors/sources/52/sync", headers=headers)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["provider"] == "drive"
    assert body["job"]["type"] == "repair_rescan"
    assert queued_requests[0]["job_type"] == "repair_rescan"
